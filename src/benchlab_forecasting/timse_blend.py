"""A TIMSE-oriented forecast, chosen to be robust to how the official TIMSE is aggregated.

The ranking metric is TIMSE, but the published note does not say over which set it is computed:
the whole live period, windows of it, or each issuance. This experiment:

1. trains the tuned LightGBM hybrid again with TIMSE sample weights, choosing the number of trees by
   validation TIMSE (train 2011–2021, validation 2022–2023);
2. forms candidates (1 − β) · submitted forecast + β · TIMSE model + δ on the validation years;
3. scores every candidate under several plausible aggregations and keeps the one with the smallest
   worst-case ratio to the best candidate of each aggregation (minimax regret);
4. scores the test period (Jan 2024 – Oct 2026) once.

    .venv/bin/python -m benchlab_forecasting.timse_blend --output-dir data/experiments/timse_blend_2011_2026
"""

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .validation.metrics import timse

HORIZON = 72


def timse_quantiles(y, p, m):
    """The note's formula with m thresholds at quantiles of the observations (left rectangles)."""
    thresholds = np.unique(np.quantile(y, np.linspace(0, 1, m)))
    total = 0.0
    for low, high in zip(thresholds[:-1], thresholds[1:]):
        tail = y >= low
        total += (high - low) * np.mean((y[tail] - p[tail]) ** 2)
    return total


def _grouped(y, p, splits, score, reduce):
    return float(reduce([score(y[idx], p[idx]) for idx in splits if len(idx) > 1]))


AGGREGATIONS = {
    "pooled, unique values": lambda y, p, s: timse(y, p),
    "pooled, 20 quantiles": lambda y, p, s: timse_quantiles(y, p, 20),
    "90-day windows, unique values (mean)": lambda y, p, s: _grouped(y, p, s["window"], timse, np.mean),
    "90-day windows, 20 quantiles (mean)": lambda y, p, s: _grouped(
        y, p, s["window"], lambda a, b: timse_quantiles(a, b, 20), np.mean),
    "per month (mean)": lambda y, p, s: _grouped(y, p, s["month"], timse, np.mean),
    "per issuance (mean)": lambda y, p, s: _grouped(y, p, s["issuance"], timse, np.mean),
    "per issuance (median)": lambda y, p, s: _grouped(y, p, s["issuance"], timse, np.median),
}


def groups_of(origins, timestamps):
    origins = pd.DatetimeIndex(origins)
    days = (origins - origins.min()) / pd.Timedelta(days=1)
    return {"window": (days // 90).astype(int).to_numpy(),
            "month": (pd.DatetimeIndex(timestamps).year * 12 + pd.DatetimeIndex(timestamps).month).to_numpy(),
            "issuance": origins.asi8}


def prepare(y, groups):
    """Observed pairs and, for each grouping, the index arrays of its groups (computed once)."""
    observed = np.isfinite(y)
    splits = {}
    for key, labels in groups.items():
        labels = labels[observed]
        order = np.argsort(labels, kind="stable")
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        splits[key] = np.split(order, bounds)
    return {"observed": observed, "y": y[observed], "splits": splits}


def score_all(prepared, p):
    q = prepared
    return {name: f(q["y"], p[q["observed"]], q["splits"]) for name, f in AGGREGATIONS.items()}


def rmse(y, p):
    m = np.isfinite(y)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def minimax_choice(table, aggregations):
    """Candidate with the smallest worst-case ratio to the best candidate of each aggregation."""
    ratios = table[aggregations] / table[aggregations].min()
    worst = ratios.max(1)
    return int(worst.idxmin()), float(worst.min())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level2", type=Path, default=Path("data/ace_level2/ac_h2_swe_hourly.parquet"))
    parser.add_argument("--operational", type=Path, default=Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet"))
    parser.add_argument("--coronal-holes", type=Path, default=Path("data/features/coronal_holes_0193_6h.parquet"))
    parser.add_argument("--tuning-report", type=Path, default=Path("data/experiments/hybrid_lightgbm_tuning/report.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("submission/model"))
    parser.add_argument("--train", nargs=2, default=["2011-02-01", "2021-12-28"])
    parser.add_argument("--validation", nargs=2, default=["2022-01-01", "2023-12-28"])
    parser.add_argument("--test", nargs=2, default=["2024-01-01", "2026-10-23T00:00"])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from .hybrid import HybridConfig, HybridForecaster, build_pairs, load_plasma, origins_between
    from .som import base_forecasts

    start = time.monotonic()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plasma = load_plasma(args.level2, args.operational)
    holes = pd.read_parquet(args.coronal_holes)
    holes = holes[holes.error.isna()].copy()
    holes["origin_utc"] = pd.to_datetime(holes.origin_utc, utc=True)
    splits = {k: origins_between(*getattr(args, k)) for k in ("train", "validation", "test")}

    # 1. Tuned hyperparameters, TIMSE sample weights, trees chosen by validation TIMSE.
    tuned = json.loads(args.tuning_report.read_text())["models"]["tuned"]["config"]
    known = {f.name for f in fields(HybridConfig)}
    config = HybridConfig(**{**{k: v for k, v in tuned.items() if k in known},
                             "weighting": "timse", "selection_metric": "timse", "threads": args.threads})
    tables = {k: build_pairs(plasma, holes, v, config) for k, v in splits.items()}
    model = HybridForecaster(config).fit(tables["train"], tables["validation"])
    model.save(out / "model_timse_tuned.pkl")
    model.model.booster_.save_model(str(out / "timse_tuned.txt"), num_iteration=model.iterations)
    print(f"TIMSE model: {model.iterations} trees, {model.training['seconds']:.0f} s", flush=True)

    data = {}
    for name in ("validation", "test"):
        submitted, y = base_forecasts(plasma, holes, splits[name], args.model_dir)
        t = tables[name]
        data[name] = {"y": y.ravel(), "submitted": submitted.ravel(), "timse": model.predict(t),
                      "groups": prepare(y.ravel(), groups_of(t.origin_utc, t.timestamp_utc)),
                      "origin": t.origin_utc.to_numpy(),
                      "horizon": t.horizon.to_numpy()}

    # 2–3. Candidates on validation, scored under every aggregation.
    betas = np.round(np.linspace(0, 1, 11), 2)
    deltas = np.arange(0, 110, 10)
    val = data["validation"]
    rows = []
    for beta in betas:
        for delta in deltas:
            p = (1 - beta) * val["submitted"] + beta * val["timse"] + delta
            rows.append({"beta": beta, "delta": int(delta), "rmse": rmse(val["y"], p),
                         **score_all(val["groups"], p)})
        print(f"validation: β = {beta} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_candidates.csv", index=False)
    names = list(AGGREGATIONS)
    without_median = [n for n in names if n != "per issuance (median)"]
    choices = {}
    for label, aggs in (("robust, all aggregations", names), ("robust, without per-issuance median", without_median)):
        i, worst = minimax_choice(table, aggs)
        choices[label] = {"beta": float(table.beta[i]), "delta": int(table.delta[i]), "worst_ratio_on_validation": worst}
    for name in names:
        i = int(table[name].idxmin())
        choices[f"best for {name}"] = {"beta": float(table.beta[i]), "delta": int(table.delta[i])}

    # 4. Test, once, for the submitted forecast, the TIMSE model alone and the chosen candidates.
    test = data["test"]
    candidates = {"submitted (RMSE-oriented)": (0.0, 0), "TIMSE model alone": (1.0, 0)}
    for label in ("robust, all aggregations", "robust, without per-issuance median"):
        candidates[label] = (choices[label]["beta"], choices[label]["delta"])
    results = {}
    for label, (beta, delta) in candidates.items():
        p = (1 - beta) * test["submitted"] + beta * test["timse"] + delta
        results[label] = {"beta": beta, "delta": delta, "rmse": rmse(test["y"], p),
                          **score_all(test["groups"], p)}
    constant = np.full_like(test["y"], 700.0)
    results["constant 700 km/s (reference)"] = {"rmse": rmse(test["y"], constant),
                                                **score_all(test["groups"], constant)}
    pd.DataFrame({"origin_utc": test["origin"], "horizon": test["horizon"], "y": test["y"],
                  "submitted": test["submitted"], "timse_model": test["timse"]}).to_parquet(
        out / "test_predictions.parquet", index=False)
    report = {"timse_model": {"config": asdict(config), "iterations": model.iterations,
                              "seconds": model.training["seconds"]},
              "candidates_grid": {"beta": betas.tolist(), "delta": deltas.tolist()},
              "choices_on_validation": choices, "test": results, "seconds": time.monotonic() - start}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"choices": choices, "test": results}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
