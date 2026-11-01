"""Bounded Prophet search with causal regressors and an untouched final period."""

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data import load_series, utc_hour
from .features import load_regressors
from .models import ForecastConfig, ProphetForecaster, baseline, future_grid
from .validation.reporting import evaluate_predictions


def default_candidates(base):
    """Small, deterministic search; includes ablations and the existing model."""
    candidates = [{"name": "prophet_current", "group": "none", "config": base}]

    def add(name, group="none", **overrides):
        candidates.append({"name": name, "group": group, "config": replace(base, **overrides)})

    for order in (1, 3, 5, 8):
        add(f"fourier_{order}", rotation_days=27, fourier_order=order, seasonality_prior_scale=1)
    for group in ("plasma", "plasma_mag"):
        add(f"{group}_only", group)
    for order in (1, 3, 5):
        for prior in (0.1, 1.0):
            add(f"mag_fourier_{order}_prior_{prior:g}", "plasma_mag", rotation_days=27,
                fourier_order=order, seasonality_prior_scale=1, regressor_prior_scale=prior)
    for lag in (96, 648):
        add(f"plasma_fourier_lag_{lag}", "plasma", rotation_days=27,
            fourier_order=3, seasonality_prior_scale=1, exogenous_lag_hours=lag)
    for period in (26, 28):
        add(f"mag_period_{period}", "plasma_mag", rotation_days=period,
            fourier_order=3, seasonality_prior_scale=1)
    for prior in (0.01, 0.05):
        add(f"mag_linear_{prior:g}", "plasma_mag", rotation_days=27, fourier_order=3,
            seasonality_prior_scale=1, growth="linear", changepoint_prior_scale=prior,
            train_days=90)
    add("mag_strong_regularization", "plasma_mag", rotation_days=27, fourier_order=3,
        seasonality_prior_scale=0.1, regressor_prior_scale=0.1)
    return candidates


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def forecast_table(series, origins, horizon):
    labels = series.set_index("ds").y
    blocks = []
    for origin in origins:
        dates = future_grid(origin, horizon)
        blocks.append(pd.DataFrame({"origin_utc": origin, "timestamp_utc": dates,
                                    "horizon": np.arange(1, horizon + 1),
                                    "y": labels.reindex(dates).to_numpy()}))
    return pd.concat(blocks, ignore_index=True)


def predict_candidate(series, inputs, origins, candidate):
    values, failures, timings = [], [], []
    model = None
    for origin in origins:
        start = time.monotonic()
        try:
            model = ProphetForecaster(candidate["config"]).fit(
                series, origin, inputs[candidate["group"]])
            values.extend(model.predict().speed_km_s)
        except Exception as exc:
            failures.append({"origin_utc": origin.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            values.extend([np.nan] * candidate["config"].horizon)
            model = None
        timings.append(time.monotonic() - start)
    return values, failures, timings, model


def check_schedule(series, folds, test_origins, horizon):
    all_validation = pd.DatetimeIndex([t for fold in folds for t in fold])
    if all_validation.empty or not all_validation.is_unique or test_origins.empty:
        raise ValueError("Validation folds must be nonempty and must not overlap")
    if all_validation.max() + pd.Timedelta(hours=horizon) >= test_origins.min():
        raise ValueError("Validation targets must end before the final test begins")
    blocks = sorted(folds, key=lambda x: x.min())
    if any(a.max() + pd.Timedelta(hours=horizon) >= b.min() for a, b in zip(blocks, blocks[1:])):
        raise ValueError("Validation target intervals must not overlap between folds")
    if min(all_validation.min(), test_origins.min()) <= series.ds.min():
        raise ValueError("No training history before the evaluation origins")
    if test_origins.max() + pd.Timedelta(hours=horizon) > series.ds.max():
        raise ValueError("Final test plus horizon extends beyond the observations")
    return all_validation.sort_values()


def run_tuning(series, inputs, candidates, folds, test_origins, output, metadata=None):
    if not candidates or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidates must have unique names")
    reserved = {"origin_utc", "timestamp_utc", "horizon", "y", "persistence", "recurrence"}
    if any(c["name"] in reserved or c["group"] not in inputs for c in candidates):
        raise ValueError("Invalid candidate name or unavailable feature group")
    horizon = candidates[0]["config"].horizon
    if any(c["config"].horizon != horizon for c in candidates):
        raise ValueError("All candidates must use the same horizon")
    origins = check_schedule(series, folds, test_origins, horizon)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty experiment directory")
    manifest = {"selection_metric": "RMSE on common validation forecast-origin/lead pairs",
                "selection_rule": "Only candidates with complete finite forecasts are eligible",
                "horizon": horizon, "validation_origins": [t.isoformat() for t in origins],
                "test_origins": [t.isoformat() for t in test_origins],
                "candidates": [{**c, "config": asdict(c["config"])} for c in candidates],
                "metadata": metadata or {},
                "availability_note": "Historical publication latency is assumed unless source available_at exists",
                "final_test_note": "Sequential refits may use earlier test observations once available; configuration remains fixed"}
    write_json(output / "manifest.json", manifest)
    predictions = forecast_table(series, origins, horizon)
    diagnostics, eligible = {}, []
    for i, candidate in enumerate(candidates, 1):
        name = candidate["name"]
        values, failures, timings, _ = predict_candidate(series, inputs, origins, candidate)
        predictions[name] = values
        diagnostics[name] = {"failures": failures, "fit_seconds": timings,
                             "mean_fit_seconds": float(np.mean(timings)),
                             "prediction_coverage": float(np.isfinite(values).mean())}
        if np.isfinite(values).all():
            eligible.append(name)
        predictions.to_parquet(output / "validation_predictions.parquet", index=False)
        write_json(output / "diagnostics.json", diagnostics)
        print(f"[{i}/{len(candidates)}] {name}: {sum(timings):.1f}s, {len(failures)} failures", flush=True)
    if not eligible:
        raise ValueError("No candidate produced complete validation forecasts")
    report = evaluate_predictions(predictions, eligible, horizon=horizon, expected_origins=origins)
    if report["common_pairs"] == 0:
        raise ValueError("No observed validation targets")
    write_json(output / "validation_report.json", report)
    rows = []
    for candidate in candidates:
        name = candidate["name"]
        row = {"candidate": name, "group": candidate["group"], "eligible": name in eligible,
               **asdict(candidate["config"]), "mean_fit_seconds": diagnostics[name]["mean_fit_seconds"],
               "prediction_coverage": diagnostics[name]["prediction_coverage"]}
        if name in eligible:
            row.update(report["metrics"][name]["common"])
            for i, fold in enumerate(folds, 1):
                block = predictions[predictions.origin_utc.isin(fold)]
                mask = block.y.notna()
                error = block.loc[mask, name] - block.loc[mask, "y"]
                row[f"fold_{i}_rmse"] = float(np.sqrt(np.mean(error**2))) if len(error) else None
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values("rmse", na_position="last", kind="stable")
    ranking.to_csv(output / "leaderboard.csv", index=False)
    winner = next(c for c in candidates if c["name"] == ranking.iloc[0].candidate)
    # Persist selection BEFORE reading or scoring final test targets.
    write_json(output / "best_config.json", {**winner, "config": asdict(winner["config"]),
                                             "validation_rmse": float(ranking.iloc[0].rmse)})
    print(f"Selected on validation: {winner['name']}; RMSE={ranking.iloc[0].rmse:.3f} km/s", flush=True)
    test_dir = output / "test"
    test_dir.mkdir()
    test = forecast_table(series, test_origins, horizon)
    current = next((c for c in candidates if c["name"] == "prophet_current"), candidates[0])
    test_diagnostics = {}
    for name, candidate in (("prophet_current", current), ("prophet_tuned", winner)):
        values, failures, timings, model = predict_candidate(series, inputs, test_origins, candidate)
        test[name] = values
        test_diagnostics[name] = {"failures": failures, "fit_seconds": timings}
        if name == "prophet_tuned" and model is not None:
            model.save(test_dir / "model_last_origin.json")
    base = current["config"]
    for name in ("persistence", "recurrence"):
        values, failures = [], []
        for origin in test_origins:
            try:
                values.extend(baseline(series, origin, base, name).speed_km_s)
            except Exception as exc:
                values.extend([np.nan] * horizon)
                failures.append({"origin_utc": origin.isoformat(), "error": str(exc)})
        test[name] = values
        test_diagnostics[name] = {"failures": failures}
    test.to_parquet(test_dir / "predictions.parquet", index=False)
    test_report = evaluate_predictions(test, ["prophet_current", "prophet_tuned", "persistence", "recurrence"],
                                       horizon=horizon, expected_origins=test_origins)
    test_report.update({"config": {"horizon": horizon}, "diagnostics": test_diagnostics,
                        "selected_candidate": winner["name"],
                        "expected_origins": [t.isoformat() for t in test_origins]})
    write_json(test_dir / "report.json", test_report)
    # Also compare without recurrence's missing historical values removing targets.
    full_report = evaluate_predictions(test, ["prophet_current", "prophet_tuned", "persistence"],
                                       horizon=horizon, expected_origins=test_origins)
    write_json(test_dir / "comparison_without_recurrence.json", full_report)
    return winner, test_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mag-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-starts", nargs="+", required=True,
                        help="UTC starts for separate walk-forward blocks")
    parser.add_argument("--validation-hours", type=int, default=24)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--latency-minutes", type=int, default=60)
    parser.add_argument("--candidates", type=Path,
                        help="Optional JSON list of {name, group, config} overrides")
    args = parser.parse_args(argv)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    try:
        if min(args.validation_hours, args.test_hours, args.step_hours) < 1:
            raise ValueError("Block lengths and cadence must be positive")
        base = ForecastConfig(latency_minutes=args.latency_minutes)
        candidates = default_candidates(base)
        if args.candidates:
            candidates = [{**c, "config": replace(base, **c.get("config", {}))}
                          for c in json.loads(args.candidates.read_text())]
        series = load_series(args.data)
        inputs = {"none": None, "plasma": load_regressors(args.data),
                  "plasma_mag": load_regressors(args.data, args.mag_data, "plasma_mag")}

        def schedule(start, hours):
            start = utc_hour(start)
            return pd.date_range(start, start + pd.Timedelta(hours=hours),
                                 freq=f"{args.step_hours}h", inclusive="left")

        folds = [schedule(start, args.validation_hours) for start in args.validation_starts]
        test = schedule(args.test_start, args.test_hours)
        hashes = {}
        for path in (args.data, args.mag_data):
            with path.open("rb") as stream:
                hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob("*.py")}
        metadata = {"data_sha256": hashes, "source_sha256": source_hashes,
                    "versions": {name: version(name) for name in ("prophet", "pandas", "numpy")},
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        run_tuning(series, inputs, candidates, folds, test, args.output_dir, metadata)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
