"""Two linked self-organizing maps that post-process a 72-hour solar-wind forecast.

Map A learns the shapes of the solar-wind speed in the 72 hours before the cut-off
(t0 − 1 h), relative to the last value. Map B learns the shapes of the 72 hours that
follow t0, relative to the same value. Counting, on the training issuances, how often a
past shape (node i of A) is followed by a future shape (node j of B) gives P(j | i).

At forecast time the issuance's past window picks node i. The future prototypes are then
weighted by P(j | i) and by their distance to the model's forecast, so the model, which
sees the coronal holes, chooses the scenario and the map supplies its shape, peaks
included:

    w_j ∝ P(j | i) · exp(−d_j² / 2σ²),   d_j = RMS distance between forecast and prototype j
    final(h) = α(h) · forecast(h) + (1 − α(h)) · Σ_j w_j · prototype_j(h)

σ and α are chosen on the validation years; the test period is scored once.

Example:
  .venv/bin/python -m benchlab_forecasting.som experiment --output-dir data/experiments/som_2011_2026
"""

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

WINDOW = 72
HORIZON = 72


# ---------------------------------------------------------------- the map

@dataclass(frozen=True)
class SOMConfig:
    rows: int = 10
    cols: int = 10
    epochs: int = 40
    sigma_start: float | None = None  # neighbourhood radius in nodes; default: half the larger side
    sigma_end: float = 0.7
    seed: int = 0


class SOM:
    """Batch self-organizing map on a rectangular grid with a Gaussian neighbourhood.

    Batch training is deterministic: every epoch assigns each sample to its best-matching
    unit and replaces each prototype by the neighbourhood-weighted mean of the samples.
    Prototypes start on the plane of the two leading principal components.
    """

    def __init__(self, config=None):
        self.config = config or SOMConfig()
        c = self.config
        self.grid = np.array([(r, k) for r in range(c.rows) for k in range(c.cols)], dtype=float)
        self.prototypes = None

    @property
    def nodes(self):
        return len(self.grid)

    def _kernel(self, sigma):
        d2 = ((self.grid[:, None, :] - self.grid[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    def _init(self, data):
        mean = data.mean(0)
        _, s, vt = np.linalg.svd(data - mean, full_matrices=False)
        scale = s[:2] / np.sqrt(len(data))
        rows = np.linspace(-1, 1, self.config.rows) if self.config.rows > 1 else np.zeros(1)
        cols = np.linspace(-1, 1, self.config.cols) if self.config.cols > 1 else np.zeros(1)
        return np.array([mean + r * scale[0] * vt[0] + k * scale[1] * vt[1] for r in rows for k in cols])

    def _distances(self, data):
        data = np.asarray(data, dtype=float)
        return (data ** 2).sum(1)[:, None] - 2 * data @ self.prototypes.T + (self.prototypes ** 2).sum(1)[None, :]

    def bmu(self, data):
        return self._distances(data).argmin(1)

    def fit(self, data):
        data = np.asarray(data, dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("SOM training data must be finite")
        c = self.config
        self.prototypes = self._init(data)
        start = c.sigma_start or max(c.rows, c.cols) / 2
        for epoch in range(c.epochs):
            sigma = start * (c.sigma_end / start) ** (epoch / max(c.epochs - 1, 1))
            winners = self.bmu(data)
            counts = np.bincount(winners, minlength=self.nodes)
            sums = np.zeros_like(self.prototypes)
            np.add.at(sums, winners, data)
            h = self._kernel(sigma)
            denominator = h @ counts
            keep = denominator > 1e-12
            self.prototypes[keep] = (h @ sums)[keep] / denominator[keep, None]
        self.sigma_final = sigma
        return self

    def quantization_error(self, data):
        data = np.asarray(data, dtype=float)
        return float(np.sqrt(((data - self.prototypes[self.bmu(data)]) ** 2).mean()))

    def topographic_error(self, data):
        """Share of samples whose two closest prototypes are not grid neighbours."""
        first, second = np.argsort(self._distances(data), 1)[:, :2].T
        return float((np.abs(self.grid[first] - self.grid[second]).max(1) > 1).mean())


# ---------------------------------------------------------------- windows

def past_windows(speed, origins, latency_hours=1, window=WINDOW, min_valid=0.8):
    """Speed in the `window` hours up to the cut-off, minus the last valid value.

    Returns (shapes, last, ok): gaps are interpolated inside the window; issuances with
    fewer than `min_valid` valid hours or no valid value within 6 h of the cut-off are
    marked not ok.
    """
    origins = pd.DatetimeIndex(origins)
    cutoff = origins - pd.Timedelta(hours=latency_hours)
    offsets = np.arange(-window + 1, 1) * np.timedelta64(1, "h")
    times = pd.DatetimeIndex((cutoff.values[:, None] + offsets[None, :]).ravel(), tz="UTC")
    values = speed.reindex(times).to_numpy(float).reshape(len(origins), window)
    valid = np.isfinite(values)
    recent = valid[:, -6:].any(1)
    ok = (valid.mean(1) >= min_valid) & recent
    filled = pd.DataFrame(values.T).interpolate(limit_direction="both").to_numpy().T
    last = np.full(len(origins), np.nan)
    for i in np.flatnonzero(recent):
        last[i] = values[i, np.flatnonzero(valid[i])[-1]]
    return filled - last[:, None], last, ok & np.isfinite(filled).all(1)


def recurrence_windows(speed, origins, last, horizon=HORIZON, rotation_hours=27 * 24):
    """Speed one solar rotation before each of the next `horizon` hours, minus `last`.

    Known at t0 (it ends 27 days − 72 h before t0); gaps are interpolated, and rows with no
    data at all are filled with 0 (no change).
    """
    origins = pd.DatetimeIndex(origins)
    offsets = (np.arange(1, horizon + 1) - rotation_hours) * np.timedelta64(1, "h")
    times = pd.DatetimeIndex((origins.values[:, None] + offsets[None, :]).ravel(), tz="UTC")
    smooth = speed.rolling(7, center=True, min_periods=3).mean()
    values = smooth.reindex(times).to_numpy(float).reshape(len(origins), horizon)
    filled = pd.DataFrame(values.T).interpolate(limit_direction="both").to_numpy().T
    return np.nan_to_num(filled - last[:, None])


def future_windows(speed, origins, last, horizon=HORIZON, min_valid=0.9):
    """Observed speed at t0 + 1 … t0 + horizon minus `last`; ok when mostly observed."""
    origins = pd.DatetimeIndex(origins)
    offsets = np.arange(1, horizon + 1) * np.timedelta64(1, "h")
    times = pd.DatetimeIndex((origins.values[:, None] + offsets[None, :]).ravel(), tz="UTC")
    values = speed.reindex(times).to_numpy(float).reshape(len(origins), horizon)
    ok = np.isfinite(values).mean(1) >= min_valid
    filled = pd.DataFrame(values.T).interpolate(limit_direction="both").to_numpy().T
    return filled - last[:, None], ok & np.isfinite(filled).all(1) & np.isfinite(last)


# ---------------------------------------------------------------- link and post-processing

def link_matrix(past_map, future_map, past, future, smoothing=1.0, prior=1.0):
    """P(future node | past node), smoothed over past-map neighbours, with a marginal prior."""
    counts = np.zeros((past_map.nodes, future_map.nodes))
    np.add.at(counts, (past_map.bmu(past), future_map.bmu(future)), 1)
    if smoothing > 0:
        counts = past_map._kernel(smoothing) @ counts
    marginal = counts.sum(0) / counts.sum()
    counts = counts + prior * marginal[None, :]
    return counts / counts.sum(1, keepdims=True)


def scenario_weights(link, past_nodes, forecast_rel, prototypes, sigma):
    """w_j ∝ P(j | i) · exp(−d_j² / 2σ²) for each issuance (rows)."""
    prior = link[past_nodes]
    if sigma is None or not np.isfinite(sigma):
        return prior
    d2 = ((forecast_rel[:, None, :] - prototypes[None, :, :]) ** 2).mean(-1)
    logw = np.log(prior + 1e-300) - (d2 - d2.min(1, keepdims=True)) / (2 * sigma ** 2)
    w = np.exp(logw - logw.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


def alpha_curve(a_first, a_last, horizon=HORIZON):
    return np.linspace(a_first, a_last, horizon)


def post_process(forecast, last, weights, prototypes, alpha):
    """α(h) · forecast + (1 − α(h)) · (last + Σ_j w_j prototype_j); rows without a scenario keep the forecast."""
    scenario = last[:, None] + weights @ prototypes
    result = alpha[None, :] * forecast + (1 - alpha[None, :]) * scenario
    return np.where(np.isfinite(result), result, forecast)


# ---------------------------------------------------------------- evaluation

def rmse(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def lead_rmse(y, p, lo, hi):
    return rmse(y[:, lo - 1:hi], p[:, lo - 1:hi])


def rise_metrics(y, p, last, rise=100.0):
    """On issuances whose observed speed rises by at least `rise` km/s within 72 h:
    error of the peak height (forecast max − observed max), absolute error of the peak hour,
    and RMSE."""
    observed_peak = np.nanmax(y, 1)
    events = np.isfinite(last) & (observed_peak - last >= rise) & (np.isfinite(y).mean(1) >= 0.9)
    if not events.any():
        return {"issuances": 0}
    yy, pp = y[events], p[events]
    peak_hour_obs = np.nanargmax(yy, 1)
    peak_hour_fc = np.nanargmax(pp, 1)
    return {"issuances": int(events.sum()), "rmse": rmse(yy, pp),
            "peak_height_bias": float(np.nanmean(np.nanmax(pp, 1) - np.nanmax(yy, 1))),
            "peak_height_mae": float(np.nanmean(np.abs(np.nanmax(pp, 1) - np.nanmax(yy, 1)))),
            "peak_hour_mae": float(np.mean(np.abs(peak_hour_fc - peak_hour_obs)))}


def block_bootstrap_gain(origins, y, base, other, block_days=27, draws=2000, seed=0):
    """RMSE(base) − RMSE(other) with a 95% interval from resampled 27-day blocks."""
    origins = pd.DatetimeIndex(origins)
    blocks = ((origins - origins.min()) / pd.Timedelta(days=block_days)).astype(int).to_numpy()
    ids = np.unique(blocks)
    se_base = np.nansum((y - base) ** 2, 1)
    se_other = np.nansum((y - other) ** 2, 1)
    n = np.isfinite(y).sum(1)
    per_block = pd.DataFrame({"b": blocks, "sb": se_base, "so": se_other, "n": n}).groupby("b").sum()
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(draws):
        pick = per_block.loc[rng.choice(ids, len(ids))]
        gains.append(np.sqrt(pick.sb.sum() / pick.n.sum()) - np.sqrt(pick.so.sum() / pick.n.sum()))
    point = np.sqrt(per_block.sb.sum() / per_block.n.sum()) - np.sqrt(per_block.so.sum() / per_block.n.sum())
    low, high = np.percentile(gains, [2.5, 97.5])
    return {"gain_km_s": float(point), "ci95": [float(low), float(high)], "blocks": len(ids)}


def _plugin_mi(a, b):
    joint = np.array(pd.crosstab(a, b).to_numpy(float))
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())
    hb = float(-(pb[pb > 0] * np.log2(pb[pb > 0])).sum())
    return mi, hb


def mutual_information(a, b, shuffles=0, seed=0):
    """Mutual information (bits) between two node assignments and its share of H(b).

    The plug-in estimate is biased upwards when there are few samples per cell; with
    `shuffles` > 0 the mean MI of randomly re-paired assignments is reported and subtracted.
    """
    mi, hb = _plugin_mi(a, b)
    result = {"bits": mi, "share_of_future_entropy": mi / hb}
    if shuffles:
        rng = np.random.default_rng(seed)
        null = float(np.mean([_plugin_mi(a, rng.permutation(b))[0] for _ in range(shuffles)]))
        result.update(shuffled_bits=null, excess_bits=mi - null, excess_share=(mi - null) / hb)
    return result


# ---------------------------------------------------------------- experiment

def base_forecasts(plasma, coronal_holes, origins, model_dir):
    """The submitted forecast: tuned LightGBM blended with persistence (submission/model/tuned.*)."""
    import lightgbm

    from .hybrid import HybridConfig, build_pairs

    meta = json.loads((model_dir / "tuned.json").read_text())
    booster = lightgbm.Booster(model_file=str(model_dir / "tuned.txt"))
    frame = build_pairs(plasma, coronal_holes, origins, HybridConfig())
    prediction = booster.predict(frame[meta["features"]])
    persistence = frame.last_speed.to_numpy()
    recurrence = frame.recurrence_27d.fillna(frame.last_speed).to_numpy()
    prediction = np.where(np.isfinite(prediction), np.maximum(prediction, 0), recurrence)
    w = np.asarray(meta["persistence_weight_by_lead"])[frame.horizon.to_numpy() - 1]
    blended = np.where(np.isfinite(persistence), w * persistence + (1 - w) * prediction, prediction)
    return blended.reshape(len(origins), HORIZON), frame.y.to_numpy().reshape(len(origins), HORIZON)


def apply(maps, link, past, last, ok, forecast, sigma, alpha):
    past_map, future_map = maps
    nodes = past_map.bmu(np.where(ok[:, None], past, 0))
    weights = scenario_weights(link, nodes, forecast - last[:, None], future_map.prototypes, sigma)
    result = post_process(forecast, last, weights, future_map.prototypes, alpha)
    return np.where(ok[:, None], result, forecast), weights, nodes


def evaluate(y, p, last):
    return {"rmse": rmse(y, p), "rmse_1_24": lead_rmse(y, p, 1, 24), "rmse_25_48": lead_rmse(y, p, 25, 48),
            "rmse_49_72": lead_rmse(y, p, 49, 72), "rises_100": rise_metrics(y, p, last)}


def experiment(args):
    from .hybrid import load_plasma, origins_between

    start = time.monotonic()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plasma = load_plasma(args.level2, args.operational)
    speed = plasma.speed_km_s
    holes = pd.read_parquet(args.coronal_holes)
    holes = holes[holes.error.isna()].copy()
    holes["origin_utc"] = pd.to_datetime(holes.origin_utc, utc=True)
    splits = {"train": origins_between(*args.train), "validation": origins_between(*args.validation),
              "test": origins_between(*args.test)}
    data = {}
    for name, origins in splits.items():
        past, last, ok_past = past_windows(speed, origins)
        if args.with_recurrence:
            # Same km/s units in both halves, so neither dominates the distances by scale.
            past = np.hstack([past, recurrence_windows(speed, origins, last)])
        future, ok_future = future_windows(speed, origins, last)
        data[name] = {"origins": origins, "past": past, "last": last, "ok": ok_past, "future": future,
                      "ok_future": ok_future}
    for name in ("validation", "test"):
        data[name]["forecast"], data[name]["y"] = base_forecasts(plasma, holes, splits[name], args.model_dir)
    train = data["train"]
    usable = train["ok"] & train["ok_future"]
    print(f"train issuances: {usable.sum()} of {len(usable)} with complete past and future windows", flush=True)

    # Map sizes, σ and α are chosen on validation RMSE; the test is scored once at the end.
    val = data["validation"]
    sigmas = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
    alphas = [(a, b) for a in np.linspace(0, 1, 11) for b in np.linspace(0, 1, 11)]
    search, best = [], None
    for past_side, future_side in itertools.product(args.past_sizes, args.future_sizes):
        past_map = SOM(SOMConfig(past_side, past_side, seed=args.seed)).fit(train["past"][usable])
        future_map = SOM(SOMConfig(future_side, future_side, seed=args.seed)).fit(train["future"][usable])
        link = link_matrix(past_map, future_map, train["past"][usable], train["future"][usable])
        for sigma in sigmas:
            nodes = past_map.bmu(np.where(val["ok"][:, None], val["past"], 0))
            weights = scenario_weights(link, nodes, val["forecast"] - val["last"][:, None],
                                       future_map.prototypes, sigma)
            scenario = val["last"][:, None] + weights @ future_map.prototypes
            for a_first, a_last in alphas:
                alpha = alpha_curve(a_first, a_last)
                p = np.where(val["ok"][:, None], alpha * val["forecast"] + (1 - alpha) * scenario, val["forecast"])
                row = {"past_side": past_side, "future_side": future_side, "sigma": sigma,
                       "alpha_first": round(a_first, 2), "alpha_last": round(a_last, 2),
                       "validation_rmse": rmse(val["y"], p)}
                search.append(row)
                if best is None or row["validation_rmse"] < best[0]["validation_rmse"]:
                    best = (row, past_map, future_map, link)
        print(f"maps {past_side}x{past_side} / {future_side}x{future_side}: best validation RMSE so far "
              f"{best[0]['validation_rmse']:.2f}", flush=True)
    choice, past_map, future_map, link = best
    pd.DataFrame(search).to_csv(output / "validation_search.csv", index=False)

    # Variants for the report, all with the chosen maps: the user's original analog forecast
    # (P(j | i) only, no model), the same as a post-processing, and the model-guided version.
    alpha = alpha_curve(choice["alpha_first"], choice["alpha_last"])
    test = data["test"]
    variants = {"base (submission)": test["forecast"]}
    variants["SOM analog only: Σ P(j|i)·prototype"] = apply(
        (past_map, future_map), link, test["past"], test["last"], test["ok"],
        test["forecast"], np.inf, np.zeros(HORIZON))[0]
    analog_val = apply((past_map, future_map), link, val["past"], val["last"], val["ok"],
                       val["forecast"], np.inf, np.zeros(HORIZON))[0]
    blend_rows = []
    for a_first, a_last in alphas:
        a = alpha_curve(a_first, a_last)
        blend_rows.append((rmse(val["y"], a * val["forecast"] + (1 - a) * analog_val), a))
    variants["base + analog, α on validation"] = apply(
        (past_map, future_map), link, test["past"], test["last"], test["ok"], test["forecast"], np.inf,
        min(blend_rows, key=lambda r: r[0])[1])[0]
    guided, weights, nodes = apply((past_map, future_map), link, test["past"], test["last"], test["ok"],
                                   test["forecast"], choice["sigma"], alpha)
    variants["model-guided SOM (chosen)"] = guided

    report = {"config": {"window_hours": WINDOW, "horizon_hours": HORIZON,
                         "past_input": "past 72 h" + (" + 27-day recurrence of the next 72 h"
                                                       if args.with_recurrence else ""), "splits": {
        k: [v.min().isoformat(), v.max().isoformat(), len(v)] for k, v in splits.items()}},
        "train_usable_issuances": int(usable.sum()), "choice": choice,
        "maps": {"past": {"side": choice["past_side"],
                          "quantization_error_km_s": past_map.quantization_error(train["past"][usable]),
                          "topographic_error": past_map.topographic_error(train["past"][usable])},
                 "future": {"side": choice["future_side"],
                            "quantization_error_km_s": future_map.quantization_error(train["future"][usable]),
                            "topographic_error": future_map.topographic_error(train["future"][usable])}},
        "mutual_information": {}, "test": {}, "test_gain_vs_base": {}}
    for name in ("train", "test"):
        d = data[name]
        both = d["ok"] & d["ok_future"]
        report["mutual_information"][name] = mutual_information(past_map.bmu(d["past"][both]),
                                                                future_map.bmu(d["future"][both]), shuffles=20)
    for name, values in variants.items():
        report["test"][name] = evaluate(test["y"], values, test["last"])
        if name != "base (submission)":
            report["test_gain_vs_base"][name] = block_bootstrap_gain(test["origins"], test["y"],
                                                                     test["forecast"], values)
    np.savez_compressed(output / "maps.npz", past=past_map.prototypes, future=future_map.prototypes,
                        past_grid=past_map.grid, future_grid=future_map.grid, link=link,
                        alpha=alpha, sigma=choice["sigma"])
    pd.DataFrame({"origin_utc": np.repeat(test["origins"], HORIZON),
                  "horizon": np.tile(np.arange(1, HORIZON + 1), len(test["origins"])),
                  "y": test["y"].ravel(), "last": np.repeat(test["last"], HORIZON),
                  "past_node": np.repeat(nodes, HORIZON), "ok": np.repeat(test["ok"], HORIZON),
                  **{name: values.ravel() for name, values in variants.items()}}).to_parquet(
        output / "test_predictions.parquet", index=False)
    train_counts = np.bincount(past_map.bmu(train["past"][usable]), minlength=past_map.nodes)
    future_counts = np.bincount(future_map.bmu(train["future"][usable]), minlength=future_map.nodes)
    np.savez_compressed(output / "train_assignments.npz", past_counts=train_counts, future_counts=future_counts)
    report["seconds"] = time.monotonic() - start
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"choice": choice, "test": report["test"], "gain": report["test_gain_vs_base"],
                      "mutual_information": report["mutual_information"]}, indent=2, default=float))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("experiment")
    run.add_argument("--level2", type=Path, default=Path("data/ace_level2/ac_h2_swe_hourly.parquet"))
    run.add_argument("--operational", type=Path, default=Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet"))
    run.add_argument("--coronal-holes", type=Path, default=Path("data/features/coronal_holes_0193_6h.parquet"))
    run.add_argument("--model-dir", type=Path, default=Path("submission/model"))
    run.add_argument("--train", nargs=2, default=["2011-02-01", "2021-12-28"])
    run.add_argument("--validation", nargs=2, default=["2022-01-01", "2023-12-28"])
    run.add_argument("--test", nargs=2, default=["2024-01-01", "2026-10-23T00:00"])
    run.add_argument("--past-sizes", type=int, nargs="+", default=[6, 10, 14])
    run.add_argument("--future-sizes", type=int, nargs="+", default=[4, 6, 8])
    run.add_argument("--with-recurrence", action="store_true",
                     help="Map A also sees the speed 27 days before each of the next 72 hours")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
