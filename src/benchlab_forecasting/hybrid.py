"""Hybrid forecaster: coronal holes + 27-day recurrence + L1 state, one boosted model for all lead times.

Each row is one (origin, lead hour) pair. Every input respects the cut-off
origin - latency: L1 plasma up to that hour, recurrence values from one and two
solar rotations before the target hour, and coronal-hole fractions from the AIA
image selected for the origin and the two previous days. A single
HistGradientBoostingRegressor learns all 72 lead times, with the lead hour as a
feature. Training can weight pairs so that the squared error equals TIMSE.

Example:
  .venv/bin/python -m benchlab_forecasting.hybrid experiment \
    --output-dir data/experiments/hybrid_2011_2026
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Metrics are imported where they are used, so inference (build_pairs and predict) needs only
# numpy, pandas and the boosting library; the submission container relies on this.

ROTATION_HOURS = 27 * 24
CH_THRESHOLDS = (25, 35)
CH_LONGITUDES = ("-60_-45", "-45_-30", "-30_-15", "-15_+0", "+0_+15", "+15_+30")
CH_LAGS_HOURS = (0, 24, 48)


@dataclass(frozen=True)
class HybridConfig:
    horizon: int = 72
    latency_hours: int = 1
    use_coronal_holes: bool = True
    weighting: str = "timse"  # "timse" or "mse"
    max_iter: int = 600
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    seed: int = 42
    # "hgb" (scikit-learn HistGradientBoosting), "lightgbm", "xgboost", "catboost" or "ngboost"
    model: str = "hgb"
    threads: int = 16
    selection_metric: str = "timse"  # validation metric that picks the number of trees: timse or rmse
    colsample: float = 1.0  # not for hgb: fraction of features per tree
    subsample: float = 1.0  # not for hgb: fraction of rows per tree


# Libraries whose first N trees are the model trained with N trees (no refit needed).
STAGED_MODELS = ("lightgbm", "xgboost", "catboost", "ngboost")


def load_plasma(level2_path, operational_path):
    """Hourly speed, density and temperature: Level 2 where valid, operational elsewhere."""
    columns = ["speed_km_s", "density_cm3", "temperature_k"]
    frames = []
    for path in (level2_path, operational_path):
        if path is None:
            continue
        frame = pd.read_parquet(path, columns=["timestamp_utc", *columns])
        frames.append(frame.set_index(pd.to_datetime(frame.timestamp_utc, utc=True))[columns])
    if not frames:
        raise ValueError("At least one plasma file is required")
    start = min(f.index.min() for f in frames).floor("h")
    end = max(f.index.max() for f in frames).floor("h")
    grid = pd.date_range(start, end, freq="h")
    plasma = frames[0].reindex(grid)
    for frame in frames[1:]:
        plasma = plasma.fillna(frame.reindex(grid))
    return plasma


def coronal_hole_columns():
    return [f"ch_t{t}_lat60_lon{lon}" for t in CH_THRESHOLDS for lon in CH_LONGITUDES]


def build_pairs(plasma, coronal_holes, origins, config):
    """Feature table with one row per origin and lead hour, plus the observed target `y`."""
    origins = pd.DatetimeIndex(origins)
    speed = plasma.speed_km_s
    cutoff = origins - pd.Timedelta(hours=config.latency_hours)
    at = lambda series, times: series.reindex(times).to_numpy(dtype=float)

    hours = lambda index: (index - pd.Timestamp("2000-01-01", tz="UTC")) / pd.Timedelta(hours=1)
    valid_time = pd.Series(np.where(speed.notna(), hours(speed.index), np.nan), index=speed.index).ffill()
    state = {
        "last_speed": at(speed.ffill(), cutoff),
        # Age of the latest valid speed in thousands of hours: the scale the saved models were
        # trained on (pandas' microsecond timestamps divided by nanoseconds per hour).
        "last_age_h": (np.asarray(hours(cutoff)) - at(valid_time, cutoff)) / 1000,
        "mean_6h": at(speed.rolling(6, min_periods=2).mean(), cutoff),
        "mean_24h": at(speed.rolling(24, min_periods=6).mean(), cutoff),
        "mean_72h": at(speed.rolling(72, min_periods=12).mean(), cutoff),
        "max_24h": at(speed.rolling(24, min_periods=6).max(), cutoff),
        "density_24h": at(plasma.density_cm3.rolling(24, min_periods=6).mean(), cutoff),
        "log_temperature_24h": np.log10(at(plasma.temperature_k.rolling(24, min_periods=6).mean(), cutoff)),
    }
    state["trend_24h"] = state["mean_6h"] - at(speed.rolling(6, min_periods=2).mean(),
                                               cutoff - pd.Timedelta(hours=24))
    # How well did one rotation ago describe the present? Large errors mean a changing Sun.
    rotation_now = at(speed.rolling(6, min_periods=2).mean(), cutoff - pd.Timedelta(hours=ROTATION_HOURS))
    state["recurrence_error_now"] = state["mean_6h"] - rotation_now

    smooth = speed.rolling(7, center=True, min_periods=3).mean()
    leads = np.arange(1, config.horizon + 1)
    target_times = origins.values[:, None] + (leads * np.timedelta64(1, "h"))[None, :]
    flat_targets = pd.DatetimeIndex(target_times.ravel(), tz="UTC")
    frame = pd.DataFrame({
        "origin_utc": np.repeat(origins, config.horizon),
        "timestamp_utc": flat_targets,
        "horizon": np.tile(leads, len(origins)),
        "y": at(speed, flat_targets),
        # Centered 7-hour means reach 3 hours past t - 27 d, still before the cut-off.
        "recurrence_27d": at(smooth, flat_targets - pd.Timedelta(hours=ROTATION_HOURS)),
        "recurrence_54d": at(smooth, flat_targets - pd.Timedelta(hours=2 * ROTATION_HOURS)),
    })
    for name, values in state.items():
        frame[name] = np.repeat(values, config.horizon)
    frame["recurrence_minus_last"] = frame.recurrence_27d - frame.last_speed
    if config.use_coronal_holes:
        table = coronal_holes.set_index("origin_utc")
        columns = coronal_hole_columns()
        for lag in CH_LAGS_HOURS:
            values = table[columns].reindex(origins - pd.Timedelta(hours=lag)).to_numpy(dtype=float)
            for i, column in enumerate(columns):
                frame[f"{column}_lag{lag}h"] = np.repeat(values[:, i], config.horizon)
        observed = pd.DatetimeIndex(pd.to_datetime(table.observation_utc.reindex(origins), utc=True))
        age = (origins - observed) / pd.Timedelta(hours=1)
        frame["ch_image_age_h"] = np.repeat(np.asarray(age, dtype=float), config.horizon)
    return frame


def feature_columns(frame):
    return [c for c in frame.columns if c not in ("origin_utc", "timestamp_utc", "y")]


def baseline_predictions(frame):
    """Persistence, and 27-day recurrence that falls back to persistence when missing."""
    return {"persistence": frame.last_speed.to_numpy(),
            "recurrence": frame.recurrence_27d.fillna(frame.last_speed).to_numpy()}


class HybridForecaster:
    def __init__(self, config=None):
        self.config = config or HybridConfig()
        self.model = None

    def _regressor(self, iterations):
        c = self.config
        if c.model == "lightgbm":
            from lightgbm import LGBMRegressor

            # Same tree size, leaf minimum, L2 penalty and step as the scikit-learn model.
            return LGBMRegressor(
                n_estimators=iterations, learning_rate=c.learning_rate, num_leaves=c.max_leaf_nodes,
                min_child_samples=c.min_samples_leaf, reg_lambda=c.l2_regularization,
                colsample_bytree=c.colsample, subsample=c.subsample, subsample_freq=int(c.subsample < 1),
                random_state=c.seed, n_jobs=c.threads, deterministic=True, force_row_wise=True,
                verbose=-1)
        if c.model == "xgboost":
            from xgboost import XGBRegressor

            # Leaf-wise growth with a leaf cap, like LightGBM. With squared error the hessian
            # is the sample weight, so min_child_weight equals the leaf minimum for unit weights.
            return XGBRegressor(
                n_estimators=iterations, learning_rate=c.learning_rate, tree_method="hist",
                grow_policy="lossguide", max_depth=0, max_leaves=c.max_leaf_nodes,
                min_child_weight=c.min_samples_leaf, reg_lambda=c.l2_regularization,
                colsample_bytree=c.colsample, subsample=c.subsample,
                random_state=c.seed, n_jobs=c.threads)
        if c.model == "catboost":
            from catboost import CatBoostRegressor

            # Leaf-wise trees with a leaf cap instead of CatBoost's default symmetric trees,
            # so the tree size means the same as in the other libraries.
            return CatBoostRegressor(
                iterations=iterations, learning_rate=c.learning_rate, grow_policy="Lossguide",
                max_leaves=c.max_leaf_nodes, min_data_in_leaf=c.min_samples_leaf,
                l2_leaf_reg=c.l2_regularization, rsm=c.colsample,
                bootstrap_type="Bernoulli" if c.subsample < 1 else "No",
                **({"subsample": c.subsample} if c.subsample < 1 else {}),
                random_seed=c.seed, thread_count=c.threads, verbose=0, allow_writing_files=False)
        if c.model == "ngboost":
            from ngboost.distns import Normal

            from .ngboost_trees import FastNGBRegressor, LightGBMTree

            # A normal distribution per pair; each parameter gets one LightGBM tree per iteration.
            return FastNGBRegressor(
                Dist=Normal, n_estimators=iterations, learning_rate=c.learning_rate,
                Base=LightGBMTree(num_leaves=c.max_leaf_nodes, min_child_samples=c.min_samples_leaf,
                                  reg_lambda=c.l2_regularization, threads=c.threads, seed=c.seed),
                minibatch_frac=c.subsample, col_sample=c.colsample, random_state=c.seed, verbose=False)
        if c.colsample != 1 or c.subsample != 1:
            raise ValueError("colsample and subsample are not available with hgb")
        if c.model == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=iterations, learning_rate=c.learning_rate, max_leaf_nodes=c.max_leaf_nodes,
                min_samples_leaf=c.min_samples_leaf, l2_regularization=c.l2_regularization,
                early_stopping=False, random_state=c.seed)
        raise ValueError("model must be hgb, lightgbm, xgboost, catboost or ngboost")

    def _staged(self, model, features, step):
        """Validation predictions after every `step` trees and after the last one."""
        last = self.config.max_iter
        if self.config.model in STAGED_MODELS and self.config.model != "ngboost":
            for iterations in [*range(step, last, step), last]:
                yield iterations, self._predict_first(model, features, iterations)
            return
        for iterations, prediction in enumerate(model.staged_predict(features), 1):
            if iterations % step == 0 or iterations == last:
                yield iterations, prediction

    def _predict_first(self, model, features, iterations):
        """Prediction of the first `iterations` trees of a LightGBM, XGBoost, CatBoost or NGBoost model."""
        if self.config.model == "ngboost":
            return model.predict(features, max_iter=iterations)
        if self.config.model == "lightgbm":
            return model.predict(features, num_iteration=iterations)
        if self.config.model == "catboost":
            return model.predict(features, ntree_end=iterations)
        return model.predict(features, iteration_range=(0, iterations))

    def _weights(self, y):
        from .validation import timse_weights

        if self.config.weighting == "mse":
            return np.ones(len(y))
        if self.config.weighting == "timse":
            weights = timse_weights(y)
            # Keep every pair in play: slow-wind pairs otherwise get almost no weight.
            return weights / weights.mean() + 0.05
        raise ValueError("weighting must be timse or mse")

    def fit(self, train, validation, step=25):
        """Fit on `train` and choose the number of trees by the validation metric.

        LightGBM, XGBoost, CatBoost and NGBoost training is deterministic and sequential, so their
        first N trees are the model trained with N trees: they keep the fitted booster and
        predict with N.
        The scikit-learn model is refit with the selected number of trees.
        """
        from .validation import regression_metrics, timse

        train = train[train.y.notna()]
        validation = validation[validation.y.notna()]
        self.features = feature_columns(train)
        start = time.monotonic()
        model = self._regressor(self.config.max_iter).fit(
            train[self.features], train.y, sample_weight=self._weights(train.y.to_numpy()))
        history = [{"iterations": iterations, "validation_timse": timse(validation.y, prediction),
                    "validation_rmse": regression_metrics(validation.y, prediction)["rmse"]}
                   for iterations, prediction in self._staged(model, validation[self.features], step)]
        metric = self.config.selection_metric
        if metric not in ("timse", "rmse"):
            raise ValueError("selection_metric must be timse or rmse")
        best = min(history, key=lambda row: row[f"validation_{metric}"])
        self.iterations = best["iterations"]
        if self.config.model in STAGED_MODELS:
            self.model = model
        else:
            self.model = self._regressor(best["iterations"]).fit(
                train[self.features], train.y, sample_weight=self._weights(train.y.to_numpy()))
        self.training = {"selected_iterations": best["iterations"], "history": history,
                         "train_pairs": len(train), "validation_pairs": len(validation),
                         "seconds": time.monotonic() - start,
                         "selection": f"Minimum validation {metric.upper()} over boosting iterations; test unused"}
        return self

    def predict(self, frame):
        if self.config.model in STAGED_MODELS:
            prediction = self._predict_first(self.model, frame[self.features], self.iterations)
        else:
            prediction = self.model.predict(frame[self.features])
        # Never return a missing value: fall back to recurrence, then persistence.
        fallback = baseline_predictions(frame)["recurrence"]
        return np.where(np.isfinite(prediction), np.maximum(prediction, 0), fallback)

    def predict_scale(self, frame):
        """NGBoost only: standard deviation of the predicted normal distribution (km/s)."""
        if self.config.model != "ngboost":
            raise ValueError("only ngboost predicts a distribution")
        distribution = self.model.pred_dist(frame[self.features], max_iter=self.iterations)
        return distribution.params["scale"]

    def save(self, path):
        with open(path, "wb") as stream:
            pickle.dump({"config": asdict(self.config), "features": self.features,
                         "model": self.model, "iterations": self.iterations,
                         "training": self.training}, stream)


def origins_between(start, end, step_hours=6):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq=f"{step_hours}h")


def score(frame, predictions):
    """TIMSE, RMSE and lead-time RMSE of each model on the pairs with an observed target."""
    from .validation import regression_metrics, timse

    observed = frame.y.notna().to_numpy()
    y = frame.y.to_numpy()[observed]
    results = {}
    for name, values in predictions.items():
        values = np.asarray(values)[observed]
        by_lead = pd.Series((values - y) ** 2).groupby(frame.horizon.to_numpy()[observed]).mean() ** 0.5
        results[name] = {"timse": timse(y, values), **regression_metrics(y, values),
                         "rmse_by_lead_block": {f"{lo}-{hi}h": float(np.sqrt(((values - y) ** 2)[
                             (frame.horizon.to_numpy()[observed] >= lo) &
                             (frame.horizon.to_numpy()[observed] <= hi)].mean()))
                             for lo, hi in ((1, 24), (25, 48), (49, 72))},
                         "rmse_by_lead": by_lead.round(2).tolist()}
    return results


SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "max_leaf_nodes": [15, 31, 63, 127, 255],
    "min_samples_leaf": [50, 100, 200, 500, 1000, 2000],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0, 30.0],
    "colsample": [0.5, 0.7, 0.9, 1.0],
    "subsample": [0.6, 0.8, 1.0],
}


def search_configs(trials, seed, threads, model="lightgbm"):
    """Trial 0 is the reference configuration; the rest are drawn at random from SEARCH_SPACE."""
    base = HybridConfig(weighting="mse", model=model, selection_metric="rmse", threads=threads)
    rng = np.random.default_rng(seed)
    configs = [replace(base, max_iter=800)]
    while len(configs) < trials:
        values = {key: options[rng.integers(len(options))] for key, options in SEARCH_SPACE.items()}
        values = {k: v.item() if hasattr(v, "item") else v for k, v in values.items()}
        # Smaller steps need more trees; the validation curve picks how many to keep.
        config = replace(base, max_iter=min(2000, int(40 / values["learning_rate"])), **values)
        if config not in configs:
            configs.append(config)
    return configs


def tune(plasma, coronal_holes, splits, args):
    """Random search on validation RMSE; the test is scored once, for the winner only."""
    from .validation import regression_metrics, timse

    output = args.output_dir
    log = output / "trials.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            done[json.dumps(row["config"], sort_keys=True)] = row
    configs = search_configs(args.trials, args.seed, args.threads, args.model)
    tables = {k: build_pairs(plasma, coronal_holes, v, configs[0]) for k, v in splits.items()}
    validation = tables["validation"][tables["validation"].y.notna()]
    for number, config in enumerate(configs):
        key = json.dumps(asdict(config), sort_keys=True)
        if key in done:
            continue
        model = HybridForecaster(config).fit(tables["train"], tables["validation"])
        prediction = model.predict(validation)
        row = {"trial": number, "config": asdict(config), "iterations": model.iterations,
               "validation_rmse": regression_metrics(validation.y, prediction)["rmse"],
               "validation_timse": timse(validation.y, prediction),
               "seconds": model.training["seconds"]}
        done[key] = row
        with log.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"trial {number:3d}: validation RMSE {row['validation_rmse']:.2f}  "
              f"({model.iterations} trees, {row['seconds']:.0f}s)  "
              + " ".join(f"{k}={getattr(config, k)}" for k in SEARCH_SPACE), flush=True)

    ranked = sorted(done.values(), key=lambda row: row["validation_rmse"])
    best = HybridConfig(**ranked[0]["config"])
    reference = configs[0]
    test_rows = tables["test"][["origin_utc", "timestamp_utc", "horizon", "y"]].copy()
    for name, values in baseline_predictions(tables["test"]).items():
        test_rows[name] = values
    report = {"selection": "Lowest validation RMSE; test scored once after the search",
              "trials": len(done), "ranking": ranked[:10], "models": {}}
    # The winner, the reference configuration, and the winner without coronal holes.
    for name, config in (("tuned", best), ("reference", reference),
                         ("tuned_no_ch", replace(best, use_coronal_holes=False))):
        data = tables if config.use_coronal_holes else {
            k: build_pairs(plasma, coronal_holes, v, config) for k, v in splits.items()}
        model = HybridForecaster(config).fit(data["train"], data["validation"])
        model.save(output / f"model_{name}.pkl")
        test_rows[f"{args.model}_{name}"] = model.predict(data["test"])
        report["models"][name] = {"config": asdict(config), "iterations": model.iterations}
    models = [c for c in test_rows.columns if c not in ("origin_utc", "timestamp_utc", "horizon", "y")]
    report["test"] = score(test_rows, {m: test_rows[m] for m in models})
    test_rows.to_parquet(output / "test_predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("\nTop 5 on validation:")
    for row in ranked[:5]:
        print(f"  trial {row['trial']:3d}: RMSE {row['validation_rmse']:.2f}  "
              + " ".join(f"{k}={row['config'][k]}" for k in SEARCH_SPACE))
    print("\nTest (scored once):")
    for name, metrics in sorted(report["test"].items(), key=lambda item: item[1]["rmse"]):
        blocks = "  ".join(f"{k} {v:.0f}" for k, v in metrics["rmse_by_lead_block"].items())
        print(f"  {name:22s} RMSE={metrics['rmse']:6.1f}  TIMSE={metrics['timse']:>12,.0f}  [{blocks}]")
    print(f"Results: {output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("experiment")
    search = sub.add_parser("tune", help="Random search of boosting hyperparameters, "
                                         "chosen by validation RMSE")
    for command in (run, search):
        command.add_argument("--level2", type=Path, default=Path("data/ace_level2/ac_h2_swe_hourly.parquet"))
        command.add_argument("--operational", type=Path,
                             default=Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet"))
        command.add_argument("--coronal-holes", type=Path,
                             default=Path("data/features/coronal_holes_0193_6h.parquet"))
        command.add_argument("--train", nargs=2, default=["2011-02-01", "2021-12-28"])
        command.add_argument("--validation", nargs=2, default=["2022-01-01", "2023-12-28"])
        command.add_argument("--test", nargs=2, default=["2024-01-01", "2026-10-23T00:00"])
        command.add_argument("--output-dir", type=Path, required=True)
    search.add_argument("--trials", type=int, default=40)
    search.add_argument("--threads", type=int, default=32)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--model", choices=STAGED_MODELS, default="lightgbm")
    run.add_argument("--variants", nargs="+", default=["timse", "mse", "timse_no_ch"],
                     help="timse, mse, timse_no_ch and/or mse_no_ch (variants without coronal holes)")
    run.add_argument("--model", choices=["hgb", *STAGED_MODELS], default="hgb",
                     help="Boosting library used by every variant")
    run.add_argument("--max-iter", type=int, help="Most trees per variant (default: HybridConfig.max_iter)")
    args = parser.parse_args(argv)

    output = args.output_dir
    # The search appends to its own log and can resume; experiments need a fresh directory.
    if args.command == "experiment" and output.exists() and any(output.iterdir()):
        print("Error: use an empty output directory", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    plasma = load_plasma(args.level2, args.operational)
    coronal_holes = pd.read_parquet(args.coronal_holes)
    coronal_holes = coronal_holes[coronal_holes.error.isna()]
    splits = {name: origins_between(*getattr(args, name)) for name in ("train", "validation", "test")}
    if splits["train"].max() + pd.Timedelta(hours=72) >= splits["validation"].min() or \
            splits["validation"].max() + pd.Timedelta(hours=72) >= splits["test"].min():
        print("Error: targets of one split must end before the next split begins", file=sys.stderr)
        return 1
    if args.command == "tune":
        return tune(plasma, coronal_holes, splits, args)

    variants = {"timse": HybridConfig(), "mse": HybridConfig(weighting="mse"),
                "timse_no_ch": HybridConfig(use_coronal_holes=False),
                "mse_no_ch": HybridConfig(weighting="mse", use_coronal_holes=False)}
    test_rows, report = None, {"splits": {k: [v.min().isoformat(), v.max().isoformat(), len(v)]
                                          for k, v in splits.items()}, "variants": {}}
    for name in args.variants:
        config = replace(variants[name], model=args.model)
        if args.max_iter:
            config = replace(config, max_iter=args.max_iter)
        tables = {k: build_pairs(plasma, coronal_holes, v, config) for k, v in splits.items()}
        print(f"{name}: {len(tables['train'])} train pairs, {len(feature_columns(tables['train']))} features",
              flush=True)
        model = HybridForecaster(config).fit(tables["train"], tables["validation"])
        model.save(output / f"model_{name}.pkl")
        if test_rows is None:
            test_rows = tables["test"][["origin_utc", "timestamp_utc", "horizon", "y"]].copy()
            for baseline, values in baseline_predictions(tables["test"]).items():
                test_rows[baseline] = values
        test_rows[f"hybrid_{name}"] = model.predict(tables["test"])
        if args.model == "ngboost":
            test_rows[f"hybrid_{name}_scale"] = model.predict_scale(tables["test"])
        report["variants"][name] = {"config": asdict(config), "features": model.features,
                                    "training": model.training}
        print(f"{name}: {model.training['selected_iterations']} trees, "
              f"{model.training['seconds']:.0f}s", flush=True)
    models = [c for c in test_rows.columns
              if c not in ("origin_utc", "timestamp_utc", "horizon", "y") and not c.endswith("_scale")]
    report["test"] = score(test_rows, {m: test_rows[m] for m in models})
    report["test_note"] = ("Targets after 2024-07-09 come from the operational product only (Level 2 ends); "
                           "the test was not used for any choice.")
    test_rows.to_parquet(output / "test_predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for name, metrics in sorted(report["test"].items(), key=lambda item: item[1]["timse"]):
        blocks = "  ".join(f"{k} {v:.0f}" for k, v in metrics["rmse_by_lead_block"].items())
        print(f"{name:20s} TIMSE={metrics['timse']:>12,.0f}  RMSE={metrics['rmse']:6.1f}  [{blocks}]")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
