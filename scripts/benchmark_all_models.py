#!/usr/bin/env python3
"""Score every model on the same issuances: the hybrid test period, Jan 2024 – Oct 2026.

Neural networks are loaded from their saved checkpoints: the ones retrained on the hybrid's
splits (scripts/slurm/train_networks.sbatch) and the earlier 2019 and 2003–2009 runs and Prophet is refit at
every issuance with its fixed configuration, as in its own experiments. Hybrid forecasts and the
persistence/recurrence baselines come from the saved hybrid test predictions. Every model is
scored against the same observed target on the pairs where all of them have a forecast.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_all_models.py --workers 32
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path("data/experiments")
PLASMA = Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet")
MAG = Path("data/ace_mag_hourly_2011_2026/ace_mag_hourly.parquet")

# column: (label, family, training period)
SUBMISSION = {"submission": ("Submission · tuned LightGBM + persistence blend", "Hybrid boosting", "2011–2021")}
HYBRIDS = {
    "hybrid_lightgbm_tuning/lightgbm_tuned": ("Hybrid · LightGBM, tuned", "Hybrid boosting", "2011–2021"),
    "hybrid_xgboost_2011_2026/hybrid_mse": ("Hybrid · XGBoost", "Hybrid boosting", "2011–2021"),
    "hybrid_2011_2026/hybrid_mse": ("Hybrid · scikit-learn HGB", "Hybrid boosting", "2011–2021"),
    "hybrid_lightgbm_2011_2026/hybrid_mse": ("Hybrid · LightGBM", "Hybrid boosting", "2011–2021"),
    "hybrid_catboost_2011_2026/hybrid_mse": ("Hybrid · CatBoost", "Hybrid boosting", "2011–2021"),
    "hybrid_ngboost_2011_2026/hybrid_mse": ("Hybrid · NGBoost", "Hybrid boosting", "2011–2021"),
    "hybrid_lightgbm_tuning/lightgbm_tuned_no_ch": ("Hybrid · LightGBM, tuned, no coronal holes",
                                                    "Hybrid boosting", "2011–2021"),
    "hybrid_xgboost_2011_2026/hybrid_timse": ("Hybrid · XGBoost, TIMSE weights", "Hybrid boosting", "2011–2021"),
}
NETWORKS = {
    "lstm_2011_2021": ("LSTM", "Neural network", "2011–2021"),
    "gru_2011_2021": ("GRU", "Neural network", "2011–2021"),
    "attention_gru_2011_2021": ("GRU with attention", "Neural network", "2011–2021"),
    "attention_lstm_2011_2021": ("LSTM with attention", "Neural network", "2011–2021"),
    "transformer_2011_2021": ("Transformer", "Neural network", "2011–2021"),
    "patch_transformer_2011_2021": ("Patch Transformer", "Neural network", "2011–2021"),
    "lstm_2019": ("LSTM", "Neural network", "2019"),
    "gru_2019": ("GRU", "Neural network", "2019"),
    "attention_gru_2019_nov": ("GRU with attention", "Neural network", "2019"),
    "attention_lstm_2019_nov": ("LSTM with attention", "Neural network", "2019"),
    "transformer_2019_nov": ("Transformer", "Neural network", "2019"),
    "patch_transformer_2019_nov": ("Patch Transformer", "Neural network", "2019"),
    "attention_gru_2003_2010": ("GRU with attention", "Neural network", "2003–2009"),
    "transformer_2003_2010": ("Transformer", "Neural network", "2003–2009"),
    "patch_transformer_2003_2010": ("Patch Transformer", "Neural network", "2003–2009"),
}
PROPHETS = {
    "prophet_tuned": ("Prophet, tuned (plasma regressors)", "Statistical", "180 days before each issuance"),
    "prophet_default": ("Prophet", "Statistical", "180 days before each issuance"),
}
BASELINES = {
    "persistence": ("Persistence", "Baseline", "—"),
    "recurrence": ("27-day recurrence", "Baseline", "—"),
}


def reference():
    """Issuances, target hours, observed speed and baselines from the tuned hybrid run."""
    frame = pd.read_parquet(EXPERIMENTS / "hybrid_lightgbm_tuning" / "test_predictions.parquet")
    for column in ("origin_utc", "timestamp_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame[["origin_utc", "timestamp_utc", "horizon", "y", "persistence", "recurrence"]]


def add_hybrids(table):
    for key in HYBRIDS:
        folder, column = key.split("/")
        other = pd.read_parquet(EXPERIMENTS / folder / "test_predictions.parquet")
        other["origin_utc"] = pd.to_datetime(other.origin_utc, utc=True)
        table = table.merge(other[["origin_utc", "horizon", column]].rename(columns={column: key}),
                            on=["origin_utc", "horizon"], how="left", validate="one_to_one")
    return table


def add_submission(table):
    """The submitted forecast: per-lead persistence weights from submission/model/tuned.json."""
    meta = json.loads(Path("submission/model/tuned.json").read_text())
    w = np.asarray(meta["persistence_weight_by_lead"])[table.horizon.to_numpy() - 1]
    hybrid = table["hybrid_lightgbm_tuning/lightgbm_tuned"].to_numpy()
    table["submission"] = np.where(np.isfinite(table.persistence), w * table.persistence + (1 - w) * hybrid, hybrid)
    return table


def network_forecasts(name, origins):
    """Forecasts of one saved checkpoint at every issuance, using the same steps as its predict()."""
    import torch
    from benchlab_forecasting.attention_gru import AttentionGRUForecaster
    from benchlab_forecasting.attention_lstm import AttentionLSTMForecaster
    from benchlab_forecasting.gru import GRUForecaster
    from benchlab_forecasting.lstm import LSTMForecaster, causal_window, indexed_frame, load_lstm_data
    from benchlab_forecasting.transformer import PatchTransformerForecaster, TransformerForecaster

    classes = {c.architecture: c for c in (LSTMForecaster, GRUForecaster, AttentionGRUForecaster,
                                           AttentionLSTMForecaster, TransformerForecaster,
                                           PatchTransformerForecaster)}
    torch.set_num_threads(1)
    path = EXPERIMENTS / name / "model.pt"
    manifest = json.loads((EXPERIMENTS / name / "manifest.json").read_text())
    architecture = manifest.get("architecture", "lstm")  # the first LSTM run predates the field
    model = classes[architecture].load(path)
    frame = indexed_frame(load_lstm_data(PLASMA, MAG, model.config.input_group))
    rows, failures = [], 0
    for origin in origins:
        if origin < model.available_at:
            raise ValueError(f"{name}: issuance before the checkpoint was available")
        try:
            values, flags, anchor = causal_window(frame, origin, model.columns, model.config)
            with torch.inference_mode():
                correction = model.model(model.transform(values[None], flags[None])).numpy()[0]
            forecast = np.maximum(anchor + correction * model.target_scale, 0)
        except Exception:  # no valid recent speed: the model cannot issue a forecast
            forecast = np.full(model.config.horizon, np.nan)
            failures += 1
        rows.append(forecast)
    print(f"{name}: {len(origins) - failures}/{len(origins)} issuances", flush=True)
    return name, np.concatenate(rows)


_PROPHET = {}


def _prophet_init():
    from benchlab_forecasting.features import load_regressors
    from benchlab_forecasting.lstm import load_lstm_data

    frame = load_lstm_data(PLASMA, None, "speed")
    series = frame[["ds", "y"]].copy()
    if "y__available_at" in frame:
        series["available_at"] = frame["y__available_at"]
    selection = json.loads((EXPERIMENTS / "prophet_tuning_2019" / "best_config.json").read_text())
    _PROPHET.update(series=series, selection=selection,
                    regressors=load_regressors(PLASMA, MAG, selection["group"]))


def _prophet_one(origin):
    import logging

    from benchlab_forecasting.models import ForecastConfig, ProphetForecaster

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    out = {}
    for name, config, regressors in (
            ("prophet_tuned", ForecastConfig(**_PROPHET["selection"]["config"]), _PROPHET["regressors"]),
            ("prophet_default", ForecastConfig(horizon=72, latency_minutes=60, max_age_hours=24), None)):
        try:
            out[name] = ProphetForecaster(config).fit(_PROPHET["series"], origin, regressors).predict() \
                .speed_km_s.to_numpy()
        except Exception:
            out[name] = np.full(72, np.nan)
    return out


def timse_by_issuance(frame, column):
    from benchlab_forecasting.validation.metrics import timse

    return frame.groupby("origin_utc").apply(lambda g: timse(g.y.to_numpy(), g[column].to_numpy()))


def score(table, columns):
    from benchlab_forecasting.validation.metrics import timse

    common = table.dropna(subset=["y", *columns])
    rows = []
    for column in columns:
        error = common[column] - common.y
        rmse = lambda mask: float(np.sqrt((error[mask] ** 2).mean()))
        rows.append({"column": column, "rmse": rmse(slice(None)),
                     "rmse_1_24": rmse(common.horizon <= 24),
                     "rmse_25_48": rmse(common.horizon.between(25, 48)),
                     "rmse_49_72": rmse(common.horizon >= 49), "mae": float(error.abs().mean()),
                     "timse_pooled_millions": timse(common.y.to_numpy(), common[column].to_numpy()) / 1e6,
                     "timse_median_thousands": float(timse_by_issuance(common, column).median()) / 1e3})
    result = pd.DataFrame(rows)
    persistence = result.set_index("column").loc["persistence", "rmse"]
    result["skill_vs_persistence_pct"] = 100 * (1 - result.rmse / persistence)
    return result, common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "benchmark_2024_2026")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = add_submission(add_hybrids(reference()).sort_values(["origin_utc", "horizon"]).reset_index(drop=True))
    origins = list(table.origin_utc.drop_duplicates())
    with Pool(min(args.workers, len(NETWORKS))) as pool:
        for name, values in pool.starmap(network_forecasts, [(n, origins) for n in NETWORKS]):
            table[name] = values
    with Pool(args.workers, initializer=_prophet_init) as pool:
        prophet = pool.map(_prophet_one, origins, chunksize=8)
    for name in PROPHETS:
        table[name] = np.concatenate([p[name] for p in prophet])
    table.to_parquet(args.output_dir / "predictions.parquet", index=False)

    labels = {**SUBMISSION, **HYBRIDS, **NETWORKS, **PROPHETS, **BASELINES}
    result, common = score(table, list(labels))
    for i, key in enumerate(("model", "family", "training")):
        result[key] = result.column.map(lambda c: labels[c][i])
    result = result.sort_values("rmse").reset_index(drop=True)
    result.to_csv(args.output_dir / "benchmark.csv", index=False)
    summary = {"issuances": int(common.origin_utc.nunique()), "pairs": int(len(common)),
               "issuances_total": len(origins),
               "first_issuance": common.origin_utc.min().isoformat(),
               "last_issuance": common.origin_utc.max().isoformat()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(result[["model", "training", "rmse", "rmse_49_72", "timse_pooled_millions",
                  "skill_vs_persistence_pct"]].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
