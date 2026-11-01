#!/usr/bin/env python3
"""Fit one causal Prophet forecasting strategy for each complete ACE window length.

Prophet is fitted again at each forecast origin using only the speed history in
that origin's 3/7/14-day input window. A small chronological validation selects
flat or linear growth; a later common-origin test evaluates the selected model.
The saved model for each duration is the fit at its latest eligible origin.
"""

import argparse
import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


DURATIONS = (3, 7, 14)
HORIZON = 72
SEED = 42
BASE_PROPHET_CONFIG = {
    "n_changepoints": 0,
    "daily_seasonality": False,
    "weekly_seasonality": False,
    "yearly_seasonality": False,
    "uncertainty_samples": 0,
}
PROPHET_VARIANTS = {
    "flat": {**BASE_PROPHET_CONFIG, "growth": "flat"},
    "linear": {**BASE_PROPHET_CONFIG, "growth": "linear"},
}


def load_windows(data_dir, source_sha256):
    bundles = {}
    for days in DURATIONS:
        folder = data_dir / f"{days}d_72h"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_sha256:
            raise ValueError(f"Fonte das janelas de {days} dias não corresponde ao dataset atual")
        if manifest["lookback_hours"] != days * 24 or manifest["horizon_hours"] != HORIZON:
            raise ValueError(f"Horizonte ou entrada inesperada em {folder}")
        if manifest["input_features"][0] != "speed_km_s":
            raise ValueError(f"A primeira entrada deve ser speed_km_s em {folder}")
        inputs = np.load(folder / "inputs.npy", mmap_mode="r", allow_pickle=False)
        targets = np.load(folder / "targets.npy", mmap_mode="r", allow_pickle=False)
        samples = pd.read_parquet(folder / "samples.parquet")
        count = manifest["samples"]
        if (inputs.shape != (count, days * 24, 9)
                or targets.shape != (count, HORIZON)
                or len(samples) != count or count == 0):
            raise ValueError(f"Formato de janelas inconsistente em {folder}")
        if samples.t0_utc.duplicated().any() or not samples.t0_utc.is_monotonic_increasing:
            raise ValueError(f"Origens duplicadas ou desordenadas em {folder}")
        bundles[days] = {"folder": folder, "manifest": manifest, "inputs": inputs,
                         "targets": targets, "samples": samples,
                         "by_origin": {origin: idx for idx, origin in enumerate(samples.t0_utc)}}
    return bundles


def select_common_origins(bundles, count):
    """Pick recent common origins with nonoverlapping 72-hour target intervals."""
    common = set(bundles[3]["by_origin"]) & set(bundles[7]["by_origin"])
    common &= set(bundles[14]["by_origin"])
    chosen = []
    for origin in sorted(common, reverse=True):
        if not chosen or chosen[-1] - origin >= pd.Timedelta(hours=HORIZON):
            chosen.append(origin)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise ValueError(f"Apenas {len(chosen)} origens comuns sem sobreposição; requeridas {count}")
    return sorted(chosen)


def fit_window(speed, sample, config, seed=SEED):
    """Fit Prophet on exactly the input hours and predict t0+1 through t0+72."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

    speed = np.asarray(speed, dtype=np.float64)
    origin = pd.Timestamp(sample.t0_utc)
    start = pd.Timestamp(sample.input_start_utc)
    end = pd.Timestamp(sample.input_end_utc)
    if (len(speed) < 48 or not np.isfinite(speed).all() or (speed < 0).any()
            or end != origin - pd.Timedelta(hours=1)
            or start != origin - pd.Timedelta(hours=len(speed))):
        raise ValueError("Janela de entrada inválida para Prophet")
    history_dates = pd.date_range(start=start, periods=len(speed), freq="h")
    history = pd.DataFrame({"ds": history_dates.tz_localize(None), "y": speed})
    model = Prophet(**config)
    model.fit(history, seed=seed)
    future_dates = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    future = pd.DataFrame({"ds": future_dates.tz_localize(None)})
    forecast = np.maximum(model.predict(future).yhat.to_numpy(dtype=np.float64), 0)
    if len(forecast) != HORIZON or not np.isfinite(forecast).all():
        raise ValueError("Prophet produziu trajetória inválida")
    return model, forecast, future_dates


def metrics(observed, predicted):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - observed
    return {"pairs": int(error.size), "rmse_km_s": float(np.sqrt(np.mean(error ** 2))),
            "mae_km_s": float(np.mean(np.abs(error))),
            "bias_km_s": float(np.mean(error)),
            "rmse_hour_72_km_s": float(np.sqrt(np.mean(error[:, -1] ** 2)))}


def backtest_duration(bundle, origins, config, seed):
    rows = []
    for origin in origins:
        index = bundle["by_origin"][origin]
        sample = bundle["samples"].iloc[index]
        speed = np.asarray(bundle["inputs"][index, :, 0], dtype=np.float64)
        target = np.asarray(bundle["targets"][index], dtype=np.float64)
        _, prediction, dates = fit_window(speed, sample, config, seed=seed)
        if not np.isfinite(target).all():
            raise ValueError(f"Alvos incompletos em {origin}")
        rows.append(pd.DataFrame({
            "origin_utc": origin, "timestamp_utc": dates,
            "horizon": np.arange(1, HORIZON + 1, dtype=np.int16),
            "observed_km_s": target, "prophet_km_s": prediction,
            "persistence_km_s": np.full(HORIZON, speed[-1]),
        }))
    return pd.concat(rows, ignore_index=True)


def score_backtest(predictions):
    expected = predictions.groupby("origin_utc").size()
    if not expected.eq(HORIZON).all():
        raise ValueError("Backtest sem 72 alvos por origem")
    observed = predictions.observed_km_s.to_numpy().reshape(-1, HORIZON)
    return {name: metrics(observed, predictions[column].to_numpy().reshape(-1, HORIZON))
            for name, column in (("prophet", "prophet_km_s"),
                                 ("persistence", "persistence_km_s"))}


def write_duration(bundle, validation_origins, test_origins, output_dir, seed, source_sha256):
    from prophet.serialize import model_to_json

    days = bundle["manifest"]["lookback_hours"] // 24
    validation_frames = {
        name: backtest_duration(bundle, validation_origins, config, seed)
        for name, config in PROPHET_VARIANTS.items()
    }
    validation_scores = {
        name: score_backtest(frame)["prophet"]
        for name, frame in validation_frames.items()
    }
    selected = min(validation_scores,
                   key=lambda name: (validation_scores[name]["rmse_km_s"], name))
    selected_config = PROPHET_VARIANTS[selected]
    test_frame = backtest_duration(bundle, test_origins, selected_config, seed)
    test_scores = score_backtest(test_frame)
    latest = bundle["samples"].iloc[-1]
    speed = np.asarray(bundle["inputs"][-1, :, 0], dtype=np.float64)
    target = np.asarray(bundle["targets"][-1], dtype=np.float64)
    model, forecast, dates = fit_window(speed, latest, selected_config, seed=seed)
    latest_frame = pd.DataFrame({
        "origin_utc": latest.t0_utc, "timestamp_utc": dates,
        "horizon": np.arange(1, HORIZON + 1, dtype=np.int16),
        "observed_km_s": target, "prophet_km_s": forecast,
        "persistence_km_s": np.full(HORIZON, speed[-1]),
    })
    manifest = {
        "duration_days": days, "horizon_hours": HORIZON,
        "model_kind": "Prophet re-fitted per forecast origin",
        "input_field": "speed_km_s", "input_units": "km/s",
        "other_input_fields_used": False,
        "reason_other_fields_unused": "Their future values are unknown at the forecast origin",
        "selected_variant": selected,
        "prophet_parameters": selected_config, "seed": seed,
        "prophet_version": version("prophet"),
        "source_sha256": source_sha256,
        "latest_origin_utc": latest.t0_utc.isoformat(),
        "latest_input_start_utc": latest.input_start_utc.isoformat(),
        "latest_input_end_utc": latest.input_end_utc.isoformat(),
        "validation_origins": len(validation_origins),
        "validation_first_origin_utc": validation_origins[0].isoformat(),
        "validation_last_origin_utc": validation_origins[-1].isoformat(),
        "validation_variants": validation_scores,
        "test_origins": len(test_origins),
        "test_first_origin_utc": test_origins[0].isoformat(),
        "test_last_origin_utc": test_origins[-1].isoformat(),
        "test": test_scores,
        "latest_window": {"prophet": metrics(target[None, :], forecast[None, :]),
                          "persistence": metrics(target[None, :],
                                                 np.full((1, HORIZON), speed[-1]))},
        "note": "Historical fitted model is tied to its saved origin; refit at a new origin."
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        pending = Path(temporary)
        (pending / "model.json").write_text(model_to_json(model), encoding="utf-8")
        latest_frame.to_parquet(pending / "latest_forecast.parquet", index=False)
        for name, frame in validation_frames.items():
            frame.to_parquet(pending / f"validation_{name}.parquet", index=False)
        test_frame.to_parquet(pending / "test.parquet", index=False)
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for name in ("model.json", "latest_forecast.parquet", "test.parquet",
                     "validation_flat.parquet", "validation_linear.parquet", "manifest.json"):
            (pending / name).replace(output_dir / name)
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/training_windows")
    parser.add_argument("--source", type=Path,
                        default=root / "data/quality/ace_interpolated_all_internal.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/models/prophet_complete_windows")
    parser.add_argument("--validation-origins", type=int, default=12)
    parser.add_argument("--test-origins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if min(args.validation_origins, args.test_origins, args.seed) < 1:
        parser.error("validation-origins, test-origins e seed devem ser positivos")
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    source = args.source if args.source.is_absolute() else root / args.source
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    try:
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        bundles = load_windows(data_dir, source_sha256)
        origins = select_common_origins(
            bundles, args.validation_origins + args.test_origins)
        validation_origins = origins[:args.validation_origins]
        test_origins = origins[args.validation_origins:]
        results = {}
        for days in DURATIONS:
            manifest = write_duration(bundles[days], validation_origins, test_origins,
                                      output_dir / f"{days}d_72h", args.seed, source_sha256)
            results[f"{days}d_72h"] = {
                "latest_origin_utc": manifest["latest_origin_utc"],
                "selected_variant": manifest["selected_variant"],
                "test_origins": manifest["test_origins"],
                "prophet_rmse_km_s": manifest["test"]["prophet"]["rmse_km_s"],
                "persistence_rmse_km_s": manifest["test"]["persistence"]["rmse_km_s"],
            }
            print(f"{days} dias: modelo e backtest salvos", flush=True)
        summary = {"source_sha256": source_sha256,
                   "evaluation": "chronological validation and later test on the same nonoverlapping origins for all durations",
                   "validation_origins": [t.isoformat() for t in validation_origins],
                   "test_origins": [t.isoformat() for t in test_origins],
                   "results": results}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
