"""Train/evaluate or load a compact CPU LSTM for solar-wind forecasting."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

from .data import utc_hour
from .features import load_regressors
from .gru import GRUForecaster
from .lstm import LSTMConfig, LSTMForecaster, load_lstm_data
from .models import ForecastConfig, ProphetForecaster, baseline, future_grid
from .validation.reporting import evaluate_predictions


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def origins(start, hours, step):
    if hours < 1 or step < 1:
        raise ValueError("Duration and cadence must be positive")
    start = utc_hour(start)
    return pd.date_range(start, start + pd.Timedelta(hours=hours), freq=f"{step}h", inclusive="left")


def evaluate(model, frame, test_origins, prophet_selection=None, prophet_inputs=None,
             lstm_reference=None, gru_reference=None):
    config = model.config
    series = frame[["ds", "y"]].copy()
    if "y__available_at" in frame:
        series["available_at"] = frame["y__available_at"]
    labels = series.set_index("ds").y
    reference = ForecastConfig(horizon=config.horizon, latency_minutes=config.latency_minutes,
                               max_age_hours=config.max_age_hours)
    model_name = model.architecture
    networks = {model_name: model}
    for name, reference_model in (("lstm", lstm_reference), ("gru", gru_reference)):
        if reference_model is None:
            continue
        if model_name == name:
            raise ValueError(f"The {name.upper()} reference must use a different architecture")
        if (reference_model.config.horizon != config.horizon or
                reference_model.config.latency_minutes != config.latency_minutes):
            raise ValueError("Recurrent models must share horizon and assumed latency")
        if set(reference_model.columns) - set(frame.columns):
            raise ValueError(f"Input data lacks features required by the {name.upper()} reference")
        networks[name] = reference_model
    names = [*networks, "persistence", "recurrence"]
    if prophet_selection:
        names.extend(["prophet_current", "prophet_tuned"])
    rows, failures, inference_times = [], [], []
    for origin in test_origins:
        dates = future_grid(origin, config.horizon)
        block = pd.DataFrame({"origin_utc": origin, "timestamp_utc": dates,
                              "horizon": np.arange(1, config.horizon + 1),
                              "y": labels.reindex(dates).to_numpy()})
        for name in names:
            try:
                if name in networks:
                    start = time.monotonic()
                    forecast = networks[name].predict(frame, origin)
                    if name == model_name:
                        inference_times.append(time.monotonic() - start)
                elif name in ("persistence", "recurrence"):
                    forecast = baseline(series, origin, reference, name)
                else:
                    cfg = (reference if name == "prophet_current" else
                           ForecastConfig(**prophet_selection["config"]))
                    if cfg.horizon != config.horizon or cfg.latency_minutes != config.latency_minutes:
                        raise ValueError("Compared models must share horizon and assumed latency")
                    forecast = ProphetForecaster(cfg).fit(
                        series, origin, prophet_inputs if name == "prophet_tuned" else None).predict()
                block[name] = forecast.speed_km_s.to_numpy()
            except Exception as exc:
                block[name] = np.nan
                failures.append({"origin_utc": origin.isoformat(), "model": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
        rows.append(block)
    predictions = pd.concat(rows, ignore_index=True)
    report = evaluate_predictions(predictions, names, horizon=config.horizon, expected_origins=test_origins)
    report.update({"failures": failures, "config": {"horizon": config.horizon},
                   f"{model_name}_inference_seconds": inference_times,
                   f"{model_name}_inference_p50_seconds": float(np.median(inference_times)) if inference_times else None,
                   f"{model_name}_inference_p95_seconds": float(np.quantile(inference_times, 0.95)) if inference_times else None})
    complete_comparison = evaluate_predictions(predictions, [n for n in names if n != "recurrence"],
                                              horizon=config.horizon, expected_origins=test_origins)
    return predictions, report, complete_comparison


def main(argv=None, *, forecaster_class=LSTMForecaster, config_class=LSTMConfig):
    model_name = forecaster_class.architecture
    parser = argparse.ArgumentParser(description=f"Train/evaluate or load a {model_name.upper()} forecaster")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("experiment", "forecast"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--mag-data", type=Path)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "forecast":
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--origin", required=True)
        else:
            sub.add_argument("--train-start", required=True)
            sub.add_argument("--validation-start", required=True)
            sub.add_argument("--validation-hours", type=int, default=336)
            sub.add_argument("--test-start", required=True)
            sub.add_argument("--test-hours", type=int, default=168)
            sub.add_argument("--step-hours", type=int, default=6)
            sub.add_argument("--prophet-config", type=Path, help="Previously selected best_config.json; no new tuning")
            sub.add_argument("--lstm-model", type=Path, help="Optional frozen LSTM checkpoint for comparison")
            sub.add_argument("--gru-model", type=Path, help="Optional frozen GRU checkpoint for comparison")
            for key, value in asdict(config_class()).items():
                sub.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty experiment directory")
        if args.command == "forecast":
            model = forecaster_class.load(args.model)
            frame = load_lstm_data(args.data, args.mag_data, model.config.input_group)
            forecast = model.predict(frame, args.origin)
            output.mkdir(parents=True, exist_ok=True)
            forecast.to_csv(output / "forecast.csv", index=False)
            write_json(output / "report.json", {"origin": utc_hour(args.origin).isoformat(), "model": model_name,
                                                "data": str(args.data.resolve()), "config": asdict(model.config),
                                                "checkpoint": str(args.model.resolve())})
        else:
            config = config_class(**{key: getattr(args, key) for key in config_class.__dataclass_fields__})
            frame = load_lstm_data(args.data, args.mag_data, config.input_group)
            validation = origins(args.validation_start, args.validation_hours, args.step_hours)
            test = origins(args.test_start, args.test_hours, args.step_hours)
            train = pd.date_range(utc_hour(args.train_start),
                                  validation.min() - pd.Timedelta(hours=config.horizon + 1),
                                  freq=f"{args.step_hours}h")
            last_validation_label = validation.max() + pd.Timedelta(hours=config.horizon)
            if last_validation_label + pd.Timedelta(minutes=config.latency_minutes) >= test.min():
                raise ValueError("Validation labels must be available before the test begins")
            if test.max() + pd.Timedelta(hours=config.horizon) > frame.ds.max():
                raise ValueError("Test plus horizon extends beyond the observed data")
            if train.empty or train.min() - pd.Timedelta(hours=config.lookback_hours - 1) < frame.ds.min():
                raise ValueError("Insufficient history for the training window")
            selection, prophet_inputs = None, None
            references = {}
            for name, path, reference_class in (("lstm", args.lstm_model, LSTMForecaster),
                                                ("gru", args.gru_model, GRUForecaster)):
                if path is None:
                    continue
                if name == model_name:
                    raise ValueError(f"--{name}-model must use a different architecture")
                reference_model = reference_class.load(path)
                if (reference_model.config.horizon != config.horizon or
                        reference_model.config.latency_minutes != config.latency_minutes or
                        set(reference_model.columns) - set(frame.columns)):
                    raise ValueError(f"{name.upper()} reference has incompatible horizon, latency or features")
                if test.min() < reference_model.available_at:
                    raise ValueError(f"Test begins before the {name.upper()} checkpoint was available")
                references[name] = reference_model
            if args.prophet_config:
                selection = json.loads(args.prophet_config.read_text())
                if selection["group"] != "none":
                    prophet_inputs = load_regressors(args.data, args.mag_data, selection["group"])
            output.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for path in (args.data, args.mag_data, args.prophet_config, args.lstm_model, args.gru_model):
                if path:
                    with path.open("rb") as stream:
                        hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {"architecture": model_name, "config": asdict(config), "data_sha256": hashes,
                        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "versions": {n: version(n) for n in ("torch", "numpy", "pandas")},
                        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob("*.py")},
                        "train_origins": [t.isoformat() for t in train],
                        "validation_origins": [t.isoformat() for t in validation],
                        "test_origins": [t.isoformat() for t in test],
                        "availability": "Source available_at when present; otherwise assumed latency",
                        "evaluation": "Recurrent weights frozen after validation; Prophet refits per origin using its fixed configuration"}
            write_json(output / "manifest.json", manifest)
            model = forecaster_class(config).fit(frame, train, validation)
            # Save the selected checkpoint before any test forecasts or scoring.
            model.save(output / "model.pt")
            write_json(output / "training.json", model.training)
            pd.DataFrame(model.training["history"]).to_csv(output / "training_history.csv", index=False)
            test_dir = output / "test"
            test_dir.mkdir()
            predictions, report, complete = evaluate(
                model, frame, test, selection, prophet_inputs,
                references.get("lstm"), references.get("gru"),
            )
            report["arguments"] = {"start": test.min().isoformat(), "end": test.max().isoformat(),
                                    "step_hours": args.step_hours}
            report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            predictions.to_parquet(test_dir / "predictions.parquet", index=False)
            write_json(test_dir / "report.json", report)
            write_json(test_dir / "comparison_without_recurrence.json", complete)
            for name, metrics in complete["metrics"].items():
                print(f"{name}: test RMSE={metrics['common']['rmse']} km/s", flush=True)
            if report["failures"] or complete["common_pairs"] == 0:
                print("Evaluation contains failures or no common targets; inspect report.json", file=sys.stderr)
                return 1
        print(f"Results: {output}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
