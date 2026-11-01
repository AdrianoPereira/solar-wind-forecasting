"""Bounded, chronological hyperparameter search for the attention GRU."""

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
from itertools import product
import json
from math import prod
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

from .attention_gru import AttentionGRUConfig, AttentionGRUForecaster
from .data import utc_hour
from .gru import GRUForecaster
from .lstm import LSTMForecaster, load_lstm_data
from .lstm_cli import evaluate, origins, write_json
from .models import future_grid
from .validation.reporting import evaluate_predictions


def default_candidates(base):
    """Small search with a plain attention reference and two targeted changes."""
    return [
        {"name": "attention", "config": replace(base, residual_scale=0.0)},
        {"name": "attention_residual", "config": replace(base, residual_scale=1.0)},
        {"name": "attention_small", "config": replace(base, hidden_size=16,
                                                      attention_heads=2, residual_scale=0.0)},
    ]


def read_candidates(path, base):
    if path is None:
        return default_candidates(base)
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Candidates JSON must be a list")
    result = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("config", {}), dict):
            raise ValueError("Each candidate needs a name and config object")
        result.append({"name": item.get("name"),
                       "config": replace(base, **item.get("config", {}))})
    return result


def read_search_space(path, base):
    """Expand a small JSON grid in stable field-name order."""
    space = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(space, dict) or not space:
        raise ValueError("Search space must be a nonempty JSON object")
    tunable = {"lookback_hours", "hidden_size", "attention_heads", "dropout",
               "learning_rate", "weight_decay", "batch_size", "fill_hours",
               "max_age_hours", "input_group", "residual_scale"}
    if set(space) - tunable:
        raise ValueError(f"Unsupported search fields: {sorted(set(space) - tunable)}")
    keys = sorted(space)
    if any(not isinstance(space[key], list) or not space[key] for key in keys):
        raise ValueError("Each search field must contain a nonempty list")
    count = prod(len(space[key]) for key in keys)
    if count > 32:
        raise ValueError("Search grid exceeds the limit of 32 candidates")
    return [{"name": f"trial_{index:03d}",
             "config": replace(base, **dict(zip(keys, values)))}
            for index, values in enumerate(product(*(space[key] for key in keys)), 1)]


def check_schedule(frame, candidates, train, early_stop, folds, test):
    if not candidates or len(candidates) > 32:
        raise ValueError("Provide between 1 and 32 candidates")
    names = [item["name"] for item in candidates]
    if (any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in names) or len(set(names)) != len(names)):
        raise ValueError("Candidate names must be unique lowercase identifiers")
    if set(names) & {"attention_gru", "lstm", "gru", "persistence", "recurrence"}:
        raise ValueError("Candidate names must not use model report names")
    configs = [item["config"] for item in candidates]
    if any(not isinstance(config, AttentionGRUConfig) for config in configs):
        raise ValueError("All candidates must use AttentionGRUConfig")
    horizon = configs[0].horizon
    if any(config.horizon != horizon for config in configs):
        raise ValueError("All candidates must use the same forecast horizon")
    if any(config.latency_minutes != configs[0].latency_minutes for config in configs):
        raise ValueError("All candidates must use the same assumed data latency")
    all_blocks = [train, early_stop, *folds, test]
    if not folds or any(len(block) == 0 or not block.is_unique or
                        not block.is_monotonic_increasing for block in all_blocks):
        raise ValueError("Train, early stop, validation and test blocks must be nonempty and ordered")
    if any(a.max() + pd.Timedelta(hours=horizon) >= b.min()
           for a, b in zip(all_blocks, all_blocks[1:])):
        raise ValueError("Target intervals overlap across chronological blocks")
    if train.min() - pd.Timedelta(hours=max(c.lookback_hours for c in configs) - 1) < frame.ds.min():
        raise ValueError("Insufficient input history before the first training origin")
    if test.max() + pd.Timedelta(hours=horizon) > frame.ds.max():
        raise ValueError("Final test plus horizon extends beyond observed data")
    if any(early_stop.max() + pd.Timedelta(hours=horizon,
                                           minutes=c.latency_minutes) >= folds[0].min()
           for c in configs):
        raise ValueError("Early-stop labels must be available before tuning validation")
    return pd.DatetimeIndex([origin for fold in folds for origin in fold])


def forecast_table(frame, forecast_origins, horizon):
    labels = frame.set_index("ds").y
    blocks = []
    for origin in forecast_origins:
        dates = future_grid(origin, horizon)
        blocks.append(pd.DataFrame({"origin_utc": origin, "timestamp_utc": dates,
                                    "horizon": np.arange(1, horizon + 1),
                                    "y": labels.reindex(dates).to_numpy()}))
    return pd.concat(blocks, ignore_index=True)


def predict_candidate(model, frame, forecast_origins):
    values, failures, timings = [], [], []
    for origin in forecast_origins:
        start = time.monotonic()
        try:
            forecast = model.predict(frame, origin)
            values.extend(forecast.speed_km_s.to_numpy())
        except Exception as exc:
            values.extend([np.nan] * model.config.horizon)
            failures.append({"origin_utc": origin.isoformat(),
                             "error": f"{type(exc).__name__}: {exc}"})
        timings.append(time.monotonic() - start)
    return values, failures, timings


def run_search(frame, candidates, train, early_stop, folds, test, output, *,
               metric="timse", references=None, metadata=None):
    if metric not in ("timse", "rmse"):
        raise ValueError("Selection metric must be timse or rmse")
    validation = check_schedule(frame, candidates, train, early_stop, folds, test)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    horizon = candidates[0]["config"].horizon
    manifest = {
        "metric": metric, "horizon": horizon,
        "selection_rule": "Complete finite validation trajectories; common observed targets; first candidate breaks ties",
        "train_origins": [t.isoformat() for t in train],
        "early_stop_origins": [t.isoformat() for t in early_stop],
        "validation_folds": [[t.isoformat() for t in fold] for fold in folds],
        "test_origins": [t.isoformat() for t in test],
        "candidates": [{"name": c["name"], "config": asdict(c["config"])} for c in candidates],
        "metadata": metadata or {},
        "availability": "Source available_at when present; otherwise assumed latency",
        "test_policy": "Winner and checkpoint frozen before final test targets are read or scored",
    }
    write_json(output / "manifest.json", manifest)
    predictions = forecast_table(frame, validation, horizon)
    diagnostics, trained = {}, {}
    for index, candidate in enumerate(candidates, 1):
        name = candidate["name"]
        candidate_dir = output / "candidates" / name
        candidate_dir.mkdir(parents=True)
        started = time.monotonic()
        try:
            model = AttentionGRUForecaster(candidate["config"]).fit(frame, train, early_stop)
            model.save(candidate_dir / "model.pt")
            write_json(candidate_dir / "training.json", model.training)
            values, failures, timings = predict_candidate(model, frame, validation)
            trained[name] = model
            error = None
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            values = [np.nan] * len(predictions)
            failures, timings = [], []
            error = f"{type(exc).__name__}: {exc}"
        predictions[name] = values
        diagnostics[name] = {"error": error, "failures": failures,
                             "complete_validation": bool(np.isfinite(values).all()),
                             "prediction_coverage": float(np.isfinite(values).mean()),
                             "training_and_validation_seconds": time.monotonic() - started,
                             "inference_seconds": timings}
        predictions.to_parquet(output / "validation_predictions.parquet", index=False)
        write_json(output / "diagnostics.json", diagnostics)
        print(f"[{index}/{len(candidates)}] {name}: "
              f"coverage={diagnostics[name]['prediction_coverage']:.1%}, "
              f"elapsed={diagnostics[name]['training_and_validation_seconds']:.1f}s", flush=True)
    eligible = [c["name"] for c in candidates if diagnostics[c["name"]]["complete_validation"]]
    if not eligible:
        raise ValueError("No candidate produced complete validation forecasts")
    report = evaluate_predictions(predictions, eligible, horizon=horizon,
                                  expected_origins=validation)
    if not report["common_pairs"]:
        raise ValueError("No observed targets in tuning validation")
    if metric == "timse" and predictions.loc[predictions.y.notna(), "y"].nunique() < 2:
        raise ValueError("TIMSE cannot select models when validation targets are constant")
    write_json(output / "validation_report.json", report)
    rows = []
    for candidate in candidates:
        name = candidate["name"]
        row = {"candidate": name, "eligible": name in eligible,
               "coverage": diagnostics[name]["prediction_coverage"],
               "seconds": diagnostics[name]["training_and_validation_seconds"],
               "error": diagnostics[name]["error"], **asdict(candidate["config"])}
        if name in eligible:
            score = report["metrics"][name]
            row.update({"validation_rmse": score["common"]["rmse"],
                        "validation_timse": score["timse_common"]})
            for fold_index, fold in enumerate(folds, 1):
                subset = predictions[predictions.origin_utc.isin(fold)]
                fold_report = evaluate_predictions(subset, [name], horizon=horizon,
                                                   expected_origins=fold)
                row[f"fold_{fold_index}_{metric}"] = (
                    fold_report["metrics"][name]["timse_common"] if metric == "timse"
                    else fold_report["metrics"][name]["common"]["rmse"])
        rows.append(row)
    key = f"validation_{metric}"
    ranking = pd.DataFrame(rows).sort_values(key, na_position="last", kind="stable")
    ranking.to_csv(output / "leaderboard.csv", index=False)
    winner_name = str(ranking.iloc[0].candidate)
    winner = next(c for c in candidates if c["name"] == winner_name)
    write_json(output / "best_config.json", {
        "name": winner_name, "config": asdict(winner["config"]),
        "metric": metric, "validation_score": float(ranking.iloc[0][key]),
        "checkpoint": f"candidates/{winner_name}/model.pt",
    })
    print(f"Selected on validation: {winner_name}; {metric}={ranking.iloc[0][key]:.3f}", flush=True)
    test_dir = output / "test"
    test_dir.mkdir()
    refs = references or {}
    test_predictions, test_report, full_report = evaluate(
        trained[winner_name], frame, test, lstm_reference=refs.get("lstm"),
        gru_reference=refs.get("gru"),
    )
    test_report["selected_candidate"] = winner_name
    test_predictions.to_parquet(test_dir / "predictions.parquet", index=False)
    write_json(test_dir / "report.json", test_report)
    write_json(test_dir / "comparison_without_recurrence.json", full_report)
    if (any(failure["model"] == "attention_gru" for failure in test_report["failures"])
            or not full_report["common_pairs"]):
        raise ValueError("Final test contains failed selected-model forecasts or no common targets")
    return winner, test_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mag-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--early-stop-start", required=True)
    parser.add_argument("--early-stop-hours", type=int, default=336)
    parser.add_argument("--validation-starts", nargs="+", required=True)
    parser.add_argument("--validation-hours", type=int, default=168)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--metric", choices=("timse", "rmse"), default="timse")
    candidate_source = parser.add_mutually_exclusive_group()
    candidate_source.add_argument("--candidates", type=Path,
                                  help="JSON list of named config overrides")
    candidate_source.add_argument("--search-space", type=Path,
                                  help="JSON object of parameter lists for a grid of at most 32 candidates")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--lstm-model", type=Path)
    parser.add_argument("--gru-model", type=Path)
    args = parser.parse_args(argv)
    try:
        if min(args.early_stop_hours, args.validation_hours, args.test_hours,
               args.step_hours) < 1:
            raise ValueError("Block lengths and cadence must be positive")
        base = AttentionGRUConfig(epochs=args.epochs, patience=args.patience,
                                  seed=args.seed, threads=args.threads,
                                  horizon=args.horizon)
        candidates = (read_search_space(args.search_space, base) if args.search_space
                      else read_candidates(args.candidates, base))
        groups = {c["config"].input_group for c in candidates}
        group = "plasma_mag" if "plasma_mag" in groups else "plasma" if "plasma" in groups else "speed"
        frame = load_lstm_data(args.data, args.mag_data, group)
        train = origins(args.train_start,
                        int((utc_hour(args.early_stop_start) - utc_hour(args.train_start)) /
                            pd.Timedelta(hours=1)) - args.horizon, args.step_hours)
        early_stop = origins(args.early_stop_start, args.early_stop_hours, args.step_hours)
        folds = [origins(start, args.validation_hours, args.step_hours)
                 for start in args.validation_starts]
        test = origins(args.test_start, args.test_hours, args.step_hours)
        references = {}
        for name, path, forecaster in (("lstm", args.lstm_model, LSTMForecaster),
                                       ("gru", args.gru_model, GRUForecaster)):
            if path:
                model = forecaster.load(path)
                if (model.config.horizon != args.horizon or
                        model.config.latency_minutes != candidates[0]["config"].latency_minutes or
                        set(model.columns) - set(frame.columns) or
                        test.min() < model.available_at):
                    raise ValueError(f"{name.upper()} reference is incompatible or unavailable")
                references[name] = model
        hashes = {}
        for path in (args.data, args.mag_data, args.candidates, args.search_space,
                     args.lstm_model, args.gru_model):
            if path:
                with path.open("rb") as stream:
                    hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "input_sha256": hashes,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob("*.py")},
            "versions": {name: version(name) for name in ("torch", "numpy", "pandas")},
        }
        run_search(frame, candidates, train, early_stop, folds, test, args.output_dir,
                   metric=args.metric, references=references, metadata=metadata)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
