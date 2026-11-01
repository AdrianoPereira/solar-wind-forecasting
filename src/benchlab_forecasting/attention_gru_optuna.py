"""Resumable TPE search for the causal attention GRU."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import optuna
import pandas as pd

from .attention_gru import AttentionGRUConfig, AttentionGRUForecaster
from .attention_gru_tuning import check_schedule, forecast_table, predict_candidate
from .data import utc_hour
from .gru import GRUForecaster
from .lstm import LSTMForecaster, load_lstm_data
from .lstm_cli import evaluate, origins, write_json
from .validation.metrics import regression_metrics, timse


DEFAULT_SPACE = {
    "attention_heads": [2, 4],
    "batch_size": [32, 64, 128],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "hidden_size": [16, 32, 64],
    "input_group": ["speed", "plasma", "plasma_mag"],
    "learning_rate": {"low": 0.0003, "high": 0.003, "log": True},
    "lookback_hours": [72, 168],
    "residual_scale": [0.0, 0.5, 1.0],
    "weight_decay": {"low": 0.000001, "high": 0.001, "log": True},
}
TUNABLE = {"attention_heads", "batch_size", "dropout", "fill_hours", "hidden_size",
           "input_group", "learning_rate", "lookback_hours", "max_age_hours",
           "residual_scale", "weight_decay"}


def read_space(path):
    space = DEFAULT_SPACE if path is None else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(space, dict) or not space or set(space) - TUNABLE:
        raise ValueError("Search space must be a nonempty object of supported config fields")
    for name, spec in space.items():
        if isinstance(spec, list):
            if not spec or any(value is None for value in spec):
                raise ValueError(f"{name}: categorical choices must be nonempty")
        elif isinstance(spec, dict):
            if not {"low", "high"} <= set(spec) or set(spec) - {"low", "high", "log", "step"}:
                raise ValueError(f"{name}: use low/high and optional log/step")
            if name in {"input_group", "attention_heads", "hidden_size", "lookback_hours"}:
                raise ValueError(f"{name}: use explicit categorical choices")
            if not (isinstance(spec["low"], (int, float)) and
                    isinstance(spec["high"], (int, float)) and
                    0 <= spec["low"] < spec["high"]):
                raise ValueError(f"{name}: invalid numeric bounds")
            if spec.get("log") and "step" in spec:
                raise ValueError(f"{name}: logarithmic sampling cannot use step")
        else:
            raise ValueError(f"{name}: use a choice list or numeric range")
    return space


def suggest_config(trial, base, space):
    overrides = {}
    for name in sorted(space):
        spec = space[name]
        if isinstance(spec, list):
            overrides[name] = trial.suggest_categorical(name, spec)
        elif isinstance(getattr(base, name), int):
            overrides[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                                step=int(spec.get("step", 1)),
                                                log=bool(spec.get("log", False)))
        else:
            overrides[name] = trial.suggest_float(name, float(spec["low"]),
                                                  float(spec["high"]),
                                                  step=spec.get("step"),
                                                  log=bool(spec.get("log", False)))
    return replace(base, **overrides)


def _max_lookback(space, base):
    spec = space.get("lookback_hours", [base.lookback_hours])
    return max(spec) if isinstance(spec, list) else int(spec["high"])


def _manifest(base, space, train, early_stop, folds, test, metric, metadata):
    return {
        "base_config": asdict(base), "space": space, "metric": metric,
        "train_origins": [t.isoformat() for t in train],
        "early_stop_origins": [t.isoformat() for t in early_stop],
        "validation_folds": [[t.isoformat() for t in fold] for fold in folds],
        "test_origins": [t.isoformat() for t in test],
        "metadata": metadata or {},
        "method": "Optuna TPE, independent trials, early stopping by inner RMSE; outer TIMSE/RMSE objective",
        "selection_rule": "Only complete finite outer-validation forecasts are eligible",
    }


def _leaderboard(study, output, metric):
    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "state": trial.state.name,
               "validation_score": trial.value,
               "validation_rmse": trial.user_attrs.get("validation_rmse"),
               "validation_timse": trial.user_attrs.get("validation_timse"),
               "coverage": trial.user_attrs.get("coverage"),
               "seconds": trial.user_attrs.get("seconds"),
               "error": trial.user_attrs.get("error"), **trial.params}
        rows.append(row)
    if rows:
        pd.DataFrame(rows).sort_values("validation_score", na_position="last",
                                       kind="stable").to_csv(output / "leaderboard.csv", index=False)


@contextmanager
def _study_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_study(frame, base, space, train, early_stop, folds, test, output, *,
              target_trials, metric="timse", seed=42, resume=False,
              finalize=False, references=None, metadata=None,
              prepare_only=False, worker_id=0, recover_running=False):
    if target_trials < 1 or metric not in ("timse", "rmse"):
        raise ValueError("Trials must be positive and metric must be timse or rmse")
    maximum = replace(base, lookback_hours=_max_lookback(space, base))
    validation = check_schedule(frame, [{"name": "study", "config": maximum}],
                                train, early_stop, folds, test)
    manifest = _manifest(base, space, train, early_stop, folds, test, metric, metadata)
    if prepare_only and resume:
        raise ValueError("--prepare requires a new study directory")
    if recover_running and (not resume or finalize):
        raise ValueError("--recover-running requires --resume and cannot finalize")
    if worker_id < 0:
        raise ValueError("worker_id must be nonnegative")
    with _study_lock(output):
        if resume:
            if not (output / "manifest.json").is_file():
                raise ValueError("Cannot resume without an existing manifest")
            if json.loads((output / "manifest.json").read_text()) != manifest:
                raise ValueError("Study inputs, schedule, source or search space changed")
        else:
            if any(path.name != "study.lock" for path in output.iterdir()):
                raise ValueError("Use a new, empty study directory or --resume")
            write_json(output / "manifest.json", manifest)
        sampler = optuna.samplers.TPESampler(seed=seed + worker_id, n_startup_trials=20,
                                             multivariate=True)
        study = optuna.create_study(study_name="attention_gru", direction="minimize",
                                    sampler=sampler, storage=f"sqlite:///{(output / 'study.db').resolve()}",
                                    load_if_exists=resume)
        if recover_running:
            for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        if len(study.trials) > target_trials:
            raise ValueError("Target trial count is below the existing study size")
    if prepare_only or recover_running:
        return None
    table = forecast_table(frame, validation, base.horizon)
    observed = table.y.to_numpy(dtype=float)
    if not np.isfinite(observed).any() or (metric == "timse" and
                                            np.unique(observed[np.isfinite(observed)]).size < 2):
        raise ValueError("Outer validation has insufficient observed target variation")
    while True:
        with _study_lock(output):
            if len(study.trials) >= target_trials:
                break
            trial = study.ask()
        if trial is not None:
            trial_dir = output / "trials" / f"{trial.number:04d}"
            trial_dir.mkdir(parents=True)
            started = time.monotonic()
            try:
                config = suggest_config(trial, base, space)
                model = AttentionGRUForecaster(config).fit(frame, train, early_stop)
                values, failures, timings = predict_candidate(model, frame, validation)
                predicted = np.asarray(values, dtype=float)
                if failures or not np.isfinite(predicted).all():
                    raise ValueError("Candidate did not produce all validation forecasts")
                summary = regression_metrics(observed, predicted)
                score_timse = timse(observed, predicted)
                if score_timse is None or summary["rmse"] is None:
                    raise ValueError("No finite validation score")
                model.save(trial_dir / "model.pt")
                write_json(trial_dir / "training.json", model.training)
                trial_table = table.copy()
                trial_table["attention_gru"] = predicted
                trial_table.to_parquet(trial_dir / "validation_predictions.parquet", index=False)
                score = score_timse if metric == "timse" else summary["rmse"]
                with _study_lock(output):
                    trial.set_user_attr("validation_rmse", summary["rmse"])
                    trial.set_user_attr("validation_timse", score_timse)
                    trial.set_user_attr("coverage", 1.0)
                    trial.set_user_attr("seconds", time.monotonic() - started)
                    trial.set_user_attr("best_epoch", model.training["best_epoch"])
                    trial.set_user_attr("inference_p50_seconds", float(np.median(timings)))
                    study.tell(trial, score)
                    _leaderboard(study, output, metric)
                print(f"[{trial.number + 1}/{target_trials}] {metric}={score:.3f}, "
                      f"RMSE={summary['rmse']:.2f}, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                with _study_lock(output):
                    trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                    trial.set_user_attr("seconds", time.monotonic() - started)
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    _leaderboard(study, output, metric)
                print(f"[{trial.number + 1}/{target_trials}] failed: {exc}", file=sys.stderr, flush=True)
    with _study_lock(output):
        states = [trial.state for trial in study.trials]
        if any(not state.is_finished() for state in states):
            if finalize:
                raise ValueError("Other workers still have running trials; finalize after they finish")
            return None
        eligible = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if not eligible:
            raise ValueError("No completed trial has a valid outer-validation score")
        best = study.best_trial
        best_dir = output / "trials" / f"{best.number:04d}"
        selected = AttentionGRUForecaster.load(best_dir / "model.pt")
        best_config = {"trial": best.number, "config": asdict(selected.config),
                       "metric": metric, "validation_score": best.value,
                       "checkpoint": str((best_dir / "model.pt").relative_to(output))}
        write_json(output / "best_config.json", best_config)
        if finalize:
            test_dir = output / "test"
            if test_dir.exists() and any(test_dir.iterdir()):
                raise ValueError("Final test has already been evaluated for this study")
            test_dir.mkdir(exist_ok=True)
            refs = references or {}
            predictions, report, complete = evaluate(
                selected, frame, test, lstm_reference=refs.get("lstm"),
                gru_reference=refs.get("gru"),
            )
            report["selected_trial"] = best.number
            predictions.to_parquet(test_dir / "predictions.parquet", index=False)
            write_json(test_dir / "report.json", report)
            write_json(test_dir / "comparison_without_recurrence.json", complete)
            if (any(failure["model"] == "attention_gru" for failure in report["failures"])
                    or not complete["common_pairs"]):
                raise ValueError("Final test has failed selected-model forecasts or no common targets")
        return best_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mag-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--early-stop-start", required=True)
    parser.add_argument("--early-stop-hours", type=int, default=168)
    parser.add_argument("--validation-starts", nargs="+", required=True)
    parser.add_argument("--validation-hours", type=int, default=336)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--metric", choices=("timse", "rmse"), default="timse")
    parser.add_argument("--space", type=Path, help="JSON search space; default spans nine parameters")
    parser.add_argument("--trials", type=int, default=100,
                        help="Total target trials, including any completed on a resumed run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--lstm-model", type=Path)
    parser.add_argument("--gru-model", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true",
                        help="Create an empty study before launching parallel workers")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Distinct nonnegative sampler seed offset for each worker")
    parser.add_argument("--recover-running", action="store_true",
                        help="Mark interrupted trials failed; use only after all workers stop")
    parser.add_argument("--finalize", action="store_true",
                        help="Evaluate the reserved test only after all target trials finish")
    args = parser.parse_args(argv)
    try:
        if min(args.early_stop_hours, args.validation_hours, args.test_hours,
               args.step_hours, args.trials) < 1:
            raise ValueError("Block lengths, cadence and trial count must be positive")
        base = AttentionGRUConfig(epochs=args.epochs, patience=args.patience,
                                  seed=args.seed, threads=args.threads, device=args.device,
                                  horizon=args.horizon)
        space = read_space(args.space)
        groups = space.get("input_group", [base.input_group])
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
                        model.config.latency_minutes != base.latency_minutes or
                        set(model.columns) - set(frame.columns) or test.min() < model.available_at):
                    raise ValueError(f"{name.upper()} reference is incompatible or unavailable")
                references[name] = model
        hashes = {}
        for path in (args.data, args.mag_data, args.space,
                     args.lstm_model, args.gru_model):
            if path:
                with path.open("rb") as stream:
                    hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
        source = [Path(__file__), Path(__file__).with_name("attention_gru.py"),
                  Path(__file__).with_name("lstm.py"),
                  Path(__file__).parent / "validation" / "metrics.py"]
        metadata = {"input_sha256": hashes,
                    "source_sha256": {str(p.relative_to(Path(__file__).parent)):
                                      hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                    "versions": {name: version(name) for name in ("optuna", "torch", "numpy", "pandas")}}
        best = run_study(frame, base, space, train, early_stop, folds, test,
                         args.output_dir, target_trials=args.trials, metric=args.metric,
                         seed=args.seed, resume=args.resume, finalize=args.finalize,
                         references=references, metadata=metadata,
                         prepare_only=args.prepare, worker_id=args.worker_id,
                         recover_running=args.recover_running)
        if best is not None:
            print(f"Best trial: {best['trial']}; {args.metric}={best['validation_score']:.3f}")
        elif args.prepare:
            print(f"Study prepared: {args.output_dir}")
        elif args.recover_running:
            print("Interrupted trials marked failed; restart the workers with --resume")
        else:
            print("Worker finished; other trials are still running")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
