"""Command-line interface for local experiments."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from .data import load_series
from .evaluation import backtest
from .features import load_regressors
from .models import ForecastConfig, ProphetForecaster, baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Previsão horária ACE com Prophet e baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("forecast", "backtest"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--time-column", default="timestamp_utc")
        p.add_argument("--target-column", default="speed_km_s")
        p.add_argument("--available-column")
        p.add_argument("--exogenous", choices=["none", "plasma", "plasma_mag"], default="none")
        p.add_argument("--mag-data", type=Path)
        for name, default in (("horizon", 72), ("train-days", 180), ("latency-minutes", 60),
                              ("max-age-hours", 24), ("min-points", 48),
                              ("fourier-order", 3), ("seed", 42),
                              ("exogenous-lag-hours", 96), ("exogenous-fill-hours", 6)):
            p.add_argument(f"--{name}", type=int, default=default)
        p.add_argument("--growth", choices=["flat", "linear"], default="flat")
        p.add_argument("--rotation-days", type=float, default=0)
        p.add_argument("--changepoint-prior-scale", type=float, default=0.05)
        p.add_argument("--seasonality-prior-scale", type=float, default=10.0)
        p.add_argument("--regressor-prior-scale", type=float, default=1.0)
        if command == "forecast":
            p.add_argument("--origin", required=True, help="Emissão UTC, na hora cheia")
            p.add_argument("--model", choices=["prophet", "persistence", "recurrence"], default="prophet")
        else:
            p.add_argument("--start", required=True, help="Primeira emissão UTC")
            p.add_argument("--end", required=True, help="Última emissão UTC, inclusiva")
            p.add_argument("--step-hours", type=int, default=6)
            p.add_argument("--models", nargs="+", choices=["prophet", "persistence", "recurrence"],
                           default=["prophet", "persistence", "recurrence"])
    args = parser.parse_args(argv)
    try:
        config = ForecastConfig(**{key: getattr(args, key) for key in ForecastConfig.__dataclass_fields__})
        series = load_series(args.data, args.time_column, args.target_column, args.available_column)
        regressors = (load_regressors(args.data, args.mag_data, args.exogenous)
                      if args.exogenous != "none" else None)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # One directory per experiment prevents accidentally mixing old and new artifacts.
        if any(args.output_dir.iterdir()):
            raise ValueError("Use um diretório de saída vazio para cada experimento")
        model = None
        if args.command == "forecast":
            if args.model == "prophet":
                model = ProphetForecaster(config).fit(series, args.origin, regressors)
                result = model.predict()
            else:
                result = baseline(series, args.origin, config, args.model)
            result.to_csv(args.output_dir / "forecast.csv", index=False)
            if model:
                model.save(args.output_dir / "model.json")
            report = {"origin": args.origin, "model": args.model,
                      "training": model.training if model else None,
                      "missing_predictions": int(result.speed_km_s.isna().sum())}
            failed = bool(report["missing_predictions"])
        else:
            result, report = backtest(series, args.start, args.end, config,
                                      args.step_hours, args.models, regressors)
            result.to_parquet(args.output_dir / "predictions.parquet", index=False)
            failed = bool(report["failures"]) or any(
                item["available"]["n"] == 0 for item in report["metrics"].values())
        with args.data.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update({"config": asdict(config), "data": str(args.data.resolve()),
                       "data_sha256": digest,
                       "arguments": {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
                       "versions": {name: version(name) for name in
                                    ("benchlab", "pandas", "numpy")},
                       "availability": args.available_column or "assumed_latency"})
        if (args.command == "forecast" and args.model == "prophet") or (
            args.command == "backtest" and "prophet" in args.models
        ):
            report["versions"]["prophet"] = version("prophet")
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Resultados: {args.output_dir}")
        return 1 if failed else 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
