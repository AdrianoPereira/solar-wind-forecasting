"""Validate saved CSV/Parquet predictions without refitting any models."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .reporting import evaluate_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validação independente de previsões horárias")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--start", help="Primeira emissão esperada UTC")
    parser.add_argument("--end", help="Última emissão esperada UTC")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if bool(args.start) != bool(args.end) or args.step_hours < 1:
            raise ValueError("Informe --start e --end juntos, com passo positivo")
        expected = (pd.date_range(args.start, args.end, freq=f"{args.step_hours}h", tz="UTC")
                    if args.start else None)
        frame = (pd.read_parquet(args.predictions) if args.predictions.suffix == ".parquet"
                 else pd.read_csv(args.predictions))
        report = evaluate_predictions(frame, args.models, horizon=args.horizon,
                                      expected_origins=expected)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"Relatório: {args.output}")
        return 0 if report["common_pairs"] else 1
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
