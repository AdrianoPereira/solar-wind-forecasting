#!/usr/bin/env python3
"""Benchmark figure for the README: RMSE by lead time of the best model of each family.

    .venv/bin/python scripts/plot_benchmark.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Hybrid boosting": "#2a78d6", "Neural network": "#eb6834", "Statistical": "#1baf7a",
          "Baseline": "#52514e"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/experiments/benchmark_2024_2026"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/benchmark_rmse_by_lead.png"))
    args = parser.parse_args(argv)
    table = pd.read_csv(args.benchmark_dir / "benchmark.csv")
    predictions = pd.read_parquet(args.benchmark_dir / "predictions.parquet")
    common = predictions.dropna(subset=["y", *table.column])
    # Best model of each family, plus both baselines.
    chosen = table[table.family != "Baseline"].groupby("family").head(1)
    chosen = pd.concat([chosen, table[table.family == "Baseline"]])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for row in chosen.itertuples():
        rmse = ((common[row.column] - common.y) ** 2).groupby(common.horizon).mean() ** 0.5
        label = row.model if row.family == "Baseline" else f"{row.model} ({row.family.lower()})"
        ax.plot(rmse.index, rmse.values, lw=2.4 if row.Index == chosen.index[0] else 1.8,
                ls="--" if row.column == "persistence" else ("-." if row.column == "recurrence" else "-"),
                color=COLORS[row.family] if row.column != "recurrence" else "#eda100", label=label)
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax.set_xlim(0, 73)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lead time (hours after issuance)")
    ax.set_ylabel("RMSE (km/s)")
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    n = common.origin_utc.nunique()
    ax.set_title(f"Best model of each family on the same test issuances, Jan 2024 – Oct 2026 ({n:,} issuances)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
