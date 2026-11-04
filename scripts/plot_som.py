#!/usr/bin/env python3
"""Figures of the two linked SOMs (benchlab_forecasting.som), in English.

  som_past_map      prototypes of map A: the 72 h before the cut-off, relative to the last value,
                    over the training windows that fall in each node (thin translucent lines);
  som_future_map    prototypes of map B: the 72 h after t0, relative to the same value, likewise;
  som_link          P(future node | past node) for four past regimes;
  som_case          one test issuance with a large rise: observed, model and SOM post-processing.

    .venv/bin/python scripts/plot_som.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MUTED, GRID = "#52514e", "#e4e3df"


def training_members(report, maps):
    """Training windows of both maps and the node each one falls in (recomputed from the data)."""
    from benchlab_forecasting.hybrid import load_plasma, origins_between
    from benchlab_forecasting.som import SOM, SOMConfig, future_windows, past_windows, recurrence_windows

    start, end, _ = report["config"]["splits"]["train"]
    plasma = load_plasma("data/ace_level2/ac_h2_swe_hourly.parquet",
                         "data/ace_hourly_2011_2026/ace_swepam_hourly.parquet")
    origins = origins_between(start[:19], end[:19])
    past, last, ok = past_windows(plasma.speed_km_s, origins)
    if maps["past"].shape[1] > 72:
        past = np.hstack([past, recurrence_windows(plasma.speed_km_s, origins, last)])
    future, ok_future = future_windows(plasma.speed_km_s, origins, last)
    usable = ok & ok_future
    nodes = {}
    for name, data in (("past", past[usable]), ("future", future[usable])):
        side = int(maps[f"{name}_grid"][:, 0].max()) + 1
        som = SOM(SOMConfig(side, side))
        som.prototypes = maps[name]
        nodes[name] = (data, som.bmu(data))
    return nodes


def grid_plot(prototypes, grid, counts, title, color, ylim, xlabel, members=None, per_node=60, seed=0):
    rows, cols = int(grid[:, 0].max()) + 1, int(grid[:, 1].max()) + 1
    size = 1.6 if members is not None else 1.25
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols + 1, 0.8 * size * rows + 1), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    share = counts / counts.sum()
    width = prototypes.shape[1]
    rng = np.random.default_rng(seed)
    for node, (r, c) in enumerate(grid.astype(int)):
        ax = axes[r, c]
        ax.axhline(0, color=GRID, lw=0.8)
        if members is not None:
            data, assigned = members
            inside = np.flatnonzero(assigned == node)
            shown = rng.choice(inside, min(per_node, len(inside)), replace=False) if len(inside) else inside
            x = np.arange(-71, 1) if width > 72 or color == "#2a78d6" else np.arange(1, 73)
            for k in shown:
                ax.plot(x, data[k, :72], color=color, lw=0.5, alpha=0.12)
                if width > 72:
                    ax.plot(np.arange(1, width - 71), data[k, 72:], color=MUTED, lw=0.4, alpha=0.08)
        if width > 72:  # map A with recurrence: past 72 h, then the 27-day-earlier speed of the next 72 h
            ax.plot(np.arange(-71, 1), prototypes[node, :72], color=color, lw=1.4)
            ax.plot(np.arange(1, width - 71), prototypes[node, 72:], color=MUTED, lw=1.0, ls="--")
            ax.axvline(0, color=GRID, lw=0.8)
        else:
            x = np.arange(-71, 1) if color == "#2a78d6" else np.arange(1, 73)
            ax.plot(x, prototypes[node], color="black" if members is not None else color,
                    lw=1.6 if members is not None else 1.4)
        if members is None:
            ax.set_facecolor(plt.cm.Greys(min(share[node] / share.max(), 1) * 0.25))
        ax.text(0.03, 0.9, f"{100 * share[node]:.1f}%", transform=ax.transAxes, fontsize=6, color=MUTED, va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    axes[0, 0].set_ylim(*ylim)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    fig.text(0.5, 0.01, xlabel, ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=0.2, w_pad=0.2)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=Path("data/experiments/som_2011_2026"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--members", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the training windows of each node behind its prototype")
    parser.add_argument("--per-node", type=int, default=60, help="Most windows drawn per node")
    args = parser.parse_args(argv)
    out = args.output_dir or args.experiment / "figures"
    out.mkdir(parents=True, exist_ok=True)
    maps = np.load(args.experiment / "maps.npz")
    counts = np.load(args.experiment / "train_assignments.npz")
    report = json.loads((args.experiment / "report.json").read_text())
    past, future, link = maps["past"], maps["future"], maps["link"]

    members = training_members(report, maps) if args.members else {"past": None, "future": None}
    if args.members:
        span = float(np.percentile(np.abs(np.vstack([members["past"][0][:, :72], members["future"][0]])), 99.5))
    else:
        span = max(np.abs(past).max(), np.abs(future).max())
    lim = (-span * 1.05, span * 1.05)
    note = (f"; thin lines: up to {args.per_node} training windows of the node, black: prototype"
            if args.members else "")
    figures = {
        "som_past_map": grid_plot(past, maps["past_grid"], counts["past_counts"],
                                  "Map A: solar-wind speed in the 72 h before the cut-off, minus the last value "
                                  "(share of 2011–2021 issuances)", "#2a78d6", lim,
                                  ("each panel: −72 h … cut-off" + ("; dashed: speed 27 days before the next 72 h" if past.shape[1] > 72
                                   else "") + "; y range ±%.0f km/s" % span + note),
                                  members["past"], args.per_node),
        "som_future_map": grid_plot(future, maps["future_grid"], counts["future_counts"],
                                    "Map B: the 72 h after t0, minus the same value (share of 2011–2021 issuances)",
                                    "#eb6834", lim, "each panel: +1 … +72 h; y range ±%.0f km/s" % span + note,
                                    members["future"], args.per_node),
    }

    # P(j | i) on map B for the four most frequent past regimes.
    top = np.argsort(counts["past_counts"])[::-1][:4]
    side = int(maps["future_grid"][:, 0].max()) + 1
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), height_ratios=(1, 1.3))
    for k, node in enumerate(top):
        axes[0, k].plot(np.arange(-71, 1), past[node, :72], color="#2a78d6", lw=1.8)
        if past.shape[1] > 72:
            axes[0, k].plot(np.arange(1, 73), past[node, 72:], color=MUTED, lw=1.2, ls="--")
        axes[0, k].axhline(0, color=GRID, lw=0.8)
        axes[0, k].set_ylim(*lim)
        axes[0, k].set_title(f"past node {node} ({100 * counts['past_counts'][node] / counts['past_counts'].sum():.1f}%)",
                             fontsize=9, loc="left")
        image = axes[1, k].imshow(link[node].reshape(side, side), cmap="Oranges", vmin=0)
        axes[1, k].set_xticks([])
        axes[1, k].set_yticks([])
        fig.colorbar(image, ax=axes[1, k], fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("km/s vs last value")
    axes[1, 0].set_ylabel("P(future node | past node)\non map B")
    fig.suptitle("The link: which future shapes follow the four most frequent past shapes (training years)",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    figures["som_link"] = fig

    # Case: the test issuance with the largest RMSE gain among rises of ≥ 150 km/s.
    frame = pd.read_parquet(args.experiment / "test_predictions.parquet")
    base, guided = "base (submission)", "model-guided SOM (chosen)"
    analog = "SOM analog only: Σ P(j|i)·prototype"
    per = frame.groupby("origin_utc").apply(lambda g: pd.Series({
        "rise": g.y.max() - g["last"].iloc[0], "complete": g.y.notna().mean(),
        "gain": np.sqrt(np.nanmean((g.y - g[base]) ** 2)) - np.sqrt(np.nanmean((g.y - g[guided]) ** 2))}))
    pick = per[(per.rise >= 150) & (per.complete == 1)].gain.idxmax()
    g = frame[frame.origin_utc == pick]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(g.horizon, g.y, color="#0b0b0b", lw=2.4, label="Observed")
    for name, color, label in ((base, "#2a78d6", "Submitted model"), (guided, "#eb6834", "Model-guided SOM"),
                               (analog, "#1baf7a", "SOM analog only")):
        r = np.sqrt(np.nanmean((g.y - g[name]) ** 2))
        ax.plot(g.horizon, g[name], color=color, lw=2, label=f"{label} · RMSE {r:.0f}")
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel("Lead time (hours after issuance)")
    ax.set_ylabel("Speed (km/s)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"Test issuance {pd.Timestamp(pick):%d %b %Y %H:%M} UTC: largest SOM gain among rises ≥ 150 km/s "
                 f"(chosen σ = {report['choice']['sigma']}, maps {report['choice']['past_side']}² / "
                 f"{report['choice']['future_side']}²)", loc="left", fontsize=10)
    fig.tight_layout()
    figures["som_case"] = fig

    for name, figure in figures.items():
        figure.savefig(out / f"{name}.png", dpi=160, facecolor="white")
        plt.close(figure)
        print(out / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
