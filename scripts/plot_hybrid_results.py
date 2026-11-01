#!/usr/bin/env python3
"""Summary figures of the hybrid experiment, from saved predictions and features only.

Writes, in English (PNG and PDF):
  rmse_by_lead          RMSE of every model for each lead hour over the whole test;
  fixed_lead_window     24 h and 72 h lead forecasts against observations in one window;
  coronal_hole_lags     correlation between coronal-hole fraction and later speed;
  timse_behavior        how TIMSE and RMSE respond to a constant offset.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchlab_forecasting.validation import timse

OBSERVED_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
# Same model-to-color mapping as the single-forecast figures.
STYLE = {
    "hybrid_mse": ("Hybrid", "#2a78d6", "-"),
    "hybrid_timse": ("Hybrid, TIMSE weights", "#eb6834", "-"),
    "recurrence": ("27-day recurrence", "#1baf7a", "-"),
    "hybrid_mse_no_ch": ("Hybrid without coronal holes", "#eda100", "-"),
    "persistence": ("Persistence", MUTED_COLOR, "--"),
}
LONGITUDES = ("+15_+30", "+0_+15", "-15_+0", "-30_-15", "-45_-30", "-60_-45")


def style(ax):
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)


def label_ends(ax, x, values, gap):
    """Direct labels at the right end of each line, nudged apart."""
    names = list(values)
    ends = np.array([values[n] for n in names], dtype=float)
    order = np.argsort(ends)
    placed = ends[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    heights = np.empty_like(placed)
    heights[order] = placed - (placed - ends[order]).mean()
    for name, end, height in zip(names, ends, heights):
        ax.annotate(STYLE[name][0], (x, end), xytext=(x, height), textcoords="data", va="center",
                    fontsize=8.5, color=MUTED_COLOR, annotation_clip=False)


def plot_rmse_by_lead(predictions):
    observed = predictions.dropna(subset=["y"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ends = {}
    for model, (label, color, line) in STYLE.items():
        rmse = ((observed[model] - observed.y) ** 2).groupby(observed.horizon).mean() ** 0.5
        ax.plot(rmse.index, rmse.values, color=color, ls=line, lw=2, label=label)
        ends[model] = rmse.iloc[-1]
    label_ends(ax, 74, ends, gap=4)
    ax.set_xlim(0, 90)
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel("Lead time (hours after issuance)")
    ax.set_ylabel("RMSE (km/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    n = observed.origin_utc.nunique()
    ax.set_title(f"RMSE by lead time on the test period, Jan 2024 – Oct 2026 ({n:,} issuances every 6 h)",
                 loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    return fig


def plot_fixed_lead_window(predictions, observed, start, end):
    window = predictions[(predictions.timestamp_utc >= start) & (predictions.timestamp_utc < end)]
    models = ["hybrid_mse", "recurrence", "hybrid_mse_no_ch"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, sharey=True)
    truth = observed.loc[start:end]
    for ax, lead in zip(axes, (24, 72)):
        block = window[window.horizon == lead].sort_values("timestamp_utc")
        ax.plot(truth.index, truth.values, color=OBSERVED_COLOR, lw=1.6, label="Observed")
        for model in models:
            label, color, line = STYLE[model]
            ax.plot(block.timestamp_utc, block[model], color=color, ls=line, lw=1.6, label=label)
        valid = block.dropna(subset=["y"])
        errors = ", ".join(f"{STYLE[m][0]} {np.sqrt(np.mean((valid[m] - valid.y) ** 2)):.0f}" for m in models)
        ax.set_title(f"{lead} h lead time · RMSE (km/s): {errors}", loc="left", fontsize=10)
        ax.set_ylabel("Speed (km/s)")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        style(ax)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle(f"Forecasts verified at a fixed lead time, {pd.Timestamp(start):%d %b} – "
                 f"{pd.Timestamp(end) - pd.Timedelta(days=1):%d %b %Y} (each point is one 6-hourly issuance)",
                 x=0.06, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def spearman(a, b):
    mask = a.notna() & b.notna()
    return np.corrcoef(a[mask].rank(), b[mask].rank())[0, 1]


def plot_coronal_hole_lags(holes, observed):
    smooth = observed.rolling(6, min_periods=3).mean()
    lags = np.arange(0, 9.25, 0.25)
    colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.95, len(LONGITUDES)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    curves = {}
    for longitude in LONGITUDES:
        x = holes[f"ch_t25_lat60_lon{longitude}"]
        later = lambda days: pd.Series(smooth.reindex(x.index + pd.Timedelta(days=days)).to_numpy(), index=x.index)
        curves[longitude] = np.array([spearman(x, later(lag)) for lag in lags])
    for (longitude, values), color in zip(curves.items(), colors):
        ax.plot(lags, values, color=color, lw=2)
        peak = int(np.nanargmax(values))
        # Label right of the peak when a neighbouring curve passes above it there.
        covered = any(other[peak] > values[peak] - 0.01 for name, other in curves.items() if name != longitude)
        west, east = longitude.split("_")
        ax.annotate(f"{int(west):+d}° to {int(east):+d}°\npeak {lags[peak]:.1f} d", (lags[peak], values[peak]),
                    xytext=(16, 4) if covered else (0, 8), textcoords="offset points",
                    ha="left" if covered else "center", va="bottom", fontsize=8, color=MUTED_COLOR)
    ax.set_xlabel("Days between the AIA image and the ACE speed")
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(bottom=0, top=0.45)
    ax.set_title("Coronal-hole area in each 15° longitude strip vs solar-wind speed days later, 2011–2026 "
                 "(east = negative)", loc="left", fontsize=10.5)
    style(ax)
    fig.tight_layout()
    return fig


def plot_timse_behavior(predictions):
    observed = predictions.dropna(subset=["y"])
    y, base = observed.y.to_numpy(), observed.hybrid_mse.to_numpy()
    offsets = np.arange(0, 210, 10)
    totals = np.array([timse(y, base + d) for d in offsets]) / 1e6
    rmses = np.array([np.sqrt(np.mean((base + d - y) ** 2)) for d in offsets])
    references = {name: timse(y, observed[name]) / 1e6 for name in ("persistence", "recurrence", "hybrid_timse")}
    constant = timse(y, np.full_like(y, 700.0)) / 1e6
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.plot(offsets, totals, color=STYLE["hybrid_mse"][1], lw=2, marker="o", ms=4,
              label="Hybrid + constant offset")
    # Reference scores as lines that stop at the last offset, labelled in the free space to their right.
    lines = [(f"{STYLE[n][0]} {v:.1f}", v, STYLE[n][1], STYLE[n][2]) for n, v in references.items()]
    lines.append((f"Constant 700 km/s {constant:.1f}", constant, OBSERVED_COLOR, ":"))
    for text, value, color, line in lines:
        left.hlines(value, offsets[0], offsets[-1], color=color, ls=line, lw=1.2)
        left.annotate(text, (offsets[-1], value), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=8, color=MUTED_COLOR)
    left.set_xlim(0, 310)
    left.set_xticks(range(0, 201, 50))
    left.set_xlabel("Offset added to every hybrid forecast (km/s)")
    left.set_ylabel("TIMSE over the whole test (millions of (km/s)³)")
    left.set_title("TIMSE keeps improving as forecasts are pushed up", loc="left", fontsize=11)
    style(left)
    right.plot(offsets, rmses, color=STYLE["hybrid_mse"][1], lw=2, marker="o", ms=4)
    right.set_xlabel("Offset added to every hybrid forecast (km/s)")
    right.set_ylabel("RMSE (km/s)")
    right.set_ylim(bottom=0)
    right.set_title("…while the forecast itself gets worse", loc="left", fontsize=11)
    style(right)
    fig.suptitle("TIMSE pooled over the full test period, Jan 2024 – Oct 2026", x=0.06, ha="left", fontsize=12)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path,
                        default=Path("data/experiments/hybrid_2011_2026/test_predictions_all.parquet"))
    parser.add_argument("--observations", type=Path,
                        default=Path("data/features/ace_speed_combined_2011_2026.parquet"))
    parser.add_argument("--coronal-holes", type=Path, default=Path("data/features/coronal_holes_0193_6h.parquet"))
    parser.add_argument("--window", nargs=2, default=["2024-02-01", "2024-04-01"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/hybrid_2011_2026/figures"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args(argv)

    predictions = pd.read_parquet(args.predictions)
    speed = pd.read_parquet(args.observations)
    observed = speed.set_index(pd.to_datetime(speed.timestamp_utc, utc=True)).speed_km_s
    holes = pd.read_parquet(args.coronal_holes)
    holes = holes[holes.error.isna()].set_index("origin_utc")
    start, end = (pd.Timestamp(t, tz="UTC") for t in args.window)
    figures = {
        "rmse_by_lead": plot_rmse_by_lead(predictions),
        "fixed_lead_window": plot_fixed_lead_window(predictions, observed, start, end),
        "coronal_hole_lags": plot_coronal_hole_lags(holes, observed),
        "timse_behavior": plot_timse_behavior(predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        for fmt in args.formats:
            fig.savefig(args.output_dir / f"{name}.{fmt}", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(args.output_dir / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
