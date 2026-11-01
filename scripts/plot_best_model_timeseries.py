#!/usr/bin/env python3
"""Plot saved test forecasts of the best models against observed ACE speed.

Reads the predictions of a finished experiment; no model is refit. Ranks the
complete models by TIMSE on their common pairs and draws the best one, the best
neural network and the 27-day recurrence against the ground truth.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchlab_forecasting.data import load_series
from benchlab_forecasting.validation import timse


OBSERVED_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
# First three slots of the reference categorical palette (validated all-pairs).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
NEURAL = ("transformer", "patch_transformer", "attention_gru", "attention_lstm", "lstm", "gru")
LABELS = {
    "prophet_tuned": "Tuned Prophet", "prophet_current": "Prophet",
    "transformer": "Transformer", "patch_transformer": "Patch Transformer",
    "attention_gru": "Attention GRU", "attention_lstm": "Attention LSTM",
    "lstm": "LSTM", "gru": "GRU", "persistence": "Persistence",
    "recurrence": "27-day recurrence",
}
LEADS = (24, 48, 72)


def style(ax):
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def rank_models(predictions):
    """TIMSE and RMSE of every complete model on the pairs common to all of them."""
    candidates = [c for c in predictions.columns
                  if c not in ("origin_utc", "timestamp_utc", "horizon", "y")
                  and predictions[c].notna().all()]
    common = predictions.dropna(subset=["y", *candidates])
    scores = pd.DataFrame({
        "model": candidates,
        "timse": [timse(common.y, common[c]) for c in candidates],
        "rmse": [float(np.sqrt(np.mean((common[c] - common.y) ** 2))) for c in candidates],
    }).sort_values("timse", ignore_index=True)
    return scores, len(common)


def plot_issuances(observed, predictions, models, scores):
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 3.1 * len(models)), sharex=True, sharey=True)
    start = predictions.origin_utc.min() - pd.Timedelta(days=3)
    end = predictions.timestamp_utc.max()
    context = observed.loc[start:end]
    for ax, model, color in zip(np.atleast_1d(axes), models, COLORS):
        for _, block in predictions.groupby("origin_utc"):
            ax.plot(block.timestamp_utc, block[model], color=color, lw=1, alpha=0.35)
        ax.plot(context.index, context.values, color=OBSERVED_COLOR, lw=2, label="Observed (ground truth)")
        ax.plot([], [], color=color, lw=1.5, label=f"{LABELS[model]}: 28 forecasts of 72 h")
        ax.axvline(predictions.origin_utc.min(), color=MUTED_COLOR, lw=1, ls="--")
        row = scores.set_index("model").loc[model]
        ax.set_title(f"{LABELS[model]}  ·  TIMSE {row.timse:,.0f} (km/s)³  ·  RMSE {row.rmse:.1f} km/s",
                     loc="left", fontsize=11)
        ax.set_ylabel("Speed (km/s)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        style(ax)
    np.atleast_1d(axes)[0].annotate("first issuance", (predictions.origin_utc.min(), 1), xycoords=("data", "axes fraction"),
                                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=MUTED_COLOR)
    fig.suptitle("Every 6-hourly issuance against observed ACE solar-wind speed, 15–24 Nov 2019",
                 x=0.07, ha="left", fontsize=13)
    fig.tight_layout()
    return fig


def spread(values, gap):
    """Shift sorted label positions apart by at least gap, keeping their mean."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed -= (placed - np.asarray(values, dtype=float)[order]).mean()
    result = np.empty_like(placed)
    result[order] = placed
    return result


def plot_fixed_leads(predictions, models):
    fig, axes = plt.subplots(len(LEADS), 1, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, lead in zip(axes, LEADS):
        block = predictions[predictions.horizon == lead].sort_values("timestamp_utc")
        ax.plot(block.timestamp_utc, block.y, color=OBSERVED_COLOR, lw=2, marker="o", ms=4,
                label="Observed (ground truth)")
        ends = []
        for model, color in zip(models, COLORS):
            ax.plot(block.timestamp_utc, block[model], color=color, lw=2, marker="o", ms=4,
                    label=LABELS[model])
            ends.append(block.dropna(subset=[model]).iloc[-1])
        heights = spread([end[m] for end, m in zip(ends, models)], gap=22)
        for model, end, height in zip(models, ends, heights):
            ax.annotate(LABELS[model], (end.timestamp_utc, end[model]), xytext=(end.timestamp_utc + pd.Timedelta(hours=5), height),
                        textcoords="data", va="center", fontsize=8, color=MUTED_COLOR,
                        annotation_clip=False)
        valid = block.dropna(subset=["y"])
        errors = ", ".join(f"{LABELS[m]} {np.sqrt(np.mean((valid[m] - valid.y) ** 2)):.0f}" for m in models)
        ax.set_title(f"{lead} h lead time  ·  RMSE (km/s): {errors}", loc="left", fontsize=10)
        ax.set_ylabel("Speed (km/s)")
        style(ax)
    # Shared x-axis: leave room after the latest valid time for the direct labels.
    axes[-1].set_xlim(right=predictions.timestamp_utc.max() + pd.Timedelta(hours=30))
    axes[-1].set_xlabel("Valid time (UTC)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle("Forecasts at a fixed lead time, plotted at the hour they verify",
                 x=0.07, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_selected(observed, predictions, models, count=6):
    origins = sorted(predictions.origin_utc.unique())
    chosen = [origins[i] for i in np.linspace(0, len(origins) - 1, count).round().astype(int)]
    fig, axes = plt.subplots(2, count // 2, figsize=(13, 7), sharey=True)
    for ax, origin in zip(axes.flat, chosen):
        origin = pd.Timestamp(origin)
        block = predictions[predictions.origin_utc == origin]
        history = observed.loc[origin - pd.Timedelta(hours=48):origin]
        truth = observed.loc[origin:block.timestamp_utc.max()]
        ax.plot(history.index, history.values, color=MUTED_COLOR, lw=1.5, label="Observed before issuance")
        ax.plot(truth.index, truth.values, color=OBSERVED_COLOR, lw=2, label="Observed (ground truth)")
        for model, color in zip(models, COLORS):
            ax.plot(block.timestamp_utc, block[model], color=color, lw=2, label=LABELS[model])
        ax.axvline(origin, color=MUTED_COLOR, lw=1, ls="--")
        ax.set_title(f"Issued {origin:%d %b %H:%M} UTC", loc="left", fontsize=10)
        style(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Speed (km/s)")
    for ax in axes[-1]:
        ax.set_xlabel("Day of Nov 2019 (UTC)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle("Selected 72-hour forecasts with 48 hours of prior observations (dashed line = issuance)",
                 x=0.05, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True,
                        help="Test predictions.parquet; repeat to merge models of several experiments")
    parser.add_argument("--observations", type=Path, default=Path("data/ace_hourly/ace_swepam_hourly.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    key = ["origin_utc", "timestamp_utc", "horizon", "y"]
    predictions = None
    for path in args.predictions:
        frame = pd.read_parquet(path)
        if predictions is None:
            predictions = frame
            continue
        new = [c for c in frame.columns if c not in predictions.columns]
        merged = predictions.merge(frame[[*key, *new]], on=key[:3], how="outer",
                                   suffixes=("", "_other"), validate="one_to_one")
        if not np.allclose(merged.y.fillna(-1), merged.y_other.fillna(-1)):
            raise SystemExit(f"Error: observed targets differ in {path}")
        predictions = merged.drop(columns="y_other")
    for column in ("origin_utc", "timestamp_utc"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)

    scores, pairs = rank_models(predictions)
    best = scores.model[0]
    neural = next(m for m in scores.model if m in NEURAL and m != best)
    models = [best, neural, "recurrence"]
    series = load_series(args.observations)
    observed = series.set_index(pd.to_datetime(series.ds, utc=True)).y

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "best_model_issuances": plot_issuances(observed, predictions, [best, neural], scores),
        "fixed_lead_timeseries": plot_fixed_leads(predictions, models),
        "selected_issuances": plot_selected(observed, predictions, models),
    }
    written = []
    for name, fig in figures.items():
        for fmt in args.formats:
            path = args.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            written.append(str(path))
        plt.close(fig)
    scores.to_csv(args.output_dir / "ranking.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "predictions": [str(p) for p in args.predictions], "common_pairs": pairs,
        "best_by_timse": best, "best_neural_by_timse": neural, "plotted": models,
        "note": "Recurrence is not complete, so it is drawn but excluded from the ranking",
        "files": written,
    }, indent=2), encoding="utf-8")
    print(scores.to_string(index=False))
    print(f"Figures: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
