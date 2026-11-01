#!/usr/bin/env python3
"""Plot one saved 72-hour issuance of several models against the observed series.

Reads saved test predictions; no model is refit. The upper panel shows the
observations before the issuance, the ground truth after it and every forecast;
the lower panel shows the absolute error of each model by lead time.
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

from benchlab_forecasting.data import load_series, utc_hour
from benchlab_forecasting.validation import timse


OBSERVED_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
# First five slots of the reference categorical palette (validated on adjacent pairs).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LABELS = {
    "prophet_tuned": "Tuned Prophet", "prophet_current": "Prophet",
    "transformer": "Transformer", "patch_transformer": "Patch Transformer",
    "attention_gru": "Attention GRU", "attention_lstm": "Attention LSTM",
    "lstm": "LSTM", "gru": "GRU", "persistence": "Persistence",
    "recurrence": "27-day recurrence",
    "hybrid_mse": "Hybrid", "hybrid_mse_no_ch": "Hybrid without coronal holes",
    "hybrid_timse": "Hybrid, TIMSE weights",
}
KEY = ["origin_utc", "timestamp_utc", "horizon"]


def merge_predictions(paths):
    merged = None
    for path in paths:
        frame = pd.read_parquet(path)
        for column in ("origin_utc", "timestamp_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        if merged is None:
            merged = frame
            continue
        new = [c for c in frame.columns if c not in merged.columns]
        merged = merged.merge(frame[[*KEY, "y", *new]], on=KEY, how="outer",
                              suffixes=("", "_other"), validate="one_to_one")
        if not np.allclose(merged.y.fillna(-1), merged.y_other.fillna(-1)):
            raise SystemExit(f"Error: observed targets differ in {path}")
        merged = merged.drop(columns="y_other")
    return merged


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


def style(ax):
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)


def plot(observed, block, models, origin, history_hours, scores):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.5), height_ratios=(2.4, 1))
    history = observed.loc[origin - pd.Timedelta(hours=history_hours):origin]
    truth = observed.reindex(block.timestamp_utc)
    top.plot(history.index, history.values, color=MUTED_COLOR, lw=1.6, label="Observed before issuance")
    top.plot(truth.index, truth.values, color=OBSERVED_COLOR, lw=2.4, label="Observed (ground truth)")
    if "persistence" in block:
        top.plot(block.timestamp_utc, block.persistence, color=MUTED_COLOR, lw=1.2, ls="--",
                 label=f"Persistence · RMSE {scores['persistence']['rmse']:.0f}")
    ends = []
    for model, color in zip(models, COLORS):
        top.plot(block.timestamp_utc, block[model], color=color, lw=2,
                 label=f"{LABELS[model]} · RMSE {scores[model]['rmse']:.0f}")
        bottom.plot(block.horizon, (block[model] - block.y).abs(), color=color, lw=2, label=LABELS[model])
        ends.append(float(block[model].iloc[-1]))
    end_time = block.timestamp_utc.iloc[-1]
    for model, end, height in zip(models, ends, spread(ends, gap=18)):
        top.annotate(LABELS[model], (end_time, end), xytext=(end_time + pd.Timedelta(hours=3), height),
                     textcoords="data", va="center", fontsize=8.5, color=MUTED_COLOR, annotation_clip=False)
    top.axvline(origin, color=MUTED_COLOR, lw=1, ls=":")
    top.annotate(f"issuance {origin:%d %b %Y %H:%M} UTC", (origin, 0), xycoords=("data", "axes fraction"),
                 xytext=(-4, 6), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=MUTED_COLOR)
    top.set_xlim(history.index.min(), end_time + pd.Timedelta(hours=26))
    top.xaxis.set_major_locator(mdates.DayLocator())
    top.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    top.set_ylabel("Solar-wind speed (km/s)")
    top.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    top.set_title("Observed ACE speed and every model's 72-hour forecast (RMSE in km/s for this issuance)",
                  loc="left", fontsize=11)
    style(top)
    bottom.set_xlim(0, 75)
    bottom.set_xticks([1, 12, 24, 36, 48, 60, 72])
    bottom.set_xlabel("Lead time (hours after issuance)")
    bottom.set_ylabel("|Error| (km/s)")
    bottom.set_title("Absolute error by lead time", loc="left", fontsize=11)
    style(bottom)
    fig.suptitle(f"Single 72-hour forecast with a complete observed series, issued {origin:%d %b %Y %H:%M} UTC",
                 x=0.06, ha="left", fontsize=13)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True,
                        help="Test predictions.parquet; repeat to merge models of several experiments")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--models", nargs="+", help="Up to five models; default: every complete model")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    origin = utc_hour(args.origin)
    predictions = merge_predictions(args.predictions)
    block = predictions[predictions.origin_utc == origin].sort_values("horizon")
    if block.empty:
        raise SystemExit(f"Error: no saved issuance at {origin}")
    if block.y.isna().any():
        raise SystemExit(f"Error: {int(block.y.isna().sum())} target hours are missing at {origin}")
    candidates = args.models or [c for c in block.columns
                                 if c not in (*KEY, "y", "persistence") and block[c].notna().all()]
    if len(candidates) > len(COLORS):
        raise SystemExit(f"Error: at most {len(COLORS)} models; pass --models")
    scores = {m: {"rmse": float(np.sqrt(np.mean((block[m] - block.y) ** 2))), "timse": timse(block.y, block[m])}
              for m in [*candidates, "persistence"] if m in block}
    # Keep each model's color stable by plotting in the given order, not by score.
    series = load_series(args.observations)
    observed = series.set_index(pd.to_datetime(series.ds, utc=True)).y

    fig = plot(observed, block, candidates, origin, args.history_hours, scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"single_forecast_{origin:%Y%m%dT%H%MZ}"
    written = []
    for fmt in args.formats:
        path = args.output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        written.append(str(path))
    plt.close(fig)
    pd.DataFrame(scores).T.rename_axis("model").sort_values("timse").to_csv(args.output_dir / f"{stem}_scores.csv")
    block.to_csv(args.output_dir / f"{stem}.csv", index=False)
    (args.output_dir / f"{stem}.json").write_text(json.dumps({
        "origin_utc": origin.isoformat(), "predictions": [str(p) for p in args.predictions],
        "observed_target_hours": int(block.y.notna().sum()), "models": candidates,
        "scores": scores, "files": written,
    }, indent=2), encoding="utf-8")
    for model, row in sorted(scores.items(), key=lambda item: item[1]["timse"]):
        print(f"{model:20s} RMSE={row['rmse']:7.2f}  TIMSE={row['timse']:12,.0f}")
    print(f"Figure: {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
