#!/usr/bin/env python3
"""Result figures for the results deck, in the deck's dark palette.

Reuses the plotting code of scripts/plot_hybrid_results.py and scripts/plot_single_forecast.py on the
saved test predictions; only colors, fonts, sizes and titles change. "Hybrid" is the tuned LightGBM
model (the best single model), "Hybrid without coronal holes" its twin without AIA features.

Run from the repository root:
    OMP_NUM_THREADS=2 .venv/bin/python docs/presentation/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_hybrid_results as phr  # noqa: E402
import plot_single_forecast as psf  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
EXPERIMENTS = ROOT / "data" / "experiments"

# Deck palette ("Neon Corona").
BACKGROUND = "#16141F"  # card color on the slides
TEXT = "#F4F2FA"
MUTED = "#A7A3BA"
GRID = "#2C2938"
GREEN, PURPLE, YELLOW, ORANGE, GREY = "#39FF88", "#B026FF", "#FFC72C", "#FF8A3D", "#8A86A0"

STYLE = {
    "hybrid_mse": ("Hybrid", GREEN, "-"),
    "hybrid_timse": ("Hybrid, TIMSE weights", ORANGE, "-"),
    "recurrence": ("27-day recurrence", YELLOW, "-"),
    "hybrid_mse_no_ch": ("Hybrid without coronal holes", PURPLE, "-"),
    "persistence": ("Persistence", GREY, "--"),
}


def use_deck_style():
    for path in Path("/usr/share/fonts/urw-base35").glob("URWGothic-*.otf"):
        font_manager.fontManager.addfont(str(path))
    family = "URW Gothic" if any(f.name == "URW Gothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "text.color": TEXT, "axes.labelcolor": MUTED,
        "axes.facecolor": BACKGROUND, "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.labelsize": 11,
    })
    phr.OBSERVED_COLOR = psf.OBSERVED_COLOR = TEXT
    phr.MUTED_COLOR = psf.MUTED_COLOR = MUTED
    phr.GRID_COLOR = psf.GRID_COLOR = GRID
    phr.STYLE.clear()
    phr.STYLE.update(STYLE)
    psf.LABELS.update({k: v[0] for k, v in STYLE.items()})


def tidy(fig, size, rect=(0, 0, 1, 1)):
    """Fixed size so the picture fills its slide frame exactly; the slide title replaces the figure title."""
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    fig.set_size_inches(*size)
    for ax in fig.axes:
        ax.tick_params(labelsize=10.5)
    fig.tight_layout(rect=rect)
    return fig


def predictions():
    """Tuned LightGBM as the hybrid, with the HGB TIMSE-weighted variant kept for the metric figure."""
    tuned = pd.read_parquet(EXPERIMENTS / "hybrid_lightgbm_tuning" / "test_predictions.parquet")
    hgb = pd.read_parquet(EXPERIMENTS / "hybrid_2011_2026" / "test_predictions.parquet")
    frame = tuned.rename(columns={"lightgbm_tuned": "hybrid_mse", "lightgbm_tuned_no_ch": "hybrid_mse_no_ch"})
    frame = frame.merge(hgb[["origin_utc", "horizon", "hybrid_timse"]], on=["origin_utc", "horizon"],
                        how="left", validate="one_to_one")
    for column in ("origin_utc", "timestamp_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def rmse_by_lead(frame):
    saved = phr.STYLE.pop("hybrid_timse")
    fig = phr.plot_rmse_by_lead(frame)
    phr.STYLE["hybrid_timse"] = saved
    ax = fig.axes[0]
    ax.set_title("", loc="left")
    ax.get_legend().remove()
    for line in ax.get_lines():
        line.set_linewidth(2.6 if line.get_color() == GREEN else 2.0)
    ax.set_xlim(0, 92)
    return tidy(fig, (8.6, 5.2))


def single_forecast(frame, observed, origin):
    block = frame[frame.origin_utc == origin].sort_values("horizon")
    models = ["hybrid_mse", "hybrid_mse_no_ch", "recurrence"]
    psf.COLORS[:] = [GREEN, PURPLE, YELLOW]
    scores = {m: {"rmse": float(np.sqrt(np.mean((block[m] - block.y) ** 2)))} for m in [*models, "persistence"]}
    # Direct end labels carry the scores, so the legend (which hid the issuance line) goes.
    for m in models:
        psf.LABELS[m] = f"{STYLE[m][0]} · {scores[m]['rmse']:.0f}"
    fig = psf.plot(observed, block, models, origin, 96, scores)
    top, bottom = fig.axes
    top.get_legend().remove()
    end = block.timestamp_utc.iloc[-1]
    top.annotate(f"Persistence · {scores['persistence']['rmse']:.0f}", (end, block.persistence.iloc[-1]),
                 xytext=(end + pd.Timedelta(hours=3), block.persistence.iloc[-1]), va="center", fontsize=8.5,
                 color=MUTED, annotation_clip=False)
    peak = block.loc[block.y.idxmax()]
    top.annotate("Observed", (peak.timestamp_utc, peak.y), xytext=(-10, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=10, color=TEXT)
    top.annotate("History", (origin - pd.Timedelta(hours=84), 0.97), xycoords=("data", "axes fraction"),
                 va="top", fontsize=10, color=MUTED)
    for text in top.texts:
        if text.get_text().startswith("issuance"):
            text.xy = (text.xy[0], 1)
            text.set_va("top")
            text.xyann = (-6, -6)
    top.text(0.995, 0.02, "RMSE of this issuance in km/s", transform=top.transAxes, ha="right",
             fontsize=8.5, color=MUTED)
    top.set_title("Observed speed and 72-hour forecasts issued 3 Feb 2024 18:00 UTC", loc="left", fontsize=12)
    bottom.set_title("Absolute error by lead time (km/s)", loc="left", fontsize=12)
    bottom.set_ylabel("|Error|")
    for line in top.get_lines():
        if line.get_label().startswith("Observed (ground"):
            line.set_linewidth(2.8)
    return tidy(fig, (8.9, 5.2)), scores


def fixed_lead_window(frame, observed, start, end):
    fig = phr.plot_fixed_lead_window(frame, observed, start, end)
    return tidy(fig, (12.1, 5.0), rect=(0, 0.05, 1, 1))


def coronal_hole_lags(observed):
    holes = pd.read_parquet(ROOT / "data" / "features" / "coronal_holes_0193_6h.parquet")
    holes = holes[holes.error.isna()].set_index("origin_utc")
    fig = phr.plot_coronal_hole_lags(holes, observed)
    ax = fig.axes[0]
    ax.set_title("", loc="left")
    # West strips (reach Earth first) in sun yellow, east strips in neon purple.
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("sun", [YELLOW, ORANGE, PURPLE])
    for line, color in zip(ax.get_lines(), ramp(np.linspace(0, 1, len(phr.LONGITUDES)))):
        line.set_color(color)
        line.set_linewidth(2.4)
    ax.set_ylim(0, 0.42)
    return tidy(fig, (8.6, 5.2))


def timse_behavior(frame):
    fig = phr.plot_timse_behavior(frame)
    left, right = fig.axes
    left.set_title("Pooled TIMSE keeps improving…", loc="left", fontsize=12)
    right.set_title("…while the forecast gets worse", loc="left", fontsize=12)
    left.set_ylabel("TIMSE, millions of (km/s)³")
    for ax, color in ((left, GREEN), (right, GREEN)):
        line = ax.get_lines()[0]
        line.set_color(color)
        line.set_markerfacecolor(color)
        line.set_markeredgecolor(color)
    print("TIMSE at zero offset (millions):", round(float(left.get_lines()[0].get_ydata()[0]), 1))
    return tidy(fig, (8.9, 4.6))


def ngboost_calibration():
    """Coverage of central NGBoost intervals at every nominal level, and error by predicted spread."""
    frame = pd.read_parquet(EXPERIMENTS / "hybrid_ngboost_2011_2026" / "test_predictions.parquet").dropna(subset=["y"])
    levels = np.linspace(0.05, 0.95, 19)
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.9, 4.6), width_ratios=(1.15, 1))
    left.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1.4, label="Perfect calibration")
    variants = (("hybrid_mse_no_ch", "Without coronal holes", PURPLE), ("hybrid_mse", "With coronal holes", YELLOW))
    for name, label, color in variants:
        z = np.abs(frame.y - frame[name]) / frame[f"{name}_scale"]
        coverage = [(z <= norm.ppf(0.5 + level / 2)).mean() * 100 for level in levels]
        left.plot(levels * 100, coverage, color=color, lw=2.4, marker="o", ms=4, label=label)
    left.set_xlim(0, 100)
    left.set_ylim(0, 100)
    left.set_xlabel("Nominal interval (%)")
    left.set_ylabel("Test hours inside the interval (%)")
    left.set_title("Intervals are too narrow", loc="left", fontsize=12)
    left.legend(frameon=False, loc="upper left", fontsize=10)
    quintile = pd.qcut(frame.hybrid_mse_scale, 5, labels=False)
    error = (frame.y - frame.hybrid_mse).abs().groupby(quintile).mean()
    right.bar(error.index + 1, error.values, color=GREEN, width=0.62)
    for x, value in zip(error.index + 1, error.values):
        right.annotate(f"{value:.0f}", (x, value), xytext=(0, 4), textcoords="offset points", ha="center",
                       color=TEXT, fontsize=10.5)
    right.set_xticks(range(1, 6), ["lowest", "2", "3", "4", "highest"])
    right.set_xlabel("Predicted spread quintile")
    right.set_ylabel("Mean absolute error (km/s)")
    right.set_title("…but the spread ranks the errors", loc="left", fontsize=12)
    for ax in (left, right):
        phr.style(ax)
    right.grid(False, axis="x")
    return tidy(fig, (8.9, 4.6))


def main():
    use_deck_style()
    OUT.mkdir(exist_ok=True)
    frame = predictions()
    speed = pd.read_parquet(ROOT / "data" / "features" / "ace_speed_combined_2011_2026.parquet")
    observed = speed.set_index(pd.to_datetime(speed.timestamp_utc, utc=True)).speed_km_s
    single, scores = single_forecast(frame, observed, pd.Timestamp("2024-02-03T18:00", tz="UTC"))
    figures = {
        "rmse_by_lead": rmse_by_lead(frame),
        "single_forecast": single,
        "fixed_lead_window": fixed_lead_window(frame, observed, pd.Timestamp("2024-02-01", tz="UTC"),
                                               pd.Timestamp("2024-04-01", tz="UTC")),
        "coronal_hole_lags": coronal_hole_lags(observed),
        "timse_behavior": timse_behavior(frame),
        "ngboost_calibration": ngboost_calibration(),
    }
    for name, fig in figures.items():
        fig.savefig(OUT / f"{name}.png", dpi=220)
        plt.close(fig)
        print(OUT / f"{name}.png")
    observed_frame = frame.dropna(subset=["y"])
    print("single-forecast RMSE:", {m: round(v["rmse"], 1) for m, v in scores.items()})
    for h in (1, 72):
        sub = observed_frame[observed_frame.horizon == h]
        print(f"lead {h}:", {m: round(float(np.sqrt(((sub[m] - sub.y) ** 2).mean())), 1)
                            for m in ("hybrid_mse", "hybrid_mse_no_ch", "persistence", "recurrence")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
