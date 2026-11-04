#!/usr/bin/env python3
"""Figure of the real-series SOM scenarios (benchlab_forecasting.som sharp), in English.

  som_sharp_cases   four test issuances with a rise of at least 150 km/s, drawn at random: the sampled
                    real training series (thin), the medoid scenario, the submitted model and the
                    observations.

    .venv/bin/python scripts/plot_som_sharp.py --experiment data/experiments/som_sharp_2011_2026
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = "#e4e3df"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=Path("data/experiments/som_sharp_2011_2026"))
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    ens = np.load(args.experiment / "ensembles.npz")
    scen = np.load(args.experiment / "scenarios.npz")
    y, base, last = ens["y"], ens["forecast"], ens["last"]
    origins = pd.to_datetime(ens["origins"], utc=True)
    draws = ens["model_guided"]
    medoid_key = next(k for k in scen.files if k.startswith("medoid_model-guided_α__0"))
    medoid = scen[medoid_key]

    complete = np.isfinite(y).all(1) & np.isfinite(last)
    rises = np.flatnonzero(complete & (np.nanmax(y, 1) - last >= 150))
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(rises, 4, replace=False))
    hours = np.arange(1, 73)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, i in zip(axes.ravel(), picks):
        for k in range(draws.shape[1]):
            ax.plot(hours, draws[i, k], color="#eb6834", lw=0.6, alpha=0.15)
        rmse = lambda p: np.sqrt(np.mean((p - y[i]) ** 2))
        ax.plot(hours, y[i], color="#0b0b0b", lw=2.4, label="Observed")
        ax.plot(hours, base[i], color="#2a78d6", lw=2, label=f"Submitted model · RMSE {rmse(base[i]):.0f}")
        ax.plot(hours, medoid[i], color="#b5179e", lw=1.8, label=f"Medoid scenario · RMSE {rmse(medoid[i]):.0f}")
        ax.plot([], [], color="#eb6834", lw=1, alpha=0.6, label="Sampled real training series")
        ax.set_title(f"{origins[i]:%d %b %Y %H:%M} UTC", loc="left", fontsize=10)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("Lead time (hours after issuance)")
        ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    for ax in axes[:, 0]:
        ax.set_ylabel("Speed (km/s)")
    fig.suptitle("SOM scenarios built from real series (no averaging), model-guided, on test rises of ≥ 150 km/s",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    out = args.experiment / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "som_sharp_cases.png", dpi=160, facecolor="white")
    print(out / "som_sharp_cases.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
