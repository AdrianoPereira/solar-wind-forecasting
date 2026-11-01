"""English figures from saved forecasts and backtests; no model fitting.

Run with python -m benchlab_forecasting.plotting --experiment-dir PATH.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_series, utc_hour
from .validation.reporting import evaluate_predictions, validate_predictions


LABELS = {"prophet": "Prophet", "lstm": "LSTM", "gru": "GRU", "persistence": "Persistence",
          "recurrence": "27-day recurrence"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#333333"]


def label(model):
    return LABELS.get(model, model.replace("_", " ").title())


def save_figure(fig, output, name, formats, dpi):
    for extension in formats:
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Time (UTC)")


def plot_learning_curve(history, model_label="LSTM"):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(history.epoch, history.train_rmse_km_s, label="Training", color=COLORS[0])
    ax.plot(history.epoch, history.validation_rmse_km_s, label="Validation", color=COLORS[1])
    best = history.loc[history.validation_rmse_km_s.idxmin(), "epoch"]
    ax.axvline(best, color="#555555", linestyle="--", label=f"Selected epoch: {int(best)}")
    ax.set(title=f"{model_label} learning curves | Selection uses validation only",
           xlabel="Epoch", ylabel="RMSE (km/s)")
    ax.legend()
    return fig


def plot_trajectory(data, models, origin):
    block = data.loc[data.origin_utc.eq(origin)].sort_values("horizon")
    if block.empty:
        raise ValueError("The selected origin is absent from the forecast calendar")
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    ax.plot(block.timestamp_utc, block.y, color="#222222", lw=2, label="Observed")
    for i, model in enumerate(models):
        ax.plot(block.timestamp_utc, block[model], color=COLORS[i % len(COLORS)],
                lw=1.7, linestyle="--", label=label(model))
    ax.set(title=f"Solar wind speed forecast | Issued {origin:%Y-%m-%d %H:%M} UTC",
           ylabel="Solar wind speed (km/s)")
    date_axis(ax)
    ax.legend(ncols=min(4, len(models) + 1))
    return fig


def plot_backtest(data, models, report):
    """Yield figures using common forecast-origin/lead pairs for all errors."""
    metrics = report["metrics"]
    common = data.y.notna() & data[models].notna().all(axis=1)
    n = int(common.sum())
    names = [label(model) for model in models]
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("rmse", "mae", "bias"),
                                ("RMSE", "MAE", "Mean bias")):
        values = [metrics[model]["common"][metric] for model in models]
        ax.bar(names, [np.nan if v is None else v for v in values], color=colors)
        ax.set(title=title, ylabel="Speed error (km/s)")
        ax.tick_params(axis="x", labelrotation=20)
        ax.axhline(0, color="#555555", lw=0.8)
        if not n:
            ax.text(0.5, 0.5, "No common valid pairs", ha="center", transform=ax.transAxes)
    fig.suptitle(f"Model comparison | {n:,} common forecast pairs\n"
                 "Bias = predicted − observed; overlapping forecasts retained")
    yield "model_comparison", fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for model, color in zip(models, colors):
        by_horizon = metrics[model]["common_by_horizon"]
        leads = sorted(map(int, by_horizon))
        axes[0].plot(leads, [by_horizon[str(h)]["rmse"] for h in leads],
                     color=color, label=label(model))
        bands = metrics[model]["common_by_band"]
        axes[1].plot(list(bands), [item["rmse"] for item in bands.values()],
                     marker="o", color=color, label=label(model))
    axes[0].set(xlabel="Forecast lead time (hours)", ylabel="RMSE (km/s)",
                title="Error by forecast horizon")
    axes[1].set(xlabel="Forecast lead-time band (hours)", ylabel="RMSE (km/s)",
                title="Error by horizon band")
    axes[0].legend()
    fig.suptitle("Common valid forecast pairs; missing scores are left blank")
    yield "error_by_horizon", fig

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(len(models))
    ax.bar(x - 0.2, [100 * metrics[m]["prediction_coverage"] for m in models],
           width=0.4, label="Finite forecast values", color=COLORS[0])
    ax.bar(x + 0.2, [100 * metrics[m]["complete_origin_fraction"] for m in models],
           width=0.4, label="Complete forecast origins", color=COLORS[2])
    ax.set(xticks=x, xticklabels=names, ylim=(0, 110), ylabel="Coverage (%)",
           title=f"Forecast coverage | {report['origins']:,} expected origins\n"
                 "Completeness of saved predictions, not platform uptime")
    ax.legend(loc="lower right")
    yield "forecast_coverage", fig

    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5),
                             squeeze=False, layout="constrained")
    values = data.loc[common, ["y", *models]].to_numpy()
    limits = (values.min(), values.max()) if values.size else (0, 1)
    margin = max((limits[1] - limits[0]) * 0.05, 1)
    limits = (limits[0] - margin, limits[1] + margin)
    for ax, model in zip(axes[0], models):
        if n:
            density = ax.hexbin(data.loc[common, "y"], data.loc[common, model],
                                gridsize=35, mincnt=1, bins="log", cmap="viridis",
                                extent=(*limits, *limits))
            fig.colorbar(density, ax=ax, label="Forecast pair count (log scale)")
        else:
            ax.text(0.5, 0.5, "No common valid pairs", ha="center", transform=ax.transAxes)
        ax.plot(limits, limits, color="#D55E00", linestyle="--", lw=1)
        ax.set(title=label(model), xlabel="Observed speed (km/s)",
               ylabel="Predicted speed (km/s)", xlim=limits, ylim=limits, aspect="equal")
    fig.suptitle(f"Observed vs. predicted | {n:,} common forecast pairs")
    yield "observed_vs_predicted", fig

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    errors = [data.loc[common, model] - data.loc[common, "y"] for model in models]
    if n:
        bins = np.histogram_bin_edges(np.concatenate(errors), bins=40)
        for model, error, color in zip(models, errors, colors):
            ax.hist(error, bins=bins, histtype="step", lw=1.8, color=color, label=label(model))
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No common valid pairs", ha="center", transform=ax.transAxes)
    ax.axvline(0, color="#555555", linestyle="--", lw=1)
    ax.set(title="Forecast error distribution | Common valid pairs",
           xlabel="Predicted − observed speed (km/s)", ylabel="Forecast pair count")
    yield "error_distribution", fig


def load_experiment(directory, observations=None):
    """Read native experiment artifacts, preserving missing observations."""
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    arguments = report.get("arguments", {})
    predictions = directory / "predictions.parquet"
    if not predictions.exists():
        predictions = directory / "predictions.csv"
    if predictions.exists():
        raw = (pd.read_parquet(predictions) if predictions.suffix == ".parquet"
               else pd.read_csv(predictions))
        models = list(report.get("metrics", {})) or arguments.get("models") or [
            name for name in raw if name not in {"origin_utc", "timestamp_utc", "horizon", "y"}]
        horizon = report.get("config", {}).get("horizon", int(raw.horizon.max()))
        origins = None
        if arguments.get("start") and arguments.get("end"):
            origins = pd.date_range(utc_hour(arguments["start"]), utc_hour(arguments["end"]),
                                    freq=f"{arguments.get('step_hours', 6)}h")
        data = validate_predictions(raw, models, horizon, origins)
        # Recompute to support older reports and keep every panel consistent.
        scores = evaluate_predictions(data, models, horizon=horizon,
                                      expected_origins=origins)
        return data, models, scores

    raw = pd.read_csv(directory / "forecast.csv")
    if not report.get("origin"):
        raise ValueError("A single forecast requires report.json with an origin")
    origin = utc_hour(report["origin"])
    model = report.get("model", "forecast")
    times = pd.to_datetime(raw.timestamp_utc, utc=True)
    data = pd.DataFrame({"origin_utc": origin, "timestamp_utc": times,
                         "horizon": (times - origin).dt.total_seconds() / 3600,
                         "y": np.nan, model: raw.speed_km_s})
    source = observations or report.get("data")
    if source and Path(source).is_file():
        observed = load_series(source, arguments.get("time_column", "timestamp_utc"),
                               arguments.get("target_column", "speed_km_s"))
        data["y"] = times.map(observed.set_index("ds").y)
    elif observations:
        raise FileNotFoundError(observations)
    else:
        print("Observations unavailable; plotting the forecast only.", file=sys.stderr)
    horizon = report.get("config", {}).get("horizon", int(data.horizon.max()))
    return validate_predictions(data, [model], horizon), [model], None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Directory containing forecast.csv or predictions.parquet/CSV")
    parser.add_argument("--output-dir", type=Path, help="Default: EXPERIMENT/figures")
    parser.add_argument("--observations", type=Path,
                        help="ACE CSV/Parquet for a single forecast; defaults to report data path")
    parser.add_argument("--origin", help="Origin for the trajectory plot; default: latest saved origin")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--training-history", type=Path, help="Optional recurrent training_history.csv")
    parser.add_argument("--training-label", default="LSTM", help="Learning-curve model label, e.g. GRU")
    args = parser.parse_args(argv)
    try:
        if args.dpi < 1:
            raise ValueError("DPI must be positive")
        data, models, report = load_experiment(args.experiment_dir, args.observations)
        origin = utc_hour(args.origin) if args.origin else data.origin_utc.max()
        if origin not in set(data.origin_utc):
            raise ValueError("The selected origin is absent from the forecast calendar")
        output = args.output_dir or args.experiment_dir / "figures"
        output.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.grid": True,
                             "grid.alpha": 0.18, "axes.axisbelow": True}):
            save_figure(plot_trajectory(data, models, origin), output, "forecast_trajectory",
                        args.formats, args.dpi)
            if report is not None:
                for name, fig in plot_backtest(data, models, report):
                    save_figure(fig, output, name, args.formats, args.dpi)
            if args.training_history:
                save_figure(plot_learning_curve(pd.read_csv(args.training_history), args.training_label), output,
                            "learning_curve", args.formats, args.dpi)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
