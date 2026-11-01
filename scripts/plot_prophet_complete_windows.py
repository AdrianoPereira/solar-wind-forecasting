#!/usr/bin/env python3
"""Plot the saved, common-origin Prophet backtest for 3, 7 and 14 days."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DURATIONS = (3, 7, 14)
COLORS = {3: "#138a72", 7: "#2474b5", 14: "#7951a8"}
PERSISTENCE_COLOR = "#747474"
OBSERVED_COLOR = "#202020"
SHARED_COLUMNS = ("origin_utc", "timestamp_utc", "horizon", "observed_km_s", "persistence_km_s")


def rmse(observed, predicted):
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def load_results(output_dir):
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    frames = {}
    reference = None
    for days in DURATIONS:
        folder = output_dir / f"{days}d_72h"
        frame = pd.read_parquet(folder / "test.parquet").sort_values(
            ["origin_utc", "horizon"]
        ).reset_index(drop=True)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["duration_days"] != days or manifest["horizon_hours"] != 72:
            raise ValueError(f"Manifesto incompatível para {days} dias")
        if len(frame) != 72 * len(summary["test_origins"]):
            raise ValueError(f"Número de previsões de teste inesperado para {days} dias")
        origins = [timestamp.isoformat() for timestamp in frame.origin_utc.drop_duplicates()]
        if origins != summary["test_origins"] or not np.array_equal(
            frame.horizon.to_numpy(), np.tile(np.arange(1, 73), len(origins))
        ):
            raise ValueError(f"Origens ou horizontes incompatíveis para {days} dias")
        if reference is None:
            reference = frame
        else:
            for column in SHARED_COLUMNS:
                pd.testing.assert_series_equal(frame[column], reference[column], check_names=False)
        for method, column in (("prophet", "prophet_km_s"), ("persistence", "persistence_km_s")):
            measured = rmse(frame.observed_km_s.to_numpy(), frame[column].to_numpy())
            recorded = manifest["test"][method]["rmse_km_s"]
            if not np.isclose(measured, recorded, rtol=0, atol=1e-6):
                raise ValueError(f"RMSE em test.parquet difere do manifesto para {days} dias: {method}")
        frames[days] = frame
    return summary, frames


def save_rmse_comparison(frames, destination):
    labels = ["Persistência", "Prophet\n3 dias", "Prophet\n7 dias", "Prophet\n14 dias"]
    reference = frames[3]
    values = [rmse(reference.observed_km_s, reference.persistence_km_s)]
    values += [rmse(frames[days].observed_km_s, frames[days].prophet_km_s) for days in DURATIONS]
    colors = [PERSISTENCE_COLOR] + [COLORS[days] for days in DURATIONS]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=10)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("RMSE (km/s)")
    ax.set_title("Erro nas 72 horas: teste comum às três janelas")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.015, "12 origens de teste (2009–2010); 864 pares por estratégia", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rmse_by_horizon(frames, destination):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    reference = frames[3]
    persistence_error = (reference.persistence_km_s - reference.observed_km_s).pow(2)
    persistence_rmse = np.sqrt(persistence_error.groupby(reference.horizon).mean())
    ax.plot(persistence_rmse.index, persistence_rmse, label="Persistência",
            color=PERSISTENCE_COLOR, linewidth=2.1, linestyle="--")
    for days in DURATIONS:
        frame = frames[days]
        squared_error = (frame.prophet_km_s - frame.observed_km_s).pow(2)
        by_horizon = np.sqrt(squared_error.groupby(frame.horizon).mean())
        ax.plot(by_horizon.index, by_horizon, label=f"Prophet {days} dias",
                color=COLORS[days], linewidth=2)
    ax.set(xlim=(1, 72), xlabel="Horizonte de previsão (horas)", ylabel="RMSE (km/s)",
           title="Erro por horizonte nas 12 origens de teste")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_forecast_example(frames, origin, destination):
    selected = {days: frame.loc[frame.origin_utc == origin] for days, frame in frames.items()}
    if any(len(frame) != 72 for frame in selected.values()):
        raise ValueError(f"Origem de exemplo ausente ou incompleta: {origin}")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, days in zip(axes, DURATIONS):
        frame = selected[days]
        x = frame.horizon.to_numpy()
        ax.plot(x, frame.observed_km_s, label="Observado", color=OBSERVED_COLOR, linewidth=2)
        ax.plot(x, frame.persistence_km_s, label="Persistência", color=PERSISTENCE_COLOR,
                linewidth=1.7, linestyle="--")
        ax.plot(x, frame.prophet_km_s, label=f"Prophet {days} dias",
                color=COLORS[days], linewidth=2)
        ax.set_ylabel("Velocidade (km/s)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)
    axes[-1].set(xlim=(1, 72), xlabel="Horizonte de previsão (horas)")
    fig.suptitle(f"Previsões de 72 horas na origem {origin.strftime('%d/%m/%Y %H:%M UTC')}")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/models/prophet_complete_windows")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    summary, frames = load_results(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "comparacao_rmse_teste": "comparacao_rmse_teste.png",
        "rmse_por_horizonte": "rmse_por_horizonte.png",
        "exemplo_previsoes_72h": "exemplo_previsoes_72h.png",
    }
    example_origin = pd.Timestamp(summary["test_origins"][-1])
    save_rmse_comparison(frames, figures_dir / figures["comparacao_rmse_teste"])
    save_rmse_by_horizon(frames, figures_dir / figures["rmse_por_horizonte"])
    save_forecast_example(frames, example_origin, figures_dir / figures["exemplo_previsoes_72h"])
    (figures_dir / "manifest.json").write_text(json.dumps({
        "source": "test.parquet for 3d_72h, 7d_72h and 14d_72h",
        "test_origins": len(summary["test_origins"]),
        "horizon_hours": 72,
        "example_origin_utc": example_origin.isoformat(),
        "figures": figures,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in figures.values():
        print(figures_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
