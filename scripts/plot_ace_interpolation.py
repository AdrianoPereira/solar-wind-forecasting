#!/usr/bin/env python3
"""Create reproducible figures comparing original and interpolated ACE data."""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "benchlab_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FIELDS = {
    "swepam": (
        ("speed_km_s", "Velocidade", "km/s"),
        ("density_cm3", "Densidade", "cm⁻³"),
        ("temperature_k", "Temperatura", "K"),
    ),
    "mag": (
        ("bx_gsm_nt", "Bx GSM", "nT"),
        ("by_gsm_nt", "By GSM", "nT"),
        ("bz_gsm_nt", "Bz GSM", "nT"),
        ("bt_nt", "|B|", "nT"),
        ("latitude_gsm_deg", "Latitude GSM", "graus"),
        ("longitude_gsm_deg", "Longitude GSM", "graus"),
    ),
}
COLOR_ORIGINAL = "#293241"
COLOR_INTERPOLATED = "#dd6b20"
COLOR_RECOVERED = "#168a62"
LEGEND = [
    Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.4, marker=".", label="Original válido"),
    Line2D([0], [0], color=COLOR_INTERPOLATED, lw=1.8, label="Série após interpolação"),
    Line2D([0], [0], color=COLOR_RECOVERED, lw=0, marker="D", markersize=5,
           label="Recuperado da fonte"),
]


def longest_run(flags):
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        raise ValueError("No interpolated gap found")
    position = int(np.argmax(stops - starts))
    return int(starts[position]), int(stops[position] - 1)


def prepared_lines(window, field):
    values = window[field].to_numpy(dtype=float)
    interpolated = window[f"{field}_is_interpolated"].to_numpy(dtype=bool)
    recovered = window[f"{field}_is_recovered"].to_numpy(dtype=bool)
    original = values.copy()
    original[interpolated | recovered] = np.nan
    final = values.copy()
    if field == "longitude_gsm_deg":
        # A wrapped circular series has a display discontinuity at 0°/360°.
        jumps = np.flatnonzero(np.abs(np.diff(values)) > 180) + 1
        final[jumps] = np.nan
        original[jumps] = np.nan
    return original, final, recovered


def plot_field(ax, window, field, label, unit, highlighted=None):
    x = window["timestamp_utc"]
    original, final, recovered = prepared_lines(window, field)
    if highlighted is not None:
        left, right = highlighted
        ax.axvspan(left, right + pd.Timedelta(hours=1), color=COLOR_INTERPOLATED,
                   alpha=0.08, lw=0)
    ax.plot(x, final, color=COLOR_INTERPOLATED, lw=1.5, alpha=0.95, zorder=2)
    ax.plot(x, original, color=COLOR_ORIGINAL, lw=1.05, marker=".", markersize=2.3,
            alpha=0.92, zorder=3)
    if recovered.any():
        ax.scatter(x[recovered], window.loc[recovered, field], color=COLOR_RECOVERED,
                   s=19, marker="D", edgecolors="white", linewidths=0.4, zorder=4)
    ax.set_ylabel(f"{label}\n({unit})")
    ax.grid(axis="y", color="#d9e0e7", lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def detail_figure(frame, fields, anchor_field, title, output, pad_hours=72):
    first, last = longest_run(frame[f"{anchor_field}_is_interpolated"].to_numpy(dtype=bool))
    window = frame.iloc[max(0, first - pad_hours):min(len(frame), last + pad_hours + 1)].copy()
    left, right = frame.timestamp_utc.iloc[first], frame.timestamp_utc.iloc[last]
    ncols = 1 if len(fields) == 3 else 2
    nrows = (len(fields) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.3, 3.1 * nrows + 1),
                             sharex=True, constrained_layout=False)
    axes = np.array(axes).reshape(-1)
    for ax, (field, label, unit) in zip(axes, fields):
        plot_field(ax, window, field, label, unit, (left, right))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    gap_hours = last - first + 1
    fig.suptitle(f"{title} · lacuna de {gap_hours} horas", fontsize=16, weight="bold", y=0.995)
    show_recovered = any(window[f"{field}_is_recovered"].any() for field, _, _ in fields)
    handles = LEGEND if show_recovered else LEGEND[:2]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(handles), frameon=False)
    fig.supxlabel("Data e hora (UTC)", y=0.012)
    fig.tight_layout(rect=(0, 0.025, 1, 0.92), h_pad=1.4, w_pad=1.4)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return {"file": str(output), "gap_first_hour_utc": left.isoformat(),
            "gap_last_hour_utc": right.isoformat(), "gap_hours": gap_hours}


def recovered_figure(frame, output):
    indices = np.flatnonzero(frame["speed_km_s_is_recovered"].to_numpy(dtype=bool))
    if not len(indices):
        return None
    window = frame.iloc[max(0, int(indices[0]) - 24):min(len(frame), int(indices[-1]) + 25)]
    fig, axes = plt.subplots(3, 1, figsize=(12.7, 8.9), sharex=True)
    for ax, (field, label, unit) in zip(axes, FIELDS["swepam"]):
        plot_field(ax, window, field, label, unit)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("ACE/SWEPAM · dados originais, recuperados e interpolados",
                 fontsize=16, weight="bold", y=0.995)
    fig.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.supxlabel("Data e hora (UTC)", y=0.012)
    fig.tight_layout(rect=(0, 0.025, 1, 0.92), h_pad=1.4)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return {"file": str(output), "first_recovered_hour_utc": frame.timestamp_utc.iloc[indices[0]].isoformat(),
            "last_recovered_hour_utc": frame.timestamp_utc.iloc[indices[-1]].isoformat(),
            "recovered_speed_hours": int(len(indices))}


def coverage_figure(frame, source_ranges, output):
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.8), sharex=True)
    specs = (("swepam", "speed_km_s", "ACE/SWEPAM · velocidade"),
             ("mag", "bt_nt", "ACE/MAG · intensidade do campo"))
    for ax, (source, field, title) in zip(axes, specs):
        source_name = "ACE/SWEPAM" if source == "swepam" else "ACE/MAG"
        end = pd.Timestamp(source_ranges[source_name]["last_hour_utc"])
        subset = frame.loc[frame.timestamp_utc <= end]
        observed = (subset[field].notna()
                    & ~subset[f"{field}_is_interpolated"]
                    & ~subset[f"{field}_is_recovered"])
        available = subset[field].notna()
        groups = pd.DataFrame({"year": subset.timestamp_utc.dt.year,
                               "observed": observed.astype(int),
                               "available": available.astype(int)}).groupby("year")
        rates = groups[["observed", "available"]].mean() * 100
        years = rates.index.to_numpy()
        ax.plot(years, rates.observed, color=COLOR_ORIGINAL, marker="o", markersize=3.5,
                lw=1.6, label="Original válido")
        ax.plot(years, rates.available, color=COLOR_INTERPOLATED, marker="o", markersize=3.5,
                lw=1.6, label="Após interpolação")
        ax.fill_between(years, rates.observed, rates.available, color=COLOR_INTERPOLATED,
                        alpha=0.12)
        ax.set_ylim(-3, 103)
        ax.set_ylabel("Horas com valor (%)")
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.grid(axis="y", color="#d9e0e7", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Ano (UTC)")
    axes[-1].set_xticks(np.arange(2000, 2027, 2))
    fig.suptitle("Cobertura anual: dados ACE originais e série interpolada",
                 fontsize=16, weight="bold", y=0.99)
    fig.text(0.5, 0.015, "SWEPAM em 2026: janeiro–setembro; MAG termina em 2025. "
             "Lacunas sem âncora nas extremidades permanecem vazias.",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=1.3)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return {"file": str(output)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/quality/ace_interpolated_all_internal.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/quality/ace_interpolated_all_internal_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality/figures"))
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else root / path
    dataset, report_path, output_dir = map(resolve, (args.dataset, args.report, args.output_dir))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(dataset)
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "coverage": coverage_figure(frame, report["source_ranges"], output_dir / "ace_cobertura_anual.png"),
        "swepam_long_gap": detail_figure(
            frame, FIELDS["swepam"], "speed_km_s", "ACE/SWEPAM · maior lacuna da velocidade",
            output_dir / "ace_swepam_lacuna_longa.png"),
        "mag_long_gap": detail_figure(
            frame, FIELDS["mag"], "bt_nt", "ACE/MAG · maior lacuna de |B|",
            output_dir / "ace_mag_lacuna_longa.png"),
        "swepam_recovered": recovered_figure(frame, output_dir / "ace_swepam_recuperados.png"),
    }
    manifest = {"dataset": str(dataset), "figures": figures,
                "note": "Interpolated values are synthetic and use future endpoints."}
    (output_dir / "ace_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
