"""English TIMSE figures and an auditable report from saved forecasts only."""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import CHALLENGE, regression_metrics, timse_profile
from .reporting import validate_predictions

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#333333"]
MODEL_COLORS = dict(zip(("gru", "lstm", "persistence", "recurrence", "prophet_current", "prophet_tuned"), COLORS))
LABELS = {"gru": "GRU", "lstm": "LSTM", "recurrence": "27-day recurrence",
          "prophet_current": "Prophet (current)", "prophet_tuned": "Prophet (tuned)"}
UNIT = "TIMSE [(km/s)³]"


def label(name):
    return LABELS.get(name, name.replace("_", " ").title())


def cohort_tables(data, models):
    """All comparisons within a cohort use exactly the same target/forecast pairs."""
    common = data.y.notna() & data[models].notna().all(axis=1)
    block = data.loc[common]
    summary, profiles, groups = [], [], []
    for model in models:
        profile = timse_profile(block.y, block[model])
        strict = timse_profile(data.y, data[model])
        valid_targets = int(data.y.notna().sum())
        summary.append({"model": model, "n": len(block),
                        "timse": None if profile is None else profile["timse"],
                        "degenerate": None if profile is None else profile["degenerate"],
                        "strict_timse": None if strict is None else strict["timse"],
                        "rmse": regression_metrics(block.y, block[model])["rmse"],
                        "prediction_coverage": float(data[model].notna().mean()),
                        "target_pair_coverage": (float((data.y.notna() & data[model].notna()).sum() / valid_targets)
                                                 if valid_targets else None),
                        "complete_origins": int(data[model].notna().groupby(data.origin_utc).all().sum())})
        if profile is not None:
            profile_frame = pd.DataFrame({k: profile[k] for k in
                ("threshold", "tail_n", "tail_mse", "width", "contribution", "cumulative_timse")})
            profile_frame.insert(0, "model", model)
            profiles.append(profile_frame)
        for kind, grouped in (("horizon", data.groupby("horizon")),
                              ("origin", data.groupby("origin_utc")),
                              ("band", data.groupby(pd.cut(data.horizon, [0, 24, 48, 72],
                                     labels=["1-24", "25-48", "49-72"]), observed=True))):
            for key, group in grouped:
                group = group.loc[common.loc[group.index]]
                detail = timse_profile(group.y, group[model])
                groups.append({"model": model, "kind": kind, "group": str(key), "n": len(group),
                               "timse": None if detail is None else detail["timse"],
                               "degenerate": None if detail is None else detail["degenerate"],
                               "speed_min": None if group.empty else float(group.y.min()),
                               "speed_max": None if group.empty else float(group.y.max())})
    columns = ["model", "threshold", "tail_n", "tail_mse", "width", "contribution", "cumulative_timse"]
    return (pd.DataFrame(summary), pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=columns),
            pd.DataFrame(groups), block)


def figures(summary, profiles, groups, models, cohort):
    colors = {m: MODEL_COLORS.get(m, COLORS[i % len(COLORS)]) for i, m in enumerate(models)}
    n = int(summary.n.iloc[0])
    title = f"{cohort} | {n:,} common forecast pairs"
    ordered = summary.sort_values("timse", na_position="last")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = [label(m) for m in ordered.model]
    axes[0].barh(names, pd.to_numeric(ordered.timse, errors="coerce"), color=[colors[m] for m in ordered.model])
    axes[0].invert_yaxis()
    axes[0].set(title="Published TIMSE formula · Lower is better", xlabel=UNIT)
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for i, value in enumerate(ordered.timse):
        if pd.notna(value):
            axes[0].text(value, i, f" {value:,.0f}", va="center", fontsize=9)
    axes[0].margins(x=0.25)
    axes[1].barh(names, 100 * ordered.prediction_coverage, color=[colors[m] for m in ordered.model])
    axes[1].invert_yaxis()
    axes[1].set(title="Coverage · Complete expected forecast grid", xlabel="Finite forecasts (%)", xlim=(0, 112))
    for i, value in enumerate(ordered.prediction_coverage):
        axes[1].text(100 * value, i, f" {100 * value:.1f}%", va="center")
    fig.suptitle(title)
    if not n:
        axes[0].text(0.5, 0.5, "No common valid pairs", transform=axes[0].transAxes, ha="center")
    elif summary.degenerate.fillna(False).all():
        axes[0].set_title("Degenerate target: zero integration width")
    yield "ranking_coverage", fig
    if profiles.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 1]})
    for model in models:
        p = profiles.loc[profiles.model.eq(model)]
        axes[0].step(p.threshold, p.tail_mse, where="post", color=colors[model], label=label(model))
    p = profiles.loc[profiles.model.eq(models[0])]
    axes[1].step(p.threshold, p.tail_n, where="post", color="#333333")
    axes[0].set(title="Tail MSE at each unique observed speed", ylabel="Tail MSE [(km/s)²]")
    axes[0].legend(ncols=3)
    axes[1].set(xlabel="Observed speed threshold (km/s)", ylabel="Tail pairs", yscale="log")
    fig.suptitle(title + "\nLeft rectangles define the integral; the final threshold has zero width")
    yield "tail_mse", fig

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for model in models:
        p = profiles.loc[profiles.model.eq(model)]
        axes[0].plot(np.r_[p.threshold.iloc[0], p.threshold + p.width],
                     np.r_[0.0, p.cumulative_timse], color=colors[model], label=label(model))
    # Pool exact interval contributions into 12 display bins; no change to the score.
    edges = np.linspace(float(p.threshold.min()), float(p.threshold.max()), 13)
    if edges[-1] > edges[0]:
        for model in models:
            p = profiles.loc[profiles.model.eq(model)]
            totals, _ = np.histogram(p.threshold, bins=edges, weights=p.contribution)
            axes[1].stairs(totals, edges, color=colors[model], label=label(model))
    axes[0].set(title="Cumulative threshold integral", ylabel=UNIT)
    axes[0].legend(ncols=3)
    axes[1].set(title="Interval contributions grouped by left threshold (display only)",
                xlabel="Observed speed threshold (km/s)", ylabel=UNIT)
    fig.suptitle(title)
    yield "threshold_contributions", fig

    for kind, name, xlabel in (("horizon", "by_horizon", "Forecast lead (hours)"),
                               ("origin", "by_origin", "Forecast origin (UTC)")):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [3, 1]})
        for model in models:
            g = groups.loc[groups.kind.eq(kind) & groups.model.eq(model)].copy()
            g["x"] = pd.to_numeric(g.group) if kind == "horizon" else pd.to_datetime(g.group, utc=True)
            g = g.sort_values("x")
            axes[0].plot(g.x, g.timse, color=colors[model], label=label(model))
        axes[1].plot(g.x, g.n, color="#333333")
        axes[0].set(ylabel=UNIT)
        axes[0].legend(ncols=3)
        axes[1].set(xlabel=xlabel, ylabel="Common pairs", ylim=(0, None))
        fig.suptitle(title + "\nThresholds recomputed within each group; group scores do not sum to pooled TIMSE")
        yield name, fig


def generate_report(frame, models, output, *, horizon=72, expected_origins=None,
                    formats=("png", "pdf"), dpi=160, provenance=None):
    """Write tables, figures and a self-contained HTML report to a fresh directory."""
    models = list(models)
    if not formats or set(formats) - {"png", "pdf", "svg"} or dpi < 1:
        raise ValueError("Choose png/pdf/svg formats and a positive DPI")
    formats = tuple(dict.fromkeys(("png", *formats)))  # HTML always embeds PNG.
    data = validate_predictions(frame, models, horizon, expected_origins)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cohorts = {"all_models": models}
    complete = [m for m in models if data[m].notna().all()]
    if complete and complete != models:
        cohorts["complete_models"] = complete
    metadata = {"challenge": dict(CHALLENGE), "units": "(km/s)^3",
                "formula": "sum_j (v[j+1]-v[j]) * mean((y-prediction)^2 where y>=v[j])",
                "thresholds": "sorted unique observed speeds; left rectangles; no normalization",
                "aggregation": "origin/lead pairs; overlapping targets retained; local convention",
                "missing_policy": "Exclude missing targets; strict score null for missing predictions; common-pair comparison per cohort",
                "group_policy": "Recompute thresholds per group; group scores are not additive",
                "calendar": "explicit" if expected_origins is not None else "inferred from present origins; entirely missing origins cannot be detected",
                "origin_start": str(data.origin_utc.min()), "origin_end": str(data.origin_utc.max()),
                "horizon": horizon, "origins": int(data.origin_utc.nunique()), "forecast_pairs": len(data),
                "valid_targets": int(data.y.notna().sum()), "provenance": provenance or {}, "cohorts": {},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("metrics.py", "reporting.py", "timse_report.py")}}
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>TIMSE validation</title>",
               "<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#182536}"
               "img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px}"
               "td,th{padding:7px;border-bottom:1px solid #ccd}code{overflow-wrap:anywhere}</style>",
               "<h1>TIMSE validation</h1><p>Published participant formula: "
               "Σ<sub>j=1…m−1</sub> (v*<sub>j+1</sub> − v*<sub>j</sub>) M<sub>j</sub>, "
               "where M<sub>j</sub> is mean squared error for observations ≥ v*<sub>j</sub>. "
               "Thresholds are sorted unique observations. Units: (km/s)³. Lower is better.</p>",
               f"<p>Origins: {metadata['origin_start']} to {metadata['origin_end']}; "
               f"{metadata['origins']} origins, leads 1–{horizon} h; {len(data):,} expected pairs, "
               f"{metadata['valid_targets']:,} observed targets. Calendar: {html.escape(metadata['calendar'])}.</p>",
               "<p>This is a local evaluation of the published formula. Platform reference scorer and "
               "missing-prediction policy remain unverified. No lead weights, normalization or coverage "
               "penalty are added. Missing targets are excluded. Strict scores are undefined if a forecast "
               "is missing at an observed target. Every ranking uses identical valid pairs within its cohort. "
               "In the tables, timse and rmse use the common cohort; strict_timse uses all observed targets. "
               "prediction_coverage uses the complete forecast grid; target_pair_coverage uses observed targets only.</p>",
               "<p>Overlapping forecasts are retained; pairs are not independent. This retrospective "
               "diagnostic does not retrain or select models, establish statistical significance, or create a new holdout. "
               "A constant observed target yields zero integration width and a degenerate score of zero.</p>"]
    for name, members in cohorts.items():
        summary, profiles, groups, block = cohort_tables(data, members)
        summary.to_csv(output / f"{name}_scores.csv", index=False)
        profiles.to_csv(output / f"{name}_thresholds.csv", index=False)
        groups.to_csv(output / f"{name}_groups.csv", index=False)
        block.to_parquet(output / f"{name}_common_pairs.parquet", index=False)
        metadata["cohorts"][name] = {"models": members, "common_pairs": len(block),
                                   "unique_target_times": int(block.timestamp_utc.nunique()),
                                   "speed_min": None if block.empty else float(block.y.min()),
                                   "speed_max": None if block.empty else float(block.y.max()),
                                   "scores": json.loads(summary.to_json(orient="records", double_precision=15))}
        title = "All models" if name == "all_models" else "Models with complete forecast coverage"
        content.append(f"<h2>{title}</h2><p>{len(block):,} common pairs. "
                       "Scores across different cohorts use different target distributions and must not be ranked together.</p>")
        content.append(summary.sort_values("timse").to_html(index=False, border=0, na_rep="Undefined", float_format=lambda x: f"{x:,.3f}"))
        for stem, fig in figures(summary, profiles, groups, members, title):
            for extension in formats:
                fig.savefig(output / f"{name}_{stem}.{extension}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            encoded = base64.b64encode((output / f"{name}_{stem}.png").read_bytes()).decode("ascii")
            content.append(f"<img alt='{title}: {stem.replace('_', ' ')}' src='data:image/png;base64,{encoded}'>")
    content.append("<p>Formula: participant slide supplied by the user (page 29). "
                   f"<a href='{CHALLENGE['source']}'>Participant FAQ</a> checked 2026-10-05 "
                   "clarifies unique observed thresholds and no additional lead-time weighting.</p></html>")
    (output / "report.html").write_text("\n".join(content), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    try:
        if bool(args.start) != bool(args.end) or args.step_hours < 1:
            raise ValueError("Supply --start and --end together, with a positive step")
        expected = (pd.date_range(args.start, args.end, freq=f"{args.step_hours}h", tz="UTC") if args.start else None)
        frame = pd.read_parquet(args.predictions) if args.predictions.suffix == ".parquet" else pd.read_csv(args.predictions)
        with args.predictions.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = generate_report(frame, args.models, args.output_dir, horizon=args.horizon,
                                 expected_origins=expected, formats=args.formats, dpi=args.dpi,
                                 provenance={"predictions": str(args.predictions), "sha256": digest})
        print(f"Report: {args.output_dir / 'report.html'}")
        for name, cohort in report["cohorts"].items():
            print(f"{name}: {cohort['common_pairs']} common pairs")
        return 0 if report["cohorts"]["all_models"]["common_pairs"] else 1
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
