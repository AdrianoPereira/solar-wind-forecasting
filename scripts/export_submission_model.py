#!/usr/bin/env python3
"""Export the tuned hybrid for the submission container and fit the persistence blend.

For each saved model (with and without coronal holes) this writes the LightGBM booster as text,
the feature order and the number of trees. It also chooses, for every lead hour, the weight w of
persistence in w * persistence + (1 - w) * hybrid that minimizes the squared error on the
validation years (2022–2023). The test period is only scored afterwards, for the report.

    OMP_NUM_THREADS=8 .venv/bin/python scripts/export_submission_model.py
"""

import argparse
import json
import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from benchlab_forecasting.hybrid import HybridConfig, build_pairs, load_plasma, origins_between

SPLITS = {"validation": ("2022-01-01", "2023-12-28"), "test": ("2024-01-01", "2026-10-23T00:00")}
WEIGHTS = np.linspace(0, 1, 101)


def blend_weights(frame, prediction):
    """Per-lead weight of persistence, chosen on observed pairs; monotone non-increasing in lead."""
    observed = frame.y.notna().to_numpy()
    weights = []
    for lead in range(1, 73):
        mask = observed & (frame.horizon.to_numpy() == lead)
        y, model, persistence = frame.y.to_numpy()[mask], prediction[mask], frame.last_speed.to_numpy()[mask]
        errors = [np.mean((w * persistence + (1 - w) * model - y) ** 2) for w in WEIGHTS]
        weights.append(float(WEIGHTS[int(np.argmin(errors))]))
    # Persistence only loses value with lead time; keep the curve from rising again by noise.
    return np.minimum.accumulate(weights).tolist()


def blend(frame, prediction, weights):
    w = np.asarray(weights)[frame.horizon.to_numpy() - 1]
    persistence = frame.last_speed.to_numpy()
    return np.where(np.isfinite(persistence), w * persistence + (1 - w) * prediction, prediction)


def rmse(frame, values, lo=1, hi=72):
    mask = frame.y.notna().to_numpy() & frame.horizon.between(lo, hi).to_numpy()
    return float(np.sqrt(np.mean((values[mask] - frame.y.to_numpy()[mask]) ** 2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=Path("data/experiments/hybrid_lightgbm_tuning"))
    parser.add_argument("--level2", type=Path, default=Path("data/ace_level2/ac_h2_swe_hourly.parquet"))
    parser.add_argument("--operational", type=Path, default=Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet"))
    parser.add_argument("--coronal-holes", type=Path, default=Path("data/features/coronal_holes_0193_6h.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission/model"))
    args = parser.parse_args(argv)

    plasma = load_plasma(args.level2, args.operational)
    holes = pd.read_parquet(args.coronal_holes)
    holes = holes[holes.error.isna()].copy()
    holes["origin_utc"] = pd.to_datetime(holes.origin_utc, utc=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ("tuned", "tuned_no_ch"):
        saved = pickle.loads((args.experiment / f"model_{name}.pkl").read_bytes())
        known = {f.name for f in fields(HybridConfig)}
        config = HybridConfig(**{k: v for k, v in saved["config"].items() if k in known})
        booster = saved["model"].booster_
        tables = {k: build_pairs(plasma, holes, origins_between(*v), config) for k, v in SPLITS.items()}
        predict = {k: booster.predict(t[saved["features"]], num_iteration=saved["iterations"])
                   for k, t in tables.items()}
        weights = blend_weights(tables["validation"], predict["validation"])
        booster.save_model(str(args.output_dir / f"{name}.txt"), num_iteration=saved["iterations"])
        meta = {"features": saved["features"], "iterations": saved["iterations"],
                "config": saved["config"], "persistence_weight_by_lead": weights,
                "blend": "w(lead) * persistence + (1 - w(lead)) * hybrid, w chosen on 2022–2023"}
        (args.output_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
        test, values = tables["test"], predict["test"]
        mixed = blend(test, values, weights)
        report[name] = {"weights_first_12h": [round(w, 2) for w in weights[:12]],
                        "last_lead_with_weight": int(np.flatnonzero(np.asarray(weights) > 0).max() + 1)
                        if any(weights) else 0,
                        "test_rmse_hybrid": rmse(test, values), "test_rmse_blend": rmse(test, mixed),
                        "test_rmse_1_12_hybrid": rmse(test, values, 1, 12),
                        "test_rmse_1_12_blend": rmse(test, mixed, 1, 12),
                        "test_rmse_persistence": rmse(test, test.last_speed.to_numpy())}
        # Exported booster must reproduce the saved model exactly.
        import lightgbm
        reloaded = lightgbm.Booster(model_file=str(args.output_dir / f"{name}.txt"))
        assert np.allclose(reloaded.predict(test[saved["features"]]), values)
    (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
