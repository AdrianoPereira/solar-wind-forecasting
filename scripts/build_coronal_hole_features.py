#!/usr/bin/env python3
"""Extract coronal-hole area fractions from every AIA image selected for a forecast origin.

Reads the per-origin selections in data/aia_synoptic/reports/ (the image each
origin could have used, respecting the assumed latency) and writes one row per
origin with the features of benchlab_forecasting.coronal_holes.

Example:
  .venv/bin/python scripts/build_coronal_hole_features.py --workers 32
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aia_fits import open_aia_fits  # noqa: E402
from benchlab_forecasting.coronal_holes import (  # noqa: E402
    LATITUDE_BANDS, LONGITUDE_EDGES, THRESHOLDS, aia_geometry, coronal_hole_features,
)


def selections(aia_dir, wavelength, cadence):
    """One selected image per origin; later reports win when several cover the same origin."""
    pattern = re.compile(rf"(\d{{4}}-\d\d-\d\d)_(\d{{4}}-\d\d-\d\d)_{wavelength:04d}_{cadence}h\.json$")
    chosen = {}
    reports = sorted((aia_dir / "reports").glob(f"*_{wavelength:04d}_{cadence}h.json"),
                     key=lambda p: p.stat().st_mtime)
    for path in reports:
        if not pattern.search(path.name):
            continue
        for item in json.loads(path.read_text())["images"]:
            if item["status"] == "ok":
                chosen[item["origin_utc"]] = {"file": item["file"], "observation_utc": item["observation_utc"],
                                              "age_minutes": item["age_at_origin_minutes"], "report": path.name}
            elif item["origin_utc"] not in chosen:
                chosen[item["origin_utc"]] = None
    return chosen


def extract(task):
    origin, selection, aia_dir = task
    row = {"origin_utc": origin}
    if selection is None:
        return {**row, "error": "No acceptable image selected for this origin"}
    try:
        with open_aia_fits(aia_dir / selection["file"]) as (hdus, _):
            hdu = next(h for h in hdus if h.data is not None)
            image, header = hdu.data.astype(float), hdu.header
            center_x, center_y, radius_px, b0 = aia_geometry(header)
            exposure = float(header["EXPTIME"])
        features = coronal_hole_features(image, center_x, center_y, radius_px, b0)
        return {**row, **selection, "b0_deg": b0, "radius_px": radius_px, "exposure_s": exposure,
                **features, "error": None}
    except Exception as exc:
        return {**row, **selection, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aia-dir", type=Path, default=Path("data/aia_synoptic"))
    parser.add_argument("--wavelength", type=int, default=193)
    parser.add_argument("--cadence-hours", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("data/features/coronal_holes_0193_6h.parquet"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    chosen = selections(args.aia_dir, args.wavelength, args.cadence_hours)
    if not chosen:
        raise SystemExit("Error: no AIA reports found")
    tasks = [(origin, chosen[origin], args.aia_dir) for origin in sorted(chosen)]
    print(f"{len(tasks)} origins; {sum(s is not None for s in chosen.values())} with an image", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(extract, tasks, chunksize=16), 1):
            rows.append(row)
            if i % 1000 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows)
    frame["origin_utc"] = pd.to_datetime(frame.origin_utc, utc=True)
    frame["observation_utc"] = pd.to_datetime(frame.observation_utc, utc=True, format="ISO8601")
    frame = frame.sort_values("origin_utc", ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    errors = frame.error.dropna()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "output": str(args.output),
        "wavelength_angstrom": args.wavelength, "cadence_hours": args.cadence_hours,
        "origins": len(frame), "with_features": int(frame.error.isna().sum()),
        "first_origin": frame.origin_utc.min().isoformat(), "last_origin": frame.origin_utc.max().isoformat(),
        "thresholds_of_disk_median": THRESHOLDS, "latitude_bands_deg": LATITUDE_BANDS,
        "longitude_edges_deg": LONGITUDE_EDGES,
        "errors": errors.str.slice(0, 80).value_counts().head(20).to_dict(),
        "note": ("Image selection inherits the downloader's assumed 60-minute latency. Features are "
                 "area fractions weighted by 1/mu; filaments can be counted as coronal holes."),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("origins", "with_features", "first_origin", "last_origin", "errors")},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
