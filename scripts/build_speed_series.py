#!/usr/bin/env python3
"""Hourly ACE speed used as the target and in the figures: Level 2 where valid, operational elsewhere.

    .venv/bin/python scripts/build_speed_series.py
"""

import argparse
from pathlib import Path

import pandas as pd

from benchlab_forecasting.hybrid import load_plasma


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level2", type=Path, default=Path("data/ace_level2/ac_h2_swe_hourly.parquet"))
    parser.add_argument("--operational", type=Path, default=Path("data/ace_hourly_2011_2026/ace_swepam_hourly.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/features/ace_speed_combined_2011_2026.parquet"))
    args = parser.parse_args(argv)
    plasma = load_plasma(args.level2, args.operational)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_utc": plasma.index, "speed_km_s": plasma.speed_km_s.to_numpy()}).to_parquet(
        args.output, index=False)
    print(f"{args.output}: {len(plasma):,} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
