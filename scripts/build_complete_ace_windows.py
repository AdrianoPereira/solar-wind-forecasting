#!/usr/bin/env python3
"""Export complete ACE input windows (3/7/14 days) with 72 observed future labels.

Inputs contain the nine ACE/SWEPAM and ACE/MAG variables. Every input cell and
every future speed label must be a valid observed measurement, either in the
original cache or recovered from the same operational archive. Interpolated
cells are excluded. Forecast origins occur every six UTC hours.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


FEATURES = (
    "speed_km_s", "density_cm3", "temperature_k",
    "bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt", "bt_nt",
    "latitude_gsm_deg", "longitude_gsm_deg",
)
FEATURE_UNITS = ("km/s", "cm^-3", "K", "nT", "nT", "nT", "nT", "degrees", "degrees")
DAYS = (3, 7, 14)
HORIZON = 72
STEP_HOURS = 6
LATENCY_HOURS = 1


def read_source(path):
    required = ["timestamp_utc", *FEATURES]
    required += [f"{field}_is_interpolated" for field in FEATURES]
    required += [f"{field}_is_recovered" for field in FEATURES]
    frame = pd.read_parquet(path, columns=required)
    if frame.empty:
        raise ValueError("Source dataset is empty")
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    stamps = frame.timestamp_utc
    if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and ordered")
    if not stamps.eq(pd.date_range(start=stamps.iloc[0], periods=len(frame), freq="h")).all():
        raise ValueError("Source must have a continuous hourly UTC grid")
    if stamps.iloc[0].hour % STEP_HOURS or stamps.iloc[0].minute:
        raise ValueError("Source grid must begin on a six-hour UTC boundary")
    return frame


def valid_masks(frame):
    input_ok = np.ones(len(frame), dtype=np.bool_)
    for field in FEATURES:
        input_ok &= (np.isfinite(frame[field].to_numpy(dtype=np.float64))
                     & ~frame[f"{field}_is_interpolated"].to_numpy(dtype=np.bool_))
    target_ok = (np.isfinite(frame["speed_km_s"].to_numpy(dtype=np.float64))
                 & ~frame["speed_km_s_is_interpolated"].to_numpy(dtype=np.bool_))
    return input_ok, target_ok


def select_origins(input_ok, target_ok, lookback_hours, horizon=HORIZON,
                   step_hours=STEP_HOURS, latency_hours=LATENCY_HOURS):
    """Return t0 indices with complete input and observed future speed."""
    if len(input_ok) != len(target_ok) or min(lookback_hours, horizon, step_hours, latency_hours) < 1:
        raise ValueError("Invalid window or mask lengths")
    length = len(input_ok)
    origins = np.arange(0, length - horizon, step_hours, dtype=np.int64)
    input_stop = origins - latency_hours + 1
    input_start = input_stop - lookback_hours
    possible = input_start >= 0
    origins, input_start, input_stop = origins[possible], input_start[possible], input_stop[possible]
    bad_input = np.r_[0, np.cumsum(~input_ok, dtype=np.int64)]
    bad_target = np.r_[0, np.cumsum(~target_ok, dtype=np.int64)]
    good = ((bad_input[input_stop] - bad_input[input_start] == 0)
            & (bad_target[origins + horizon + 1] - bad_target[origins + 1] == 0))
    return origins[good], input_start[good], input_stop[good]


def make_samples(frame, origins, input_start, input_stop, horizon=HORIZON):
    dates = frame.timestamp_utc
    return pd.DataFrame({
        "sample_id": np.arange(len(origins), dtype=np.int64),
        "t0_utc": dates.iloc[origins].reset_index(drop=True),
        "input_start_utc": dates.iloc[input_start].reset_index(drop=True),
        "input_end_utc": dates.iloc[input_stop - 1].reset_index(drop=True),
        "target_start_utc": dates.iloc[origins + 1].reset_index(drop=True),
        "target_end_utc": dates.iloc[origins + horizon].reset_index(drop=True),
    })


def write_array(path, values):
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def export_one(frame, values, origins, input_start, input_stop, days, output_dir, source_sha256):
    lookback = days * 24
    count = len(origins)
    inputs = np.empty((count, lookback, len(FEATURES)), dtype=np.float32)
    targets = np.empty((count, HORIZON), dtype=np.float32)
    for position, (start, stop, origin) in enumerate(zip(input_start, input_stop, origins)):
        inputs[position] = values[start:stop]
        targets[position] = values[origin + 1:origin + HORIZON + 1, 0]
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError(f"Non-finite exported values in {days}-day windows")
    samples = make_samples(frame, origins, input_start, input_stop)
    recovered_input = np.column_stack([
        frame[f"{field}_is_recovered"].to_numpy(dtype=np.bool_) for field in FEATURES
    ]).any(axis=1)
    recovered_target = frame["speed_km_s_is_recovered"].to_numpy(dtype=np.bool_)
    input_recovery_count = np.r_[0, np.cumsum(recovered_input, dtype=np.int64)]
    target_recovery_count = np.r_[0, np.cumsum(recovered_target, dtype=np.int64)]
    has_recovered_input = (input_recovery_count[input_stop] - input_recovery_count[input_start]) > 0
    has_recovered_target = (
        target_recovery_count[origins + HORIZON + 1] - target_recovery_count[origins + 1]
    ) > 0
    years = {str(year): int(count) for year, count in
             samples.t0_utc.dt.year.value_counts().sort_index().items()}
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Original or recovered valid operational measurements; no interpolated cells",
        "input_features": list(FEATURES), "input_units": list(FEATURE_UNITS),
        "target": "speed_km_s", "target_unit": "km/s",
        "forecast_origin_cadence_hours": STEP_HOURS,
        "assumed_input_latency_hours": LATENCY_HOURS,
        "lookback_hours": lookback, "horizon_hours": HORIZON,
        "input_interval": "t0 - lookback_hours through t0 - 1 hour, inclusive",
        "target_interval": "t0 + 1 hour through t0 + 72 hours, inclusive",
        "samples": count, "inputs_shape": list(inputs.shape), "targets_shape": list(targets.shape),
        "samples_with_recovered_input": int(has_recovered_input.sum()),
        "samples_with_recovered_target": int(has_recovered_target.sum()),
        "samples_with_any_recovered_value": int((has_recovered_input | has_recovered_target).sum()),
        "dtype": "float32", "first_t0_utc": samples.t0_utc.iloc[0].isoformat() if count else None,
        "last_t0_utc": samples.t0_utc.iloc[-1].isoformat() if count else None,
        "samples_by_year": years, "source_sha256": source_sha256,
        "training_note": "Windows overlap. Split chronologically with a horizon-aware gap; do not randomly split sample rows.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        pending = Path(temporary)
        write_array(pending / "inputs.npy", inputs)
        write_array(pending / "targets.npy", targets)
        samples.to_parquet(pending / "samples.parquet", index=False)
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for name in ("inputs.npy", "targets.npy", "samples.parquet", "manifest.json"):
            (pending / name).replace(output_dir / name)
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/quality/ace_interpolated_all_internal.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training_windows"))
    args = parser.parse_args(argv)
    source = args.source if args.source.is_absolute() else root / args.source
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        frame = read_source(source)
        input_ok, target_ok = valid_masks(frame)
        values = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        results = {}
        for days in DAYS:
            origins, starts, stops = select_origins(input_ok, target_ok, days * 24)
            folder = output_dir / f"{days}d_72h"
            results[f"{days}d_72h"] = export_one(
                frame, values, origins, starts, stops, days, folder, source_sha256)
        summary = {"source": str(source), "output_dir": str(output_dir),
                   "folders": {key: {"samples": value["samples"],
                                    "first_t0_utc": value["first_t0_utc"],
                                    "last_t0_utc": value["last_t0_utc"]}
                               for key, value in results.items()}}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
