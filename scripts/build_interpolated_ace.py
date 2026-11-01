#!/usr/bin/env python3
"""Build a separate hourly ACE dataset with all internal gaps interpolated.

Every numeric field is interpolated independently between the nearest valid
hours. Leading and trailing gaps remain null. The output records provenance and
the full length of each filled gap. Interpolated values are synthetic and must
not be treated as observed targets in validation or operational backtests.
"""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import download_ace_hourly as swepam
import download_ace_mag_hourly as mag


SOURCES = (
    ("ACE/SWEPAM", "swepam", "ace_hourly", "ace_swepam_1h",
     ("speed_km_s", "density_cm3", "temperature_k"), swepam.parse_month),
    ("ACE/MAG", "mag", "ace_mag_hourly", "ace_mag_1h", mag.MAG_FIELDS, mag.parse_month),
)
STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
HOUR = timedelta(hours=1)


def month_after(key):
    year, month = int(key[:4]), int(key[4:])
    return datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)


def source_range(root, source):
    _, _, directory, suffix, _, _ = source
    data_dir = root / "data" / directory
    files = sorted((data_dir / "raw").glob(f"??????_{suffix}.txt"))
    if not files:
        raise ValueError(f"No monthly files found in {data_dir / 'raw'}")
    first = datetime(int(files[0].name[:4]), int(files[0].name[4:6]), 1, tzinfo=timezone.utc)
    report = data_dir / "download_report.json"
    if report.exists():
        requested = date.fromisoformat(json.loads(report.read_text(encoding="utf-8"))["start"])
        first = min(first, datetime(requested.year, requested.month, 1, tzinfo=timezone.utc))
    return first, month_after(files[-1].name[:6]), files


def hour_index(stamp, first, size):
    instant = datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    delta = instant - first
    hours = int(delta.total_seconds() // 3600)
    if hours < 0 or hours >= size or delta != hours * HOUR:
        raise ValueError(f"Timestamp outside hourly grid: {stamp}")
    return hours


def original_values(root, first, size, ranges):
    values = {field: np.full(size, np.nan, dtype=np.float64)
              for source in SOURCES for field in source[4]}
    statuses = {source[1]: np.full(size, -1, dtype=np.int8) for source in SOURCES}
    file_counts = {}
    for source in SOURCES:
        name, key, _, _, fields, parser = source
        files = ranges[name][2]
        file_counts[key] = len(files)
        for path in files:
            month = path.name[:6]
            rows = parser(path.read_text(encoding="utf-8-sig"), month, path.name)
            for row in rows:
                index = hour_index(row["timestamp_utc"], first, size)
                if statuses[key][index] != -1:
                    raise ValueError(f"Duplicate {name} timestamp: {row['timestamp_utc']}")
                statuses[key][index] = row["status"]
                for field in fields:
                    if row[field] is not None:
                        values[field][index] = row[field]
    return values, statuses, file_counts


def add_recovered(path, first, size, values):
    recovered = {field: np.zeros(size, dtype=np.bool_) for field in values}
    if not path.exists():
        return recovered, 0
    count = 0
    source_fields = {source[0]: set(source[4]) for source in SOURCES}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"instrument", "timestamp_utc", "field", "value", "source_file"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing columns in {path}: {sorted(required)}")
        for row in reader:
            name, field = row["instrument"], row["field"]
            if name not in source_fields or field not in source_fields[name]:
                raise ValueError(f"Unknown recovered field: {name} {field}")
            index = hour_index(row["timestamp_utc"], first, size)
            value = float(row["value"])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite recovered value: {name} {field} {row['timestamp_utc']}")
            if recovered[field][index]:
                raise ValueError(f"Duplicate recovered cell: {name} {field} {row['timestamp_utc']}")
            if np.isfinite(values[field][index]):
                raise ValueError(f"Recovered cell already valid in original: {name} {field} {row['timestamp_utc']}")
            values[field][index] = value
            recovered[field][index] = True
            count += 1
    return recovered, count


def interpolate_internal(values, circular=False):
    """Fill complete internal gaps, returning flags, gap sizes, and maximum gap."""
    filled = np.zeros(len(values), dtype=np.bool_)
    gap_sizes = np.zeros(len(values), dtype=np.int32)
    valid = np.flatnonzero(np.isfinite(values))
    largest = 0
    for left, right in zip(valid[:-1], valid[1:]):
        gap = int(right - left - 1)
        if gap == 0:
            continue
        fraction = np.arange(1, gap + 1, dtype=np.float64) / (gap + 1)
        change = values[right] - values[left]
        if circular:
            change = (change + 180) % 360 - 180
        interpolated = values[left] + change * fraction
        values[left + 1:right] = interpolated % 360 if circular else interpolated
        filled[left + 1:right] = True
        gap_sizes[left + 1:right] = gap
        largest = max(largest, gap)
    return filled, gap_sizes, largest


def build(root, recovered_path, output, report_path):
    ranges = {source[0]: source_range(root, source) for source in SOURCES}
    first = min(item[0] for item in ranges.values())
    end = max(item[1] for item in ranges.values())
    size = int((end - first) / HOUR)
    values, statuses, file_counts = original_values(root, first, size, ranges)
    recovered, recovered_count = add_recovered(recovered_path, first, size, values)
    timestamps = pa.array([first + index * HOUR for index in range(size)],
                          type=pa.timestamp("us", tz="UTC"))
    columns = {"timestamp_utc": timestamps}
    counts = {}
    for source in SOURCES:
        _, key, _, _, fields, _ = source
        original_status = statuses[key]
        columns[f"{key}_original_status"] = pa.array(
            original_status, type=pa.int8(), mask=original_status < 0)
        for field in fields:
            series = values[field]
            original_count = int(np.isfinite(series).sum() - recovered[field].sum())
            interpolated, gaps, largest = interpolate_internal(
                series, circular=(field == "longitude_gsm_deg"))
            columns[field] = pa.array(series, type=pa.float64(), mask=~np.isfinite(series))
            columns[f"{field}_is_recovered"] = pa.array(recovered[field])
            columns[f"{field}_is_interpolated"] = pa.array(interpolated)
            columns[f"{field}_interpolation_gap_hours"] = pa.array(
                gaps, type=pa.int32(), mask=~interpolated)
            counts[field] = {"observed": original_count,
                             "recovered": int(recovered[field].sum()),
                             "interpolated": int(interpolated.sum()),
                             "remaining_null": int(np.isnan(series).sum()),
                             "largest_interpolated_gap_hours": largest}
    table = pa.table(columns).replace_schema_metadata({
        b"product": b"ACE operational hourly SWEPAM and MAG; derived interpolation",
        b"interpolation": b"linear between valid hourly endpoints for every internal gap",
        b"longitude_interpolation": b"shortest angular path modulo 360 degrees",
        b"warning": b"Synthetic values use future endpoints; do not use as observed validation targets or as real-time features",
        b"source_root": str(root).encode(),
    })
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output), "rows": size,
        "first_hour_utc": first.isoformat(),
        "last_hour_utc": (end - HOUR).isoformat(),
        "interpolation": "linear for every internal gap; no edge extrapolation; longitude uses shortest angular path",
        "source_month_files": file_counts,
        "source_ranges": {name: {"first_hour_utc": span[0].isoformat(),
                                 "last_hour_utc": (span[1] - HOUR).isoformat()}
                          for name, span in ranges.items()},
        "recovered_input": str(recovered_path), "recovered_cells_loaded": recovered_count,
        "fields": counts,
        "warning": "Interpolated values use future observations. Filter *_is_interpolated for training/validation and never score them as observed targets.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pending_output = output.with_suffix(output.suffix + ".part")
    pending_report = report_path.with_suffix(report_path.suffix + ".part")
    pq.write_table(table, pending_output, compression="zstd")
    pending_output.replace(output)
    pending_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending_report.replace(report_path)
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, default=Path("data/quality/ace_recovered_values.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/quality/ace_interpolated_all_internal.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/quality/ace_interpolated_all_internal_report.json"))
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else root / path
    try:
        report = build(root, resolve(args.recovered), resolve(args.output), resolve(args.report))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": report["output"], "rows": report["rows"],
                      "recovered_cells_loaded": report["recovered_cells_loaded"],
                      "fields": report["fields"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
