#!/usr/bin/env python3
"""List missing hourly ACE values across the locally cached monthly archive.

One CSV row represents one affected hour and instrument. A missing monthly file,
an absent hourly record, a non-nominal status, and invalid individual values are
kept distinct. The audit starts at the earliest requested date in each download
report and ends with the last cached month for that instrument.
"""

import argparse
import calendar
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from download_ace_hourly import parse_month as parse_swepam_month
from download_ace_mag_hourly import MAG_FIELDS, parse_month as parse_mag_month


SOURCES = (
    ("ACE/SWEPAM", "ace_hourly", "ace_swepam_1h",
     ("speed_km_s", "density_cm3", "temperature_k"), parse_swepam_month),
    ("ACE/MAG", "ace_mag_hourly", "ace_mag_1h", MAG_FIELDS, parse_mag_month),
)
CSV_FIELDS = ("instrument", "timestamp_utc", "missing_fields", "reason", "status", "source_file")


def month_keys(first, last):
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        yield f"{year:04d}{month:02d}"
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def audit_source(root, source, writer, shared_last):
    name, directory, suffix, fields, parser = source
    data_dir = root / "data" / directory
    raw_dir = data_dir / "raw"
    cached = sorted(raw_dir.glob(f"??????_{suffix}.txt"))
    if not cached:
        raise ValueError(f"No cached monthly files for {name}: {raw_dir}")
    cached_months = [date(int(p.name[:4]), int(p.name[4:6]), 1) for p in cached]
    first = min(cached_months)
    report = data_dir / "download_report.json"
    if report.exists():
        requested_start = date.fromisoformat(json.loads(report.read_text(encoding="utf-8"))["start"])
        first = min(first, requested_start)
    # Audit every instrument through the latest month cached for any instrument.
    # Otherwise a trailing sequence of absent files is invisible.
    last = max(max(cached_months), shared_last)
    counts = Counter()
    field_counts = Counter()

    for month in month_keys(first, last):
        source_file = raw_dir / f"{month}_{suffix}.txt"
        records = {}
        file_exists = source_file.exists()
        if file_exists:
            parsed = parser(source_file.read_text(encoding="utf-8-sig"), month, source_file.name)
            records = {row["timestamp_utc"]: row for row in parsed}
        hours = calendar.monthrange(int(month[:4]), int(month[4:]))[1] * 24
        start = datetime(int(month[:4]), int(month[4:]), 1, tzinfo=timezone.utc)
        for offset in range(hours):
            stamp = (start + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
            counts["expected_hours"] += 1
            row = records.get(stamp)
            if row is None:
                missing = fields
                reason = "month_file_absent" if not file_exists else "hour_absent"
                status = ""
            else:
                missing = tuple(field for field in fields if row[field] is None)
                reason = "status_nonzero" if row["status"] != 0 else "invalid_value"
                status = row["status"]
            if not missing:
                continue
            counts[reason] += 1
            for field in missing:
                field_counts[field] += 1
            writer.writerow({"instrument": name, "timestamp_utc": stamp,
                             "missing_fields": "|".join(missing), "reason": reason,
                             "status": status, "source_file": str(source_file.relative_to(root))})

    counts["affected_hours"] = sum(counts[key] for key in
                                   ("month_file_absent", "hour_absent", "status_nonzero", "invalid_value"))
    return {"instrument": name, "first_month": first.isoformat(), "last_month": last.isoformat(),
            **dict(counts), "missing_by_field": dict(field_counts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/quality/ace_missing_history.csv"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    all_cached = [path for _, directory, suffix, _, _ in SOURCES
                  for path in (root / "data" / directory / "raw").glob(f"??????_{suffix}.txt")]
    if not all_cached:
        parser.error("No cached ACE monthly files found")
    shared_last = max(date(int(path.name[:4]), int(path.name[4:6]), 1)
                      for path in all_cached)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            summaries = [audit_source(root, source, writer, shared_last) for source in SOURCES]
        pending.replace(output)
    print(json.dumps({"output": str(output), "sources": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
