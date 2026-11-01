#!/usr/bin/env python3
"""Baixa meses ACE com lacunas e salva somente os valores recuperados.

A fonte NOAA/SWPC disponibiliza arquivos mensais, não downloads por hora. Por
padrão, tenta meses sem arquivo local e meses com horas ausentes. Use
--retry-invalid para também consultar meses com registros de status inválido ou
valores individuais inválidos. Os arquivos brutos e Parquet originais permanecem
intactos; os meses baixados ficam em um cache separado.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory

import download_ace_hourly as swepam
import download_ace_mag_hourly as mag


SOURCES = {
    "ACE/SWEPAM": ("swepam", "ace_swepam_1h", ("speed_km_s", "density_cm3", "temperature_k"), swepam),
    "ACE/MAG": ("mag", "ace_mag_1h", mag.MAG_FIELDS, mag),
}
ACQUISITION_REASONS = {"month_file_absent", "hour_absent"}
INVALID_REASONS = {"status_nonzero", "invalid_value"}
RECOVERED_COLUMNS = ("instrument", "timestamp_utc", "field", "value", "source_file")


def read_targets(path, retry_invalid=False, instrument="all"):
    """Group missing cells by instrument and source month."""
    allowed = ACQUISITION_REASONS | (INVALID_REASONS if retry_invalid else set())
    targets = defaultdict(lambda: defaultdict(set))
    reasons = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"instrument", "timestamp_utc", "missing_fields", "reason"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        for row in reader:
            name = row["instrument"]
            if name not in SOURCES:
                raise ValueError(f"Unknown instrument in audit: {name}")
            if instrument != "all" and SOURCES[name][0] != instrument:
                continue
            reason = row["reason"]
            if reason not in ACQUISITION_REASONS | INVALID_REASONS:
                raise ValueError(f"Unknown audit reason: {reason}")
            if reason not in allowed:
                continue
            stamp = row["timestamp_utc"]
            parsed_stamp = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
            if parsed_stamp.minute or parsed_stamp.second:
                raise ValueError(f"Timestamp is not hourly: {stamp}")
            fields = row["missing_fields"].split("|")
            unknown = set(fields) - set(SOURCES[name][2])
            if unknown or not fields or "" in fields:
                raise ValueError(f"Invalid missing fields for {name}: {fields}")
            targets[(name, stamp[:4] + stamp[5:7])][stamp].update(fields)
            reasons[reason] += 1
    return targets, reasons


def load_month(cache_dir, name, month, timeout, attempts, ca_data, refresh=False):
    """Fetch and validate one monthly file, using a checksum-verified cache."""
    _, suffix, _, module = SOURCES[name]
    filename = f"{month}_{suffix}.txt"
    path = cache_dir / SOURCES[name][0] / filename
    metadata_path = path.with_suffix(".json")
    url = module.BASE_URL + filename
    if not refresh and path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = path.read_bytes()
            if metadata.get("url") == url and metadata.get("sha256") == hashlib.sha256(payload).hexdigest():
                parsed = module.parse_month(payload.decode("utf-8-sig"), month, filename)
                return parsed, "cache", path
        except (UnicodeError, ValueError):
            pass
    payload = module.fetch(url, timeout, attempts, ca_data)
    parsed = module.parse_month(payload.decode("utf-8-sig"), month, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest()}
    module.atomic_write(path, payload)
    module.atomic_write(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))
    return parsed, "download", path


def recovered_values(name, requested, parsed, source_file):
    """Emit only cells absent in the original audit, never other monthly values."""
    output = []
    for row in parsed:
        stamp = row["timestamp_utc"]
        for field in sorted(requested.get(stamp, ())):
            value = row[field]
            if value is not None:
                output.append({"instrument": name, "timestamp_utc": stamp, "field": field,
                               "value": value, "source_file": str(source_file)})
    return output


def read_recovered(path):
    """Keep values obtained by earlier runs, including other instruments."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != set(RECOVERED_COLUMNS):
            raise ValueError(f"Unexpected columns in existing output: {path}")
        return {(row["instrument"], row["timestamp_utc"], row["field"]): row
                for row in reader}


def write_results(output, report_path, recovered, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RECOVERED_COLUMNS)
            writer.writeheader()
            writer.writerows(recovered)
        pending.replace(output)
    with TemporaryDirectory(dir=report_path.parent) as temporary:
        pending = Path(temporary) / report_path.name
        pending.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pending.replace(report_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/quality/ace_missing_history.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/quality/ace_recovery_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/quality/ace_recovered_values.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/quality/ace_recovery_report.json"))
    parser.add_argument("--instrument", choices=("all", "swepam", "mag"), default="all")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Baixar novamente meses já presentes no cache de recuperação")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar o plano sem acessar a rede nem escrever arquivos")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--ca-file", type=Path)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts < 1:
        parser.error("timeout and attempts must be positive")
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    audit, cache_dir, output, report_path = map(resolve, (args.audit, args.cache_dir, args.output, args.report))
    try:
        targets, reasons = read_targets(audit, args.retry_invalid, args.instrument)
        ca_data = swepam.read_ca_file(resolve(args.ca_file)) if args.ca_file and not args.dry_run else None
    except (OSError, ValueError, KeyError, ssl.SSLError) as exc:
        parser.error(str(exc))
    selected = sorted(targets)
    planned = {"months": len(selected), "by_instrument": dict(Counter(name for name, _ in selected)),
               "reasons": dict(reasons), "requested_cells": sum(len(fields) for hours in targets.values()
                                                        for fields in hours.values())}
    if args.dry_run:
        print(json.dumps({"plan": planned, "selected_months": selected}, indent=2))
        return 0

    try:
        prior = read_recovered(output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    recovered, attempts = [], []
    for name, month in selected:
        requested = targets[(name, month)]
        try:
            parsed, provenance, source_file = load_month(
                cache_dir, name, month, args.timeout, args.attempts, ca_data, args.refresh)
            try:
                source_label = source_file.relative_to(root)
            except ValueError:
                source_label = source_file
            rows = recovered_values(name, requested, parsed, source_label)
            recovered.extend(rows)
            attempts.append({"instrument": name, "month": month, "result": provenance,
                             "requested_cells": sum(map(len, requested.values())),
                             "recovered_cells": len(rows)})
            print(f"{name} {month}: {provenance}, {len(rows)} values recovered", flush=True)
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            attempts.append({"instrument": name, "month": month, "result": "failure", "error": str(exc),
                             "requested_cells": sum(map(len, requested.values())), "recovered_cells": 0})
            print(f"{name} {month}: failed: {exc}", flush=True)
    for row in recovered:
        prior[(row["instrument"], row["timestamp_utc"], row["field"])] = row
    merged = [prior[key] for key in sorted(prior)]
    remaining = sum((name, stamp, field) not in prior
                    for (name, _), hours in targets.items()
                    for stamp, fields in hours.items() for field in fields)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "audit": str(audit),
              "plan": planned, "recovered_cells_this_run": len(recovered),
              "recovered_cells_in_output": len(merged),
              "unresolved_cells": remaining,
              "attempts": attempts,
              "note": "Only original missing cells are exported. The source serves whole months; original raw/Parquet files are unchanged."}
    write_results(output, report_path, merged, report)
    print(json.dumps({"output": str(output), "report": str(report_path),
                      "recovered_cells_this_run": len(recovered),
                      "recovered_cells_in_output": len(merged),
                      "failed_months": sum(a["result"] == "failure" for a in attempts)},
                     indent=2))
    return 1 if any(a["result"] == "failure" for a in attempts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
