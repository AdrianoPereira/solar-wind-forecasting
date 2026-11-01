import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_missing_ace as recovery


SWEPAM_PAYLOAD = ("# Hourly Averaged Real-time SWEPAM values\n"
                  "2024 03 01 0000 61 0 0 5.0 400.0 100000.0\n"
                  "2024 03 01 0100 61 0 0 6.0 410.0 110000.0\n").encode()
AUDIT_COLUMNS = ("instrument", "timestamp_utc", "missing_fields", "reason", "status", "source_file")


class MissingAceDownloadTests(unittest.TestCase):
    def test_swepam_next_month_boundary_record_is_ignored(self):
        following = "2024 04 01 0000 61 0 0 7.0 420.0 120000.0\n"
        rows = recovery.swepam.parse_month(
            SWEPAM_PAYLOAD.decode() + following, "202403", "fixture.txt")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["timestamp_utc"], "2024-03-01T01:00:00Z")

    def write_audit(self, path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerow({"instrument": "ACE/SWEPAM", "timestamp_utc": "2024-03-01T00:00:00Z",
                             "missing_fields": "speed_km_s|density_cm3|temperature_k",
                             "reason": "hour_absent"})
            writer.writerow({"instrument": "ACE/SWEPAM", "timestamp_utc": "2024-03-01T01:00:00Z",
                             "missing_fields": "speed_km_s", "reason": "status_nonzero"})

    def test_default_selects_acquisition_gaps_only(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.csv"
            self.write_audit(audit)
            targets, reasons = recovery.read_targets(audit)
            self.assertEqual(reasons, {"hour_absent": 1})
            self.assertEqual(targets[("ACE/SWEPAM", "202403")],
                             {"2024-03-01T00:00:00Z":
                              {"speed_km_s", "density_cm3", "temperature_k"}})
            targets, reasons = recovery.read_targets(audit, retry_invalid=True)
            self.assertEqual(reasons["status_nonzero"], 1)
            self.assertIn("2024-03-01T01:00:00Z", targets[("ACE/SWEPAM", "202403")])

    def test_cache_and_only_requested_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(recovery.swepam, "fetch", return_value=SWEPAM_PAYLOAD) as fetch:
                parsed, provenance, source = recovery.load_month(
                    cache, "ACE/SWEPAM", "202403", 5, 1, None)
                self.assertEqual(provenance, "download")
                self.assertEqual(fetch.call_count, 1)
                rows = recovery.recovered_values(
                    "ACE/SWEPAM", {"2024-03-01T00:00:00Z": {"speed_km_s"}}, parsed, source)
                self.assertEqual([(row["timestamp_utc"], row["field"], row["value"]) for row in rows],
                                 [("2024-03-01T00:00:00Z", "speed_km_s", 400.0)])
                _, provenance, _ = recovery.load_month(cache, "ACE/SWEPAM", "202403", 5, 1, None)
                self.assertEqual(provenance, "cache")
                self.assertEqual(fetch.call_count, 1)
                source.write_text("corrupt", encoding="utf-8")
                _, provenance, _ = recovery.load_month(cache, "ACE/SWEPAM", "202403", 5, 1, None)
                self.assertEqual(provenance, "download")
                self.assertEqual(fetch.call_count, 2)

    def test_main_preserves_previous_results_and_reports_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audit, output, report = (folder / name for name in ("audit.csv", "values.csv", "report.json"))
            self.write_audit(audit)
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=recovery.RECOVERED_COLUMNS)
                writer.writeheader()
                writer.writerow({"instrument": "ACE/MAG", "timestamp_utc": "2024-03-01T00:00:00Z",
                                 "field": "bt_nt", "value": 5.3, "source_file": "earlier.txt"})
            args = ["--audit", str(audit), "--cache-dir", str(folder / "cache"),
                    "--output", str(output), "--report", str(report)]
            with patch.object(recovery.swepam, "fetch", return_value=SWEPAM_PAYLOAD) as fetch:
                with redirect_stdout(StringIO()):
                    self.assertEqual(recovery.main(args), 0)
                    self.assertEqual(recovery.main(args), 0)
                self.assertEqual(fetch.call_count, 1)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["instrument"] == "ACE/MAG" for row in rows), 1)
            self.assertEqual(json.loads(report.read_text())["unresolved_cells"], 0)


if __name__ == "__main__":
    unittest.main()
