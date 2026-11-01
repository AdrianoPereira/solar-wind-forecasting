import csv
from datetime import date
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_ace_missing as audit


class AuditAceMissingTests(unittest.TestCase):
    def test_audits_trailing_month_missing_from_one_instrument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "ace_mag_hourly" / "raw"
            raw.mkdir(parents=True)
            (raw / "202512_ace_mag_1h.txt").write_text(
                "# Hourly Averaged Real-time Interplanetary Magnetic Field Values\n"
                "# Source: ACE Satellite - Magnetometer\n"
                "# GSM coordinates\n"
                "2025 12 01 0000 61010 0 0 -4.1 -0.4 -3.3 5.3 -39.2 185.3\n",
                encoding="utf-8",
            )
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=audit.CSV_FIELDS)
            writer.writeheader()
            summary = audit.audit_source(root, audit.SOURCES[1], writer, date(2026, 1, 1))
            rows = list(csv.DictReader(StringIO(output.getvalue())))
            january = [row for row in rows if row["timestamp_utc"].startswith("2026-01")]
            self.assertEqual(summary["last_month"], "2026-01-01")
            self.assertEqual(summary["month_file_absent"], 744)
            self.assertEqual(len(january), 744)


if __name__ == "__main__":
    unittest.main()
