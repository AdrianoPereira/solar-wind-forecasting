import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_ace_mag_hourly.py"
spec = importlib.util.spec_from_file_location("ace_mag", SCRIPT)
mag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mag)
HEADER = """# Hourly Averaged Real-time Interplanetary Magnetic Field Values
# Source: ACE Satellite - Magnetometer
# Magnetometer values are in GSM coordinates.
"""
ROW = "2024 03 01 0000 60370 0 0 -4.1 -0.4 -3.3 5.3 -39.2 185.3"


class MagneticFieldTests(unittest.TestCase):
    def parse(self, row=ROW):
        return mag.parse_month(HEADER + row, "202403", "fixture.txt")[0]

    def test_signed_components_and_utc(self):
        row = self.parse()
        self.assertEqual(row["bx_gsm_nt"], -4.1)
        self.assertEqual(row["bz_gsm_nt"], -3.3)
        self.assertEqual(row["timestamp_utc"], "2024-03-01T00:00:00Z")
        self.assertTrue(row["is_valid"])

    def test_bad_status_preserves_raw(self):
        row = self.parse(ROW.replace("60370 0 0", "60370 0 9"))
        self.assertFalse(row["is_valid"])
        self.assertEqual(row["bx_gsm_nt_raw"], -4.1)
        self.assertTrue(all(row[field] is None for field in mag.MAG_FIELDS))

    def test_sentinel_and_physical_ranges(self):
        row = self.parse(ROW.replace("-3.3", "-999.9"))
        self.assertIsNone(row["bz_gsm_nt"])
        self.assertFalse(row["is_valid"])
        self.assertIsNone(self.parse(ROW.replace(" 5.3 ", " -5.3 "))["bt_nt"])
        row = self.parse(ROW.replace("-39.2", "-99.0"))
        self.assertIsNone(row["latitude_gsm_deg"])
        self.assertTrue(row["is_valid"])

    def test_malformed_duplicate_and_wrong_month(self):
        for content in [ROW + " extra", ROW.replace("-4.1", "nan"),
                        ROW.replace("2024 03", "2024 04"), ROW + "\n" + ROW]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.parse(content)
        with self.assertRaises(ValueError):
            mag.parse_month("<html>Error</html>", "202403", "fixture.txt")

    def test_next_month_boundary_record_is_ignored(self):
        following = ROW.replace("2024 03 01", "2024 04 01")
        rows = mag.parse_month(HEADER + ROW + "\n" + following,
                               "202403", "fixture.txt")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp_utc"], "2024-03-01T00:00:00Z")

    def test_cache_integrity_and_refresh(self):
        payload = (HEADER + ROW).encode()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with patch.object(mag, "fetch", return_value=payload) as fetch:
                first = mag.download_month("202403", raw, False, 5, 1)
                self.assertIsNone(first[2])
                self.assertFalse(first[3])
                second = mag.download_month("202403", raw, False, 5, 1)
                self.assertTrue(second[3])
                self.assertEqual(fetch.call_count, 1)
                (raw / "202403_ace_mag_1h.txt").write_text("corrupted")
                self.assertFalse(mag.download_month("202403", raw, False, 5, 1)[3])
                self.assertEqual(fetch.call_count, 2)
                mag.download_month("202403", raw, True, 5, 1)
                self.assertEqual(fetch.call_count, 3)


if __name__ == "__main__":
    unittest.main()
