import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_interpolated_ace as interpolation


SWEPAM = ("# Hourly Averaged Real-time SWEPAM values\n"
          "2024 03 01 0000 61 0 0 5.0 400.0 100000.0\n"
          "2024 03 01 0300 61 0 0 8.0 490.0 130000.0\n")
MAG = ("# Hourly Averaged Real-time Interplanetary Magnetic Field Values\n"
       "# Source: ACE Satellite - Magnetometer\n"
       "# Magnetometer values are in GSM coordinates.\n"
       "2024 03 01 0000 61 0 0 -4.0 -1.0 -3.0 5.0 -20.0 359.0\n"
       "2024 03 01 0300 61 0 0 -1.0 -4.0 -6.0 8.0 -17.0 1.0\n")


class InterpolationTests(unittest.TestCase):
    def test_all_internal_gaps_and_edges(self):
        values = np.array([np.nan, 10.0, np.nan, np.nan, 40.0, np.nan])
        flags, gaps, largest = interpolation.interpolate_internal(values)
        np.testing.assert_allclose(values[1:5], [10, 20, 30, 40])
        self.assertTrue(np.isnan(values[0]) and np.isnan(values[5]))
        self.assertEqual(np.flatnonzero(flags).tolist(), [2, 3])
        self.assertEqual(gaps.tolist(), [0, 0, 2, 2, 0, 0])
        self.assertEqual(largest, 2)

    def test_longitude_crosses_zero(self):
        values = np.array([359.0, np.nan, 1.0])
        interpolation.interpolate_internal(values, circular=True)
        np.testing.assert_allclose(values, [359, 0, 1])

    def test_dataset_provenance_and_recovered_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dirname, suffix, content in (
                ("ace_hourly", "ace_swepam_1h", SWEPAM),
                ("ace_mag_hourly", "ace_mag_1h", MAG),
            ):
                raw = root / "data" / dirname / "raw"
                raw.mkdir(parents=True)
                (raw / f"202403_{suffix}.txt").write_text(content)
                (raw.parent / "download_report.json").write_text('{"start":"2024-03-01"}')
            recovered = root / "recovered.csv"
            with recovered.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("instrument", "timestamp_utc", "field", "value", "source_file"))
                writer.writeheader()
                writer.writerow({"instrument": "ACE/SWEPAM", "timestamp_utc": "2024-03-01T01:00:00Z",
                                 "field": "speed_km_s", "value": "430", "source_file": "fixture.txt"})
            output, report = root / "result.parquet", root / "result.json"
            summary = interpolation.build(root, recovered, output, report)
            self.assertEqual(summary["recovered_cells_loaded"], 1)
            frame = pq.read_table(output).slice(0, 5).to_pydict()
            self.assertEqual(frame["speed_km_s"][:4], [400, 430, 460, 490])
            self.assertTrue(frame["speed_km_s_is_recovered"][1])
            self.assertEqual(frame["speed_km_s_interpolation_gap_hours"][2], 1)
            self.assertEqual(frame["density_cm3_interpolation_gap_hours"][1:3], [2, 2])
            self.assertAlmostEqual(frame["longitude_gsm_deg"][1], 359 + 2 / 3)
            self.assertAlmostEqual(frame["longitude_gsm_deg"][2], 1 / 3)
            self.assertIsNone(frame["speed_km_s"][4])
            self.assertEqual(json.loads(report.read_text())["fields"]["speed_km_s"]["interpolated"], 1)


if __name__ == "__main__":
    unittest.main()
