import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_complete_ace_windows.py"
spec = importlib.util.spec_from_file_location("complete_ace_windows", SCRIPT)
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


class CompleteWindowTests(unittest.TestCase):
    def test_strict_origins_use_past_inputs_and_future_targets(self):
        input_ok = np.ones(36, dtype=bool)
        target_ok = np.ones(36, dtype=bool)
        input_ok[5] = False
        target_ok[15] = False
        origins, starts, stops = windows.select_origins(
            input_ok, target_ok, lookback_hours=6, horizon=4, step_hours=6)
        self.assertEqual(origins.tolist(), [18, 24, 30])
        self.assertEqual(starts.tolist(), [12, 18, 24])
        self.assertEqual(stops.tolist(), [18, 24, 30])
        frame = pd.DataFrame({"timestamp_utc": pd.date_range(
            "2024-01-01", periods=36, freq="h", tz="UTC")})
        samples = windows.make_samples(frame, origins, starts, stops, horizon=4)
        self.assertEqual(samples.input_end_utc.iloc[0], pd.Timestamp("2024-01-01T17:00:00Z"))
        self.assertEqual(samples.target_start_utc.iloc[0], pd.Timestamp("2024-01-01T19:00:00Z"))

    def test_interpolated_cells_are_invalid_and_recovered_cells_are_observed(self):
        frame = pd.DataFrame({field: np.ones(4) for field in windows.FEATURES})
        for field in windows.FEATURES:
            frame[f"{field}_is_interpolated"] = False
            frame[f"{field}_is_recovered"] = False
        frame.loc[1, "speed_km_s_is_interpolated"] = True
        frame.loc[2, "bt_nt_is_recovered"] = True
        frame.loc[2, "speed_km_s_is_recovered"] = True
        frame.loc[3, "density_cm3"] = np.nan
        input_ok, target_ok = windows.valid_masks(frame)
        self.assertEqual(input_ok.tolist(), [True, False, True, False])
        self.assertEqual(target_ok.tolist(), [True, False, True, True])


if __name__ == "__main__":
    unittest.main()
