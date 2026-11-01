import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_prophet_complete_windows.py"
spec = importlib.util.spec_from_file_location("prophet_complete_windows", SCRIPT)
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


class ProphetCompleteWindowTests(unittest.TestCase):
    def test_fit_uses_exact_past_window_and_serializes(self):
        from prophet.serialize import model_from_json, model_to_json

        origin = pd.Timestamp("2024-01-04T00:00:00Z")
        sample = pd.Series({
            "t0_utc": origin,
            "input_start_utc": origin - pd.Timedelta(hours=72),
            "input_end_utc": origin - pd.Timedelta(hours=1),
        })
        speed = 400 + 8 * np.sin(np.arange(72) / 8)
        model, prediction, dates = windows.fit_window(
            speed, sample, windows.PROPHET_VARIANTS["flat"])
        self.assertEqual(len(model.history), 72)
        self.assertEqual(model.history.ds.iloc[0], pd.Timestamp("2024-01-01"))
        self.assertEqual(model.history.ds.iloc[-1], pd.Timestamp("2024-01-03T23:00:00"))
        self.assertEqual(dates[0], origin + pd.Timedelta(hours=1))
        self.assertEqual(dates[-1], origin + pd.Timedelta(hours=72))
        self.assertTrue(np.isfinite(prediction).all())
        restored = model_from_json(model_to_json(model))
        future = pd.DataFrame({"ds": dates.tz_localize(None)})
        np.testing.assert_allclose(
            np.maximum(restored.predict(future).yhat.to_numpy(), 0), prediction)

    def test_rejects_incomplete_or_shifted_window(self):
        origin = pd.Timestamp("2024-01-04T00:00:00Z")
        sample = pd.Series({"t0_utc": origin,
                            "input_start_utc": origin - pd.Timedelta(hours=72),
                            "input_end_utc": origin - pd.Timedelta(hours=1)})
        speed = np.full(72, 400.0)
        speed[10] = np.nan
        with self.assertRaisesRegex(ValueError, "inválida"):
            windows.fit_window(speed, sample, windows.PROPHET_VARIANTS["flat"])
        speed[10] = 400.0
        sample["input_end_utc"] = origin
        with self.assertRaisesRegex(ValueError, "inválida"):
            windows.fit_window(speed, sample, windows.PROPHET_VARIANTS["flat"])


if __name__ == "__main__":
    unittest.main()
