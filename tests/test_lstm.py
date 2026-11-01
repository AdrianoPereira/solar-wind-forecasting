from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from benchlab_forecasting.lstm import (
        LSTMConfig, LSTMForecaster, causal_window, indexed_frame, masked_mse, supervised_samples,
    )
    from benchlab_forecasting.lstm_cli import main


@unittest.skipUnless(HAS_TORCH, "Install the lstm extra to test PyTorch")
class LSTMTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
        self.frame = pd.DataFrame({"ds": dates, "y": 420 + 30 * np.sin(np.arange(len(dates)) / 20)})
        self.config = LSTMConfig(horizon=6, lookback_hours=24, hidden_size=4,
                                 epochs=3, patience=2, input_group="speed", threads=1)
        self.train = dates[24:168:6]
        self.validation = dates[180:204:6]

    def test_current_and_future_inputs_are_excluded_by_latency(self):
        origin = self.frame.ds.iloc[400]
        values, flags, anchor = causal_window(indexed_frame(self.frame), origin, ["y"], self.config)
        self.assertEqual(anchor, self.frame.y.iloc[399])
        self.assertEqual(flags[-1, 0], 1)
        changed = self.frame.copy()
        changed.loc[changed.ds >= origin, "y"] = 1e8
        other = causal_window(indexed_frame(changed), origin, ["y"], self.config)
        np.testing.assert_array_equal(values, other[0])

    def test_source_availability_and_stale_history(self):
        origin = self.frame.ds.iloc[400]
        frame = self.frame.copy()
        frame["y__available_at"] = frame.ds + pd.Timedelta(hours=1)
        frame.loc[390:400, "y__available_at"] = origin + pd.Timedelta(hours=3)
        values, flags, anchor = causal_window(indexed_frame(frame), origin, ["y"], self.config)
        self.assertEqual(anchor, frame.y.iloc[389])
        self.assertTrue(np.isnan(values[-5:]).all())
        with self.assertRaisesRegex(ValueError, "recent"):
            causal_window(indexed_frame(frame), origin, ["y"], replace(self.config, max_age_hours=6))

    def test_missing_targets_are_masked_and_have_no_gradient(self):
        prediction = torch.tensor([[1., 999., 5.]], requires_grad=True)
        target = torch.tensor([[2., float("nan"), 3.]])
        loss = masked_mse(prediction, target)
        self.assertEqual(float(loss.detach()), 2.5)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 1]), 0)

    def test_label_publication_cutoff(self):
        frame = self.frame.copy()
        frame["y__available_at"] = frame.ds + pd.Timedelta(hours=1)
        frame.loc[25, "y__available_at"] = self.frame.ds.iloc[100]
        frame.loc[26, "y__available_at"] = pd.NaT
        samples = supervised_samples(indexed_frame(frame), self.train[:1], ["y"], self.config,
                                      self.frame.ds.iloc[40])
        self.assertTrue(np.isnan(samples[3][0, 0]))
        self.assertTrue(np.isnan(samples[3][0, 1]))
        self.assertTrue(np.isfinite(samples[3][0, 2:]).all())

    def test_purge_and_model_availability(self):
        model = LSTMForecaster(self.config)
        with self.assertRaisesRegex(ValueError, "targets must end"):
            model.fit(self.frame, self.train, self.train[-2:])
        model.fit(self.frame, self.train, self.validation)
        with self.assertRaisesRegex(ValueError, "precedes availability"):
            model.predict(self.frame, self.validation[0])

    def test_training_future_invariance_and_checkpoint_roundtrip(self):
        model = LSTMForecaster(self.config).fit(self.frame, self.train, self.validation)
        changed = self.frame.copy()
        changed.loc[300:, "y"] = 100000
        other = LSTMForecaster(self.config).fit(changed, self.train, self.validation)
        np.testing.assert_array_equal(model.means, other.means)
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
        origin = self.frame.ds.iloc[400]
        predicted = model.predict(self.frame, origin)
        modified_future = self.frame.copy()
        modified_future.loc[modified_future.ds >= origin, "y"] = 1e8
        np.testing.assert_array_equal(predicted.speed_km_s, model.predict(modified_future, origin).speed_km_s)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            restored = LSTMForecaster.load(path).predict(self.frame, origin)
            np.testing.assert_array_equal(restored.speed_km_s, predicted.speed_km_s)
            self.assertEqual(len(restored), 6)
            self.assertEqual(restored.timestamp_utc.iloc[0], origin + pd.Timedelta(hours=1))

    def test_cli_experiment_and_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "speed.csv"
            self.frame.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_csv(source, index=False)
            output = root / "run"
            self.assertEqual(main([
                "experiment", "--data", str(source), "--input-group", "speed",
                "--train-start", "2024-01-02", "--validation-start", "2024-01-09",
                "--validation-hours", "24", "--test-start", "2024-02-02", "--test-hours", "12",
                "--horizon", "6", "--lookback-hours", "24", "--hidden-size", "4",
                "--epochs", "2", "--threads", "1", "--output-dir", str(output)]), 0)
            self.assertTrue((output / "test/report.json").exists())
            forecast_output = root / "forecast"
            self.assertEqual(main(["forecast", "--data", str(source), "--model", str(output / "model.pt"),
                                   "--origin", "2024-02-03", "--output-dir", str(forecast_output)]), 0)
            self.assertEqual(len(pd.read_csv(forecast_output / "forecast.csv")), 6)


if __name__ == "__main__":
    unittest.main()
