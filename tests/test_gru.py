from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from benchlab_forecasting.gru import DirectGRU, GRUConfig, GRUForecaster
    from benchlab_forecasting.gru_cli import main
    from benchlab_forecasting.lstm import LSTMConfig, LSTMForecaster


@unittest.skipUnless(HAS_TORCH, "Install the lstm extra for recurrent models")
class GRUTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
        self.frame = pd.DataFrame({"ds": self.dates,
                                   "y": 420 + 30 * np.sin(np.arange(len(self.dates)) / 20)})
        self.config = GRUConfig(horizon=6, lookback_hours=24, hidden_size=4,
                                epochs=2, patience=2, input_group="speed", threads=1)
        self.train = self.dates[24:168:6]
        self.validation = self.dates[180:204:6]

    def test_gru_encoder_starts_at_persistence(self):
        network = DirectGRU(2, self.config)
        self.assertIsInstance(network.encoder, torch.nn.GRU)
        torch.testing.assert_close(network(torch.randn(3, 24, 2)), torch.zeros(3, 6))

    def test_future_invariance_and_checkpoint_architecture(self):
        model = GRUForecaster(self.config).fit(self.frame, self.train, self.validation)
        changed = self.frame.copy()
        changed.loc[300:, "y"] = 1e8
        other = GRUForecaster(self.config).fit(changed, self.train, self.validation)
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
        origin = self.dates[400]
        predicted = model.predict(self.frame, origin)
        changed = self.frame.copy()
        changed.loc[changed.ds >= origin, "y"] = 1e8
        np.testing.assert_array_equal(predicted.speed_km_s, model.predict(changed, origin).speed_km_s)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru.pt"
            model.save(path)
            restored = GRUForecaster.load(path)
            self.assertIsInstance(restored.model.encoder, torch.nn.GRU)
            np.testing.assert_array_equal(predicted.speed_km_s, restored.predict(self.frame, origin).speed_km_s)
            with self.assertRaisesRegex(ValueError, "architecture"):
                LSTMForecaster.load(path)

    def test_legacy_lstm_checkpoint_remains_compatible(self):
        model = LSTMForecaster(LSTMConfig(**asdict(self.config))).fit(self.frame, self.train, self.validation)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstm.pt"
            model.save(path)
            payload = torch.load(path, weights_only=True)
            payload.pop("architecture")
            torch.save(payload, path)
            self.assertIsInstance(LSTMForecaster.load(path).model.encoder, torch.nn.LSTM)
            with self.assertRaisesRegex(ValueError, "architecture"):
                GRUForecaster.load(path)

    def test_cli_comparison_and_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "speed.csv"
            self.frame.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_csv(source, index=False)
            reference = LSTMForecaster(LSTMConfig(**asdict(self.config))).fit(self.frame, self.train, self.validation)
            reference.save(root / "lstm.pt")
            output = root / "run"
            self.assertEqual(main([
                "experiment", "--data", str(source), "--input-group", "speed",
                "--train-start", "2024-01-02", "--validation-start", "2024-01-09",
                "--validation-hours", "24", "--test-start", "2024-02-02", "--test-hours", "12",
                "--horizon", "6", "--lookback-hours", "24", "--hidden-size", "4",
                "--epochs", "2", "--threads", "1", "--lstm-model", str(root / "lstm.pt"),
                "--output-dir", str(output)]), 0)
            report = json.loads((output / "test/report.json").read_text())
            self.assertEqual(set(report["metrics"]), {"gru", "lstm", "persistence", "recurrence"})
            self.assertEqual(report["metrics"]["gru"]["prediction_coverage"], 1)
            destination = root / "forecast"
            self.assertEqual(main(["forecast", "--data", str(source), "--model", str(output / "model.pt"),
                                   "--origin", "2024-02-03", "--output-dir", str(destination)]), 0)
            self.assertEqual(len(pd.read_csv(destination / "forecast.csv")), 6)
            self.assertEqual(json.loads((destination / "report.json").read_text())["model"], "gru")


if __name__ == "__main__":
    unittest.main()
