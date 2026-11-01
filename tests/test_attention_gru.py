from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from benchlab_forecasting.attention_gru import (
        AttentionGRUConfig, AttentionGRUForecaster, DirectAttentionGRU,
    )
    from benchlab_forecasting.attention_gru_cli import main
    from benchlab_forecasting.gru import GRUConfig, GRUForecaster
    from benchlab_forecasting.lstm import LSTMConfig, LSTMForecaster


@unittest.skipUnless(HAS_TORCH, "Install the lstm extra for recurrent models")
class AttentionGRUTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
        self.frame = pd.DataFrame({
            "ds": self.dates,
            "y": 420 + 30 * np.sin(np.arange(len(self.dates)) / 20),
        })
        self.config = AttentionGRUConfig(
            horizon=6, lookback_hours=24, hidden_size=4, attention_heads=2,
            epochs=2, patience=2, input_group="speed", threads=1,
        )
        self.train = self.dates[24:168:6]
        self.validation = self.dates[180:204:6]

    def test_attention_queries_produce_one_correction_per_horizon(self):
        network = DirectAttentionGRU(2, self.config)
        self.assertIsInstance(network.attention, torch.nn.MultiheadAttention)
        inputs = torch.randn(3, 24, 2)
        torch.testing.assert_close(network(inputs), torch.zeros(3, 6))
        with torch.no_grad():
            network.head[-1].weight.fill_(0.1)
        self.assertEqual(tuple(network(inputs).shape), (3, 6))
        changed = inputs.clone()
        changed[:, 0, :] += 10
        self.assertFalse(torch.equal(network(inputs), network(changed)))

    def test_invalid_head_count(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            AttentionGRUConfig(hidden_size=5, attention_heads=2)
        with self.assertRaisesRegex(ValueError, "residual_scale"):
            AttentionGRUConfig(residual_scale=-1)
        with self.assertRaisesRegex(ValueError, "device"):
            AttentionGRUConfig(device="gpu")

    def test_cuda_request_fails_clearly_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                AttentionGRUForecaster(replace(self.config, device="cuda")).fit(
                    self.frame, self.train, self.validation,
                )

    def test_residual_adds_last_gru_state(self):
        direct = DirectAttentionGRU(2, self.config)
        residual = DirectAttentionGRU(2, AttentionGRUConfig(
            **{**asdict(self.config), "residual_scale": 1.0}
        ))
        residual.load_state_dict(direct.state_dict())
        with torch.no_grad():
            direct.head[-1].weight.fill_(0.1)
            residual.head[-1].weight.fill_(0.1)
        direct.eval()
        residual.eval()
        inputs = torch.randn(3, 24, 2)
        self.assertFalse(torch.equal(direct(inputs), residual(inputs)))

    def test_future_invariance_and_checkpoint(self):
        model = AttentionGRUForecaster(self.config).fit(self.frame, self.train, self.validation)
        changed = self.frame.copy()
        changed.loc[300:, "y"] = 1e8
        other = AttentionGRUForecaster(self.config).fit(changed, self.train, self.validation)
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
        origin = self.dates[400]
        prediction = model.predict(self.frame, origin)
        changed = self.frame.copy()
        changed.loc[changed.ds >= origin, "y"] = 1e8
        np.testing.assert_array_equal(
            prediction.speed_km_s, model.predict(changed, origin).speed_km_s,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention_gru.pt"
            model.save(path)
            restored = AttentionGRUForecaster.load(path)
            np.testing.assert_array_equal(
                prediction.speed_km_s, restored.predict(self.frame, origin).speed_km_s,
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                GRUForecaster.load(path)

    def test_cli_compares_frozen_references_and_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "speed.csv"
            self.frame.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_csv(
                source, index=False,
            )
            common = {k: v for k, v in asdict(self.config).items()
                      if k not in ("attention_heads", "residual_scale")}
            GRUForecaster(GRUConfig(**common)).fit(
                self.frame, self.train, self.validation,
            ).save(root / "gru.pt")
            LSTMForecaster(LSTMConfig(**common)).fit(
                self.frame, self.train, self.validation,
            ).save(root / "lstm.pt")
            output = root / "run"
            self.assertEqual(main([
                "experiment", "--data", str(source), "--input-group", "speed",
                "--train-start", "2024-01-02", "--validation-start", "2024-01-09",
                "--validation-hours", "24", "--test-start", "2024-02-02",
                "--test-hours", "12", "--horizon", "6", "--lookback-hours", "24",
                "--hidden-size", "4", "--attention-heads", "2", "--epochs", "2",
                "--threads", "1", "--gru-model", str(root / "gru.pt"),
                "--lstm-model", str(root / "lstm.pt"), "--output-dir", str(output),
            ]), 0)
            report = json.loads((output / "test/report.json").read_text())
            self.assertEqual(set(report["metrics"]), {
                "attention_gru", "gru", "lstm", "persistence", "recurrence",
            })
            self.assertEqual(report["metrics"]["attention_gru"]["prediction_coverage"], 1)
            destination = root / "forecast"
            self.assertEqual(main([
                "forecast", "--data", str(source), "--model", str(output / "model.pt"),
                "--origin", "2024-02-03", "--output-dir", str(destination),
            ]), 0)
            self.assertEqual(len(pd.read_csv(destination / "forecast.csv")), 6)


if __name__ == "__main__":
    unittest.main()
