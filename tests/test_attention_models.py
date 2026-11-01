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
    from benchlab_forecasting import attention_lstm_cli, patch_transformer_cli, transformer_cli
    from benchlab_forecasting.attention_gru import AttentionGRUConfig, DirectAttentionGRU
    from benchlab_forecasting.attention_lstm import (
        AttentionLSTMConfig, AttentionLSTMForecaster, DirectAttentionLSTM,
    )
    from benchlab_forecasting.gru import GRUConfig, GRUForecaster
    from benchlab_forecasting.transformer import (
        DirectTransformer, PatchTransformerConfig, PatchTransformerForecaster,
        TransformerConfig, TransformerForecaster,
    )


SMALL = dict(horizon=6, lookback_hours=24, hidden_size=4, attention_heads=2,
             epochs=2, patience=2, input_group="speed", threads=1)


def cases():
    return [
        ("attention_lstm", AttentionLSTMForecaster, AttentionLSTMConfig(**SMALL),
         attention_lstm_cli.main, []),
        ("transformer", TransformerForecaster,
         TransformerConfig(**SMALL, layers=1, feedforward_size=8),
         transformer_cli.main, ["--layers", "1", "--feedforward-size", "8"]),
        ("patch_transformer", PatchTransformerForecaster,
         PatchTransformerConfig(**SMALL, layers=1, feedforward_size=8, patch_hours=6),
         patch_transformer_cli.main,
         ["--layers", "1", "--feedforward-size", "8", "--patch-hours", "6"]),
    ]


@unittest.skipUnless(HAS_TORCH, "Install the lstm extra for recurrent models")
class AttentionModelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
        self.frame = pd.DataFrame({
            "ds": self.dates,
            "y": 420 + 30 * np.sin(np.arange(len(self.dates)) / 20),
        })
        self.train = self.dates[24:168:6]
        self.validation = self.dates[180:204:6]

    def test_networks_start_at_persistence_and_see_whole_history(self):
        for name, forecaster, config, _, _ in cases():
            with self.subTest(name):
                network = forecaster.network_class(2, config)
                network.eval()
                inputs = torch.randn(3, 24, 2)
                torch.testing.assert_close(network(inputs), torch.zeros(3, 6))
                with torch.no_grad():
                    network.head[-1].weight.fill_(0.1)
                changed = inputs.clone()
                changed[:, 0, :] += 10
                self.assertEqual(tuple(network(inputs).shape), (3, 6))
                self.assertFalse(torch.equal(network(inputs), network(changed)))

    def test_patch_tokens_end_at_origin(self):
        config = PatchTransformerConfig(**SMALL, layers=1, feedforward_size=8, patch_hours=6)
        network = DirectTransformer(2, config)
        self.assertEqual(network.positions.num_embeddings, 4)
        self.assertEqual(network.embedding.in_features, 12)
        defaults = PatchTransformerConfig()
        self.assertEqual((defaults.lookback_hours, defaults.patch_hours), (720, 24))
        # The 27-day recurrence of every forecast hour lies inside the default history.
        self.assertGreaterEqual(defaults.lookback_hours, 27 * 24)

    def test_invalid_configs(self):
        with self.assertRaisesRegex(ValueError, "divisible by patch_hours"):
            TransformerConfig(lookback_hours=30, patch_hours=24)
        with self.assertRaisesRegex(ValueError, "layers"):
            TransformerConfig(layers=0)
        with self.assertRaisesRegex(ValueError, "divisible by attention_heads"):
            AttentionLSTMConfig(hidden_size=5, attention_heads=2)

    def test_attention_gru_weights_are_unchanged_by_encoder_hook(self):
        torch.manual_seed(0)
        gru = DirectAttentionGRU(2, AttentionGRUConfig(**SMALL))
        self.assertIsInstance(gru.encoder, torch.nn.GRU)
        torch.manual_seed(0)
        lstm = DirectAttentionLSTM(2, AttentionLSTMConfig(**SMALL))
        self.assertIsInstance(lstm.encoder, torch.nn.LSTM)
        self.assertEqual(set(gru.state_dict()) - {k for k in gru.state_dict() if k.startswith("encoder.")},
                         set(lstm.state_dict()) - {k for k in lstm.state_dict() if k.startswith("encoder.")})

    def test_future_invariance_and_checkpoint(self):
        for name, forecaster, config, _, _ in cases():
            with self.subTest(name):
                model = forecaster(config).fit(self.frame, self.train, self.validation)
                changed = self.frame.copy()
                changed.loc[300:, "y"] = 1e8
                other = forecaster(config).fit(changed, self.train, self.validation)
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
                    path = Path(tmp) / f"{name}.pt"
                    model.save(path)
                    restored = forecaster.load(path)
                    np.testing.assert_array_equal(
                        prediction.speed_km_s, restored.predict(self.frame, origin).speed_km_s,
                    )
                    with self.assertRaisesRegex(ValueError, "architecture"):
                        GRUForecaster.load(path)

    def test_cli_compares_reference_and_forecasts(self):
        for name, _, _, main, extra in cases():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "speed.csv"
                self.frame.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_csv(
                    source, index=False,
                )
                common = {k: v for k, v in SMALL.items() if k != "attention_heads"}
                GRUForecaster(GRUConfig(**common)).fit(
                    self.frame, self.train, self.validation,
                ).save(root / "gru.pt")
                output = root / "run"
                self.assertEqual(main([
                    "experiment", "--data", str(source), "--input-group", "speed",
                    "--train-start", "2024-01-02", "--validation-start", "2024-01-09",
                    "--validation-hours", "24", "--test-start", "2024-02-02",
                    "--test-hours", "12", "--horizon", "6", "--lookback-hours", "24",
                    "--hidden-size", "4", "--attention-heads", "2", "--epochs", "2",
                    "--threads", "1", "--gru-model", str(root / "gru.pt"),
                    "--output-dir", str(output), *extra,
                ]), 0)
                report = json.loads((output / "test/report.json").read_text())
                self.assertEqual(set(report["metrics"]), {name, "gru", "persistence", "recurrence"})
                self.assertEqual(report["metrics"][name]["prediction_coverage"], 1)
                destination = root / "forecast"
                self.assertEqual(main([
                    "forecast", "--data", str(source), "--model", str(output / "model.pt"),
                    "--origin", "2024-02-03", "--output-dir", str(destination),
                ]), 0)
                self.assertEqual(len(pd.read_csv(destination / "forecast.csv")), 6)


if __name__ == "__main__":
    unittest.main()
