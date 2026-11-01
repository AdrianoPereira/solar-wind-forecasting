import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from benchlab_forecasting.attention_gru import AttentionGRUConfig
    from benchlab_forecasting.attention_gru_tuning import (
        check_schedule, main, read_candidates, read_search_space, run_search,
    )
    from benchlab_forecasting.lstm_cli import origins


@unittest.skipUnless(HAS_TORCH, "Install the lstm extra for recurrent models")
class AttentionGRUTuningTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
        self.frame = pd.DataFrame({
            "ds": dates, "y": 420 + 30 * np.sin(np.arange(len(dates)) / 20),
        })
        self.base = AttentionGRUConfig(
            horizon=6, lookback_hours=24, hidden_size=4, attention_heads=2,
            epochs=2, patience=2, input_group="speed", threads=1,
        )
        self.train = origins("2024-01-02", 24 * 7 - 6, 6)
        self.early = origins("2024-01-09", 24, 6)
        self.folds = [origins("2024-01-16", 24, 6), origins("2024-01-20", 24, 6)]
        self.test = origins("2024-02-02", 12, 6)
        self.candidates = [
            {"name": "plain", "config": self.base},
            {"name": "residual", "config": AttentionGRUConfig(
                **{**self.base.__dict__, "residual_scale": 1.0}
            )},
        ]

    def test_rejects_overlapping_target_intervals_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "Target intervals overlap"):
            check_schedule(self.frame, self.candidates, self.train, self.early,
                           [origins("2024-01-09", 24, 6)], self.test)
        duplicate = [self.candidates[0], self.candidates[0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            check_schedule(self.frame, duplicate, self.train, self.early,
                           self.folds, self.test)

    def test_candidate_file_validates_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(json.dumps([{"name": "small", "config": {
                "hidden_size": 8, "attention_heads": 2, "input_group": "speed",
            }}]))
            candidates = read_candidates(path, self.base)
            self.assertEqual(candidates[0]["config"].hidden_size, 8)
            path.write_text(json.dumps([{"name": "bad", "config": {
                "hidden_size": 5, "attention_heads": 2,
            }}]))
            with self.assertRaisesRegex(ValueError, "divisible"):
                read_candidates(path, self.base)

    def test_search_space_expands_bounded_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "space.json"
            path.write_text(json.dumps({"hidden_size": [4, 8],
                                        "residual_scale": [0.0, 1.0]}))
            trials = read_search_space(path, self.base)
            self.assertEqual([trial["name"] for trial in trials],
                             ["trial_001", "trial_002", "trial_003", "trial_004"])
            self.assertEqual([(c["config"].hidden_size, c["config"].residual_scale)
                              for c in trials], [(4, 0.0), (4, 1.0), (8, 0.0), (8, 1.0)])
            path.write_text(json.dumps({"hidden_size": list(range(4, 37))}))
            with self.assertRaisesRegex(ValueError, "32"):
                read_search_space(path, self.base)

    def test_selection_is_independent_of_final_test_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            run_search(self.frame, self.candidates, self.train, self.early,
                       self.folds, self.test, first, metric="timse")
            changed = self.frame.copy()
            changed.loc[changed.ds >= self.test.min() + pd.Timedelta(hours=1), "y"] += 200
            run_search(changed, self.candidates, self.train, self.early,
                       self.folds, self.test, second, metric="timse")
            self.assertEqual(json.loads((first / "best_config.json").read_text()),
                             json.loads((second / "best_config.json").read_text()))
            pd.testing.assert_frame_equal(
                pd.read_parquet(first / "validation_predictions.parquet"),
                pd.read_parquet(second / "validation_predictions.parquet"),
            )
            self.assertNotEqual(
                json.loads((first / "test/report.json").read_text())["metrics"]["attention_gru"]["common"]["rmse"],
                json.loads((second / "test/report.json").read_text())["metrics"]["attention_gru"]["common"]["rmse"],
            )

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "speed.csv"
            self.frame.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_csv(
                source, index=False,
            )
            config = root / "search_space.json"
            config.write_text(json.dumps({
                "lookback_hours": [24], "hidden_size": [4],
                "attention_heads": [2], "input_group": ["speed"],
            }))
            output = root / "search"
            self.assertEqual(main([
                "--data", str(source), "--output-dir", str(output),
                "--train-start", "2024-01-02", "--early-stop-start", "2024-01-09",
                "--early-stop-hours", "24", "--validation-starts", "2024-01-16",
                "--validation-hours", "24", "--test-start", "2024-02-02",
                "--test-hours", "12", "--step-hours", "6", "--horizon", "6",
                "--epochs", "2", "--threads", "1", "--metric", "rmse",
                "--search-space", str(config),
            ]), 0)
            self.assertEqual(json.loads((output / "best_config.json").read_text())["name"], "trial_001")
            self.assertTrue((output / "candidates/trial_001/model.pt").exists())


if __name__ == "__main__":
    unittest.main()
