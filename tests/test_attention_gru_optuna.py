import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


HAS_HPO = (importlib.util.find_spec("torch") is not None
           and importlib.util.find_spec("optuna") is not None)
if HAS_HPO:
    from benchlab_forecasting.attention_gru import AttentionGRUConfig
    from benchlab_forecasting.attention_gru_optuna import read_space, run_study
    from benchlab_forecasting.lstm_cli import origins


def _study_worker(args, worker_id):
    run_study(*args, target_trials=2, resume=True, worker_id=worker_id)


@unittest.skipUnless(HAS_HPO, "Install the lstm and hpo extras")
class AttentionGRUOptunaTests(unittest.TestCase):
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
        self.space = {"hidden_size": [4], "attention_heads": [2],
                      "lookback_hours": [24], "input_group": ["speed"],
                      "residual_scale": [0.0, 1.0]}

    def test_resume_and_finalize_reserved_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"
            args = (self.frame, self.base, self.space, self.train,
                    self.early, self.folds, self.test, output)
            run_study(*args, target_trials=2)
            self.assertEqual(len(pd.read_csv(output / "leaderboard.csv")), 2)
            self.assertFalse((output / "test").exists())
            best = run_study(*args, target_trials=3, resume=True, finalize=True)
            self.assertEqual(len(pd.read_csv(output / "leaderboard.csv")), 3)
            self.assertTrue((output / "trials" / f"{best['trial']:04d}" / "model.pt").exists())
            self.assertTrue((output / "test" / "predictions.parquet").exists())
            self.assertEqual(json.loads((output / "test" / "report.json").read_text())[
                "selected_trial"], best["trial"])
            with self.assertRaisesRegex(ValueError, "changed"):
                run_study(self.frame, self.base, {"hidden_size": [8]},
                          self.train, self.early, self.folds, self.test,
                          output, target_trials=4, resume=True)

    def test_rejects_invalid_search_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "space.json"
            path.write_text(json.dumps({"input_group": {"low": 1, "high": 2}}))
            with self.assertRaisesRegex(ValueError, "explicit categorical"):
                read_space(path)

    def test_two_processes_share_trials_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"
            args = (self.frame, self.base, self.space, self.train,
                    self.early, self.folds, self.test, output)
            run_study(*args, target_trials=2, prepare_only=True)
            with ProcessPoolExecutor(max_workers=2,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(_study_worker, args, index) for index in (0, 1)]
                for future in futures:
                    future.result()
            best = run_study(*args, target_trials=2, resume=True, finalize=True)
            board = pd.read_csv(output / "leaderboard.csv")
            self.assertEqual(board.trial.nunique(), 2)
            self.assertTrue((board.state == "COMPLETE").all())
            self.assertTrue((output / "test" / "predictions.parquet").is_file())
            self.assertIn(best["trial"], board.trial.to_list())


if __name__ == "__main__":
    unittest.main()
