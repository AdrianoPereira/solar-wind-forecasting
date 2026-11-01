import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchlab_forecasting.validation import (
    evaluate_predictions, official_timse, regression_metrics, temporal_splits,
    timse_proxy, validate_predictions,
)
from benchlab_forecasting.validation.__main__ import main


def predictions():
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame({"origin_utc": [origin] * 3, "horizon": [1, 2, 3],
                         "timestamp_utc": pd.date_range(origin + pd.Timedelta(hours=1), periods=3, freq="h"),
                         "y": [300., 400., 800.], "a": [310., 400., 780.],
                         "b": [300., 410., 800.]})


class MetricTests(unittest.TestCase):
    def test_metrics_hand_calculation(self):
        metric = regression_metrics([1, 2, 3, np.nan], [2, 0, 3, 4])
        self.assertEqual(metric["n"], 3)
        self.assertAlmostEqual(metric["mse"], 5 / 3)
        self.assertAlmostEqual(metric["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(metric["mae"], 1)
        self.assertAlmostEqual(metric["bias"], -1 / 3)

    def test_proxy_hand_calculation_and_tail_emphasis(self):
        # Tail MSEs: 5/3, 2, 4; average = 23/9.
        self.assertAlmostEqual(timse_proxy([1, 2, 3], [2, 2, 5]), 23 / 9)
        self.assertGreater(timse_proxy([1, 2, 3], [1, 2, 4]),
                           timse_proxy([1, 2, 3], [2, 2, 3]))
        self.assertEqual(timse_proxy([1, 2], [1, 2]), 0)

    def test_proxy_ties_permutation_and_missing(self):
        y, p = np.array([1., 1., 3.]), np.array([2., 3., 4.])
        # Repeated threshold 1 has MSE 2; threshold 3 has MSE 1.
        self.assertAlmostEqual(timse_proxy(y, p), 5 / 3)
        self.assertEqual(timse_proxy(y, p), timse_proxy(y[::-1], p[::-1]))
        self.assertIsNone(timse_proxy([1, 2], [1, np.nan]))
        self.assertEqual(timse_proxy([1, np.nan], [2, np.nan]), 1)
        self.assertIsNone(timse_proxy([], []))
        with self.assertRaises(ValueError):
            timse_proxy([1], [1, 2])
        with self.assertRaises(NotImplementedError):
            official_timse(y, p)


class ReportingTests(unittest.TestCase):
    def test_common_pairs_and_no_official_claim(self):
        frame = predictions()
        frame.loc[2, "b"] = np.nan
        report = evaluate_predictions(frame, ["a", "b"], horizon=3)
        self.assertEqual(report["common_pairs"], 2)
        self.assertEqual(report["metrics"]["a"]["available"]["n"], 3)
        self.assertEqual(report["metrics"]["a"]["common"]["n"], 2)
        self.assertIsNone(report["metrics"]["b"]["timse_proxy"])
        self.assertIsNone(report["challenge"]["official_score"])
        json.dumps(report, allow_nan=False)

    def test_missing_rows_and_entire_origin_are_counted(self):
        frame = predictions().iloc[:2]
        expected = pd.date_range("2024-01-01", periods=2, freq="6h", tz="UTC")
        report = evaluate_predictions(frame, ["a"], horizon=3, expected_origins=expected)
        self.assertEqual(report["origins"], 2)
        self.assertEqual(report["forecast_pairs"], 6)
        self.assertEqual(report["metrics"]["a"]["complete_origins"], 0)
        self.assertAlmostEqual(report["metrics"]["a"]["prediction_coverage"], 1 / 3)

    def test_bad_alignment_duplicates_and_values_rejected(self):
        for key, value in (("horizon", 2.5), ("a", np.inf), ("y", -1),
                           ("timestamp_utc", pd.Timestamp("2024-01-02", tz="UTC"))):
            frame = predictions()
            frame[key] = frame[key].astype(object)
            frame.loc[0, key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_predictions(frame, ["a"], horizon=3)
        with self.assertRaises(ValueError):
            validate_predictions(pd.concat([predictions(), predictions()]), ["a"], horizon=3)

    def test_overlapping_targets_must_agree(self):
        first = predictions()
        second = first.iloc[1:2].copy()
        second["origin_utc"] += pd.Timedelta(hours=1)
        second["horizon"] = 1
        second["y"] = 999
        with self.assertRaisesRegex(ValueError, "conflitantes"):
            validate_predictions(pd.concat([first, second]), ["a"], horizon=3)

    def test_cli_no_model_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "predictions.csv", Path(directory) / "report.json"
            predictions().to_csv(path, index=False)
            self.assertEqual(main(["--predictions", str(path), "--models", "a", "b",
                                   "--horizon", "3", "--output", str(output)]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["metrics"]["a"]["available"]["n"], 3)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.origins = pd.date_range("2024-01-01", periods=100, freq="6h", tz="UTC")

    def test_expanding_purges_overlapping_labels_and_gap(self):
        folds = temporal_splits(self.origins, self.origins[50], n_splits=3,
                                validation_hours=24, gap_hours=6)
        self.assertLess(len(folds[0].train_indices), len(folds[1].train_indices))
        for fold in folds:
            label_end = self.origins[fold.train_indices] + pd.Timedelta(hours=78)
            self.assertTrue((label_end < fold.validation_start).all())
            self.assertFalse(set(fold.train_indices) & set(fold.validation_indices))
            self.assertEqual(len(fold.validation_indices), 4)

    def test_holdout_and_rolling(self):
        self.assertEqual(len(temporal_splits(self.origins, self.origins[50], strategy="holdout")), 1)
        folds = temporal_splits(self.origins, self.origins[50], strategy="rolling",
                                train_hours=168, validation_hours=24, n_splits=2)
        self.assertEqual(len(folds[0].train_indices), len(folds[1].train_indices))
        for fold in folds:
            self.assertTrue((self.origins[fold.train_indices] >= fold.validation_start - pd.Timedelta(hours=168)).all())

    def test_invalid_splits(self):
        for kwargs in ({"strategy": "random"}, {"strategy": "rolling"},
                       {"step_hours": 1}, {"gap_hours": -1}, {"horizon": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                temporal_splits(self.origins, self.origins[50], **kwargs)
        with self.assertRaises(ValueError):
            temporal_splits(self.origins[::-1], self.origins[50])


if __name__ == "__main__":
    unittest.main()
