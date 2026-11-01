import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchlab_forecasting.validation import evaluate_predictions, timse, timse_profile, timse_weights
from benchlab_forecasting.validation.timse_report import cohort_tables, main


def sample():
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame({"origin_utc": [origin] * 3, "horizon": [1, 2, 3],
                         "timestamp_utc": pd.date_range(origin + pd.Timedelta(hours=1), periods=3, freq="h"),
                         "y": [300., 400., 800.], "a": [310., 400., 780.],
                         "b": [300., 410., np.nan]})


class TimseTests(unittest.TestCase):
    def test_irregular_thresholds_ties_and_last_endpoint(self):
        # Unique thresholds 1, 3, 6; MSE tails 7.5, 12.5, 16.
        # Left rectangles: 2*7.5 + 3*12.5 = 52.5 (no final term).
        y, p = [1, 1, 3, 6], [2, 3, 6, 10]
        profile = timse_profile(y, p)
        self.assertEqual(profile["timse"], 52.5)
        np.testing.assert_array_equal(profile["tail_n"], [4, 2, 1])
        np.testing.assert_array_equal(profile["width"], [2, 3, 0])
        np.testing.assert_array_equal(profile["contribution"], [15, 37.5, 0])
        self.assertEqual(timse(y[::-1], p[::-1]), 52.5)
        self.assertEqual(timse([10*x for x in y], [10*x for x in p]), 52500)

    def test_against_direct_formula(self):
        rng = np.random.default_rng(42)
        y = rng.integers(250, 850, size=250).astype(float)
        p = y + rng.normal(0, 70, size=len(y))
        u = np.unique(y)
        expected = sum((hi-lo)*np.mean((p[y >= lo]-y[y >= lo])**2) for lo, hi in zip(u[:-1], u[1:]))
        self.assertAlmostEqual(timse(y, p), expected, places=6)
        self.assertAlmostEqual(timse_profile(y, p)["cumulative_timse"][-1], expected, places=6)

    def test_perfect_degenerate_missing_and_shape(self):
        self.assertEqual(timse([300, 800], [300, 800]), 0)
        self.assertFalse(timse_profile([300, 800], [300, 800])["degenerate"])
        self.assertTrue(timse_profile([300, 300], [900, 900])["degenerate"])
        self.assertEqual(timse([300], [900]), 0)
        self.assertIsNone(timse([], []))
        self.assertIsNone(timse([300, 800], [300, np.nan]))
        self.assertIsNone(timse([300, 800], [300, np.inf]))
        self.assertEqual(timse([300, np.nan, 800], [310, np.nan, 780]), 125000)
        with self.assertRaises(ValueError):
            timse([1], [1, 2])

    def test_weights_reproduce_timse_as_weighted_squared_error(self):
        # Slide example: every pair is in the first tail (2/4); y >= 3 adds 3/2; the last width is 0.
        # 0.5*1 + 0.5*4 + 2*9 + 2*16 = 52.5, the TIMSE of this example.
        np.testing.assert_array_equal(timse_weights([1, 1, 3, 6]), [0.5, 0.5, 2, 2])
        rng = np.random.default_rng(7)
        y = rng.integers(250, 850, size=400).astype(float)
        p = y + rng.normal(0, 70, size=len(y))
        weights = timse_weights(y)
        self.assertAlmostEqual(float(np.sum(weights * (p - y) ** 2)), timse(y, p), places=6)
        self.assertTrue(np.all(np.diff(weights[np.argsort(y)]) >= 0))
        np.testing.assert_array_equal(timse_weights([np.nan, 3, 1]), [0, 1, 1])

    def test_high_speed_errors_receive_more_weight(self):
        self.assertGreater(timse([300, 400, 800], [300, 400, 810]),
                           timse([300, 400, 800], [310, 400, 800]))

    def test_shared_pairs_strict_score_and_calendar_coverage(self):
        frame = sample()
        report = evaluate_predictions(frame, ["a", "b"], horizon=3)
        self.assertIsNone(report["metrics"]["b"]["timse"])
        self.assertEqual(report["metrics"]["a"]["timse_common"], 5000)
        self.assertEqual(report["metrics"]["b"]["timse_common"], 5000)
        summary, profiles, groups, common = cohort_tables(frame, ["a", "b"])
        self.assertEqual(len(common), 2)
        self.assertEqual(len(profiles), 4)
        self.assertTrue(summary.timse.eq(5000).all())
        self.assertTrue(groups.loc[groups.kind.eq("horizon") & groups.group.eq("3"), "timse"].isna().all())

    def test_cli_creates_figures_cohorts_and_self_contained_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample().to_csv(root / "input.csv", index=False)
            out = root / "validation"
            args = ["--predictions", str(root / "input.csv"), "--models", "a", "b", "--horizon", "3",
                    "--start", "2024-01-01", "--end", "2024-01-01", "--output-dir", str(out),
                    "--formats", "png", "--dpi", "40"]
            self.assertEqual(main(args), 0)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["cohorts"]["all_models"]["common_pairs"], 2)
            self.assertEqual(report["cohorts"]["complete_models"]["common_pairs"], 3)
            self.assertEqual(len(list(out.glob("*.png"))), 10)
            self.assertIn("data:image/png;base64,", (out / "report.html").read_text())
            self.assertIn("TIMSE validation", (out / "report.html").read_text())
            self.assertEqual(report["calendar"], "explicit")
            with self.assertRaises(SystemExit):
                main(args)  # Existing artifacts must not be silently overwritten.

    def test_cli_no_common_pairs_writes_diagnostics_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = sample()
            frame["b"] = np.nan
            frame.to_csv(root / "input.csv", index=False)
            self.assertEqual(main(["--predictions", str(root / "input.csv"), "--models", "b", "--horizon", "3",
                                   "--output-dir", str(root / "out"), "--formats", "png", "--dpi", "40"]), 1)
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertIsNone(report["cohorts"]["all_models"]["scores"][0]["timse"])


if __name__ == "__main__":
    unittest.main()
