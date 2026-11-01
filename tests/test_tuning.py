from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchlab_forecasting.features import lagged_features, load_regressors, prepare_regressors
from benchlab_forecasting.models import ForecastConfig, ProphetForecaster, future_grid
from benchlab_forecasting.tuning import check_schedule, run_tuning


class ExogenousTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=24 * 80, freq="h", tz="UTC")
        x = np.sin(np.arange(len(self.dates)) / 20)
        self.raw = pd.DataFrame({"ds": self.dates, "density_cm3": 8 + x})
        self.series = pd.DataFrame({"ds": self.dates, "y": 400 + 20 * x})
        self.origin = self.dates[-100]
        self.config = ForecastConfig(rotation_days=27, fourier_order=3)

    def test_lag_alignment_and_future_invariance(self):
        dates = future_grid(self.origin, 72)
        actual = lagged_features(self.raw, dates, self.origin, self.config)
        expected = self.raw.set_index("ds").density_cm3.reindex(dates - pd.Timedelta(hours=96))
        np.testing.assert_allclose(actual.iloc[:, 0], expected)
        changed = self.raw.copy()
        changed.loc[changed.ds >= self.origin, "density_cm3"] = 1e9
        pd.testing.assert_frame_equal(actual, lagged_features(changed, dates, self.origin, self.config))
        with self.assertRaisesRegex(ValueError, "lag must cover"):
            lagged_features(self.raw, dates, self.origin, replace(self.config, exogenous_lag_hours=72))

    def test_late_publication_and_bounded_fill(self):
        raw = self.raw.copy()
        raw["density_cm3__available_at"] = raw.ds + pd.Timedelta(hours=1)
        dates = future_grid(self.origin, 72)
        lookup = dates[0] - pd.Timedelta(hours=96)
        raw.loc[raw.ds >= lookup, "density_cm3__available_at"] = self.origin + pd.Timedelta(days=5)
        features = lagged_features(raw, dates, self.origin, self.config)
        self.assertTrue(features.iloc[:6, 0].notna().all())
        self.assertTrue(features.iloc[6:, 0].isna().all())

    def test_imputation_uses_training_only(self):
        history = self.dates[100:300]
        future = self.dates[500:572]
        raw = self.raw.copy()
        raw.loc[raw.ds.between(future.min() - pd.Timedelta(hours=110), future.max()), "density_cm3"] = np.nan
        train, forecast, metadata = prepare_regressors(raw, history, future, self.origin, self.config)
        median = next(iter(metadata["imputation_medians"].values()))
        self.assertAlmostEqual(median, lagged_features(raw, history, self.origin, self.config).iloc[:, 0].median())
        np.testing.assert_allclose(forecast.iloc[:, 0], median)
        self.assertTrue(np.isfinite(train.to_numpy()).all())

    def test_quality_flags_and_negative_magnetic_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"timestamp_utc": self.dates[:3], "status": [0, 1, 0],
                          "density_cm3": [3, 4, -9999], "temperature_k": [100, 200, 300]}).to_csv(root / "plasma.csv", index=False)
            pd.DataFrame({"timestamp_utc": self.dates[:3], "status": [0, 0, 0],
                          "bx_gsm_nt": [-2, -999.9, 1], "by_gsm_nt": [1, 1, 1],
                          "bz_gsm_nt": [-3, 0, 1], "bt_nt": [4, 1, 2]}).to_csv(root / "mag.csv", index=False)
            data = load_regressors(root / "plasma.csv", root / "mag.csv", "plasma_mag")
            self.assertEqual(data.bx_gsm_nt.iloc[0], -2)
            self.assertTrue(pd.isna(data.bx_gsm_nt.iloc[1]))
            self.assertTrue(data.density_cm3.iloc[1:].isna().all())

    def test_real_prophet_exogenous_serialization_and_no_leakage(self):
        model = ProphetForecaster(self.config).fit(self.series, self.origin, self.raw)
        expected = model.predict()
        changed_y, changed_x = self.series.copy(), self.raw.copy()
        changed_y.loc[changed_y.ds >= self.origin, "y"] = 1e6
        changed_x.loc[changed_x.ds >= self.origin, "density_cm3"] = 1e6
        other = ProphetForecaster(self.config).fit(changed_y, self.origin, changed_x).predict()
        np.testing.assert_allclose(expected.speed_km_s, other.speed_km_s)
        self.assertTrue(model.model.extra_regressors)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            np.testing.assert_allclose(ProphetForecaster.load(path).predict().speed_km_s,
                                       expected.speed_km_s)

    def test_holdout_overlap_rejected(self):
        folds = [pd.date_range("2024-02-01", periods=4, freq="6h", tz="UTC")]
        test = pd.date_range("2024-02-03", periods=4, freq="6h", tz="UTC")
        with self.assertRaisesRegex(ValueError, "Validation targets"):
            check_schedule(self.series, folds, test, 72)

    def test_selection_ignores_holdout_and_failed_candidates(self):
        config = replace(self.config, horizon=2)
        candidates = [{"name": name, "group": "none", "config": config}
                      for name in ("prophet_current", "validation_winner", "failed")]
        folds = [pd.DatetimeIndex([self.dates[900]])]
        test = pd.DatetimeIndex([self.dates[1200]])

        def mocked_predict(series, inputs, origins, candidate):
            dates = future_grid(origins[0], 2)
            truth = series.set_index("ds").y.reindex(dates).to_numpy()
            name = candidate["name"]
            if name == "failed":
                return [np.nan] * 2, [{"error": "failed"}], [0.1], None
            # This candidate wins validation but loses the final test badly.
            error = (0 if origins[0] == folds[0][0] else 500) if name == "validation_winner" else 10
            return truth + error, [], [0.1], None

        with tempfile.TemporaryDirectory() as tmp, patch(
                "benchlab_forecasting.tuning.predict_candidate", side_effect=mocked_predict):
            winner, _ = run_tuning(self.series, {"none": None}, candidates, folds, test, Path(tmp))
            self.assertEqual(winner["name"], "validation_winner")
            saved = json.loads((Path(tmp) / "best_config.json").read_text())
            self.assertEqual(saved["name"], "validation_winner")
            leaderboard = pd.read_csv(Path(tmp) / "leaderboard.csv")
            self.assertFalse(leaderboard.set_index("candidate").loc["failed", "eligible"])


if __name__ == "__main__":
    unittest.main()
