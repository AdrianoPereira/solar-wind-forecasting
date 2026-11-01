import importlib.util
import unittest

import numpy as np
import pandas as pd

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_CATBOOST = importlib.util.find_spec("catboost") is not None
HAS_NGBOOST = importlib.util.find_spec("ngboost") is not None
from benchlab_forecasting.hybrid import (  # noqa: E402
    HybridConfig, HybridForecaster, baseline_predictions, build_pairs, coronal_hole_columns,
    feature_columns, origins_between,
)


def synthetic(days=120, seed=0):
    grid = pd.date_range("2020-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.arange(len(grid)) / (27 * 24)
    speed = 450 + 120 * np.sin(phase) + rng.normal(0, 10, len(grid))
    plasma = pd.DataFrame({"speed_km_s": speed, "density_cm3": 5.0, "temperature_k": 1e5}, index=grid)
    origins = grid[::6]
    holes = pd.DataFrame({"origin_utc": origins, "observation_utc": origins - pd.Timedelta(minutes=62)})
    for column in coronal_hole_columns():
        holes[column] = rng.uniform(0, 0.1, len(origins))
    return plasma, holes


class HybridTests(unittest.TestCase):
    def test_features_ignore_everything_after_the_cutoff(self):
        plasma, holes = synthetic()
        config = HybridConfig()
        origin = pd.Timestamp("2020-03-20T12:00", tz="UTC")
        pairs = build_pairs(plasma, holes, [origin], config)
        changed = plasma.copy()
        changed.loc[changed.index > origin - pd.Timedelta(hours=1)] = 9999.0
        late_holes = holes.copy()
        late_holes.loc[late_holes.origin_utc > origin, coronal_hole_columns()] = 1.0
        other = build_pairs(changed, late_holes, [origin], config)
        features = feature_columns(pairs)
        pd.testing.assert_frame_equal(pairs[features], other[features])
        self.assertTrue((other.y == 9999).all())
        self.assertEqual(len(pairs), 72)
        self.assertEqual(pairs.horizon.tolist(), list(range(1, 73)))

    def test_recurrence_is_the_smoothed_speed_one_rotation_before_each_target(self):
        plasma, holes = synthetic()
        origin = pd.Timestamp("2020-03-20T12:00", tz="UTC")
        pairs = build_pairs(plasma, holes, [origin], HybridConfig())
        smooth = plasma.speed_km_s.rolling(7, center=True, min_periods=3).mean()
        target = origin + pd.Timedelta(hours=10)
        self.assertAlmostEqual(pairs.recurrence_27d.iloc[9], smooth[target - pd.Timedelta(days=27)])
        self.assertAlmostEqual(pairs.last_speed.iloc[0], plasma.speed_km_s[origin - pd.Timedelta(hours=1)])
        lagged = holes.set_index("origin_utc").loc[origin - pd.Timedelta(hours=24), coronal_hole_columns()[0]]
        self.assertEqual(pairs[f"{coronal_hole_columns()[0]}_lag24h"].iloc[0], lagged)

    def test_baselines_fall_back_to_persistence(self):
        frame = pd.DataFrame({"last_speed": [400.0, 410.0], "recurrence_27d": [500.0, np.nan]})
        baselines = baseline_predictions(frame)
        np.testing.assert_array_equal(baselines["recurrence"], [500, 410])

    def check_fit_beats_persistence(self, model):
        plasma, holes = synthetic(days=200)
        config = HybridConfig(max_iter=60, min_samples_leaf=20, model=model, threads=2)
        train = build_pairs(plasma, holes, origins_between("2020-02-10", "2020-05-20"), config)
        validation = build_pairs(plasma, holes, origins_between("2020-05-25", "2020-06-10"), config)
        test = build_pairs(plasma, holes, origins_between("2020-06-15", "2020-07-10"), config)
        model = HybridForecaster(config).fit(train, validation, step=10)
        prediction = model.predict(test)
        self.assertTrue(np.isfinite(prediction).all())
        error = lambda p: np.sqrt(np.nanmean((p - test.y) ** 2))
        self.assertLess(error(prediction), error(baseline_predictions(test)["persistence"]))
        self.assertLessEqual(model.training["selected_iterations"], 60)
        self.assertEqual([row["iterations"] for row in model.training["history"]], [10, 20, 30, 40, 50, 60])
        return prediction

    @unittest.skipUnless(HAS_SKLEARN, "Install the hybrid extra")
    def test_fit_beats_persistence_on_recurrent_wind(self):
        self.check_fit_beats_persistence("hgb")

    @unittest.skipUnless(HAS_LIGHTGBM, "Install the boosting extra")
    def test_lightgbm_fit_is_deterministic(self):
        first = self.check_fit_beats_persistence("lightgbm")
        np.testing.assert_array_equal(first, self.check_fit_beats_persistence("lightgbm"))

    @unittest.skipUnless(HAS_XGBOOST, "Install the boosting extra")
    def test_xgboost_fit_is_deterministic(self):
        first = self.check_fit_beats_persistence("xgboost")
        np.testing.assert_array_equal(first, self.check_fit_beats_persistence("xgboost"))

    @unittest.skipUnless(HAS_CATBOOST, "Install the boosting extra")
    def test_catboost_fit_is_deterministic(self):
        first = self.check_fit_beats_persistence("catboost")
        np.testing.assert_array_equal(first, self.check_fit_beats_persistence("catboost"))

    @unittest.skipUnless(HAS_NGBOOST and HAS_LIGHTGBM, "Install the boosting extra")
    def test_ngboost_fit_is_deterministic_and_predicts_a_spread(self):
        first = self.check_fit_beats_persistence("ngboost")
        np.testing.assert_array_equal(first, self.check_fit_beats_persistence("ngboost"))

    @unittest.skipUnless(HAS_NGBOOST and HAS_LIGHTGBM, "Install the boosting extra")
    def test_ngboost_scale_tracks_the_noise(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"a": rng.uniform(size=4000), "b": rng.uniform(size=4000)})
        frame["y"] = 400 + 100 * frame.a + rng.normal(0, 10 + 60 * frame.b)
        config = HybridConfig(max_iter=300, min_samples_leaf=50, model="ngboost", threads=2, learning_rate=0.1)
        model = HybridForecaster(config)
        model.features = ["a", "b"]
        model.model = model._regressor(config.max_iter).fit(frame[["a", "b"]], frame.y)
        model.iterations = config.max_iter
        scale = model.predict_scale(frame)
        noisy = frame.b > 0.8
        self.assertGreater(scale[noisy].mean(), 2 * scale[frame.b < 0.2].mean())

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hgb, lightgbm, xgboost, catboost or ngboost"):
            HybridForecaster(HybridConfig(model="xgb"))._regressor(10)


if __name__ == "__main__":
    unittest.main()
