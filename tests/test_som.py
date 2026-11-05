import unittest

import numpy as np
import pandas as pd

from benchlab_forecasting.som import (
    SOM, SOMConfig, alpha_curve, block_bootstrap_gain, combine_inputs, crps_ensemble, fit_scaling, medoids, future_windows, link_matrix, mutual_information,
    past_windows, post_process, recurrence_windows, scenario_weights,
)


def clusters(seed=0, n=300):
    """Three noisy curve shapes of length 12."""
    rng = np.random.default_rng(seed)
    shapes = np.stack([np.linspace(0, 100, 12), np.linspace(0, -100, 12), np.zeros(12)])
    labels = rng.integers(0, 3, n)
    return shapes[labels] + rng.normal(0, 5, (n, 12)), labels


class SOMTest(unittest.TestCase):
    def test_training_is_deterministic_and_separates_shapes(self):
        data, labels = clusters()
        a = SOM(SOMConfig(4, 4, epochs=20)).fit(data)
        b = SOM(SOMConfig(4, 4, epochs=20)).fit(data)
        np.testing.assert_array_equal(a.prototypes, b.prototypes)
        nodes = a.bmu(data)
        # Each node is (almost) pure: shapes do not share best-matching units.
        for node in np.unique(nodes):
            self.assertGreater(np.bincount(labels[nodes == node]).max() / (nodes == node).sum(), 0.95)
        self.assertLess(a.quantization_error(data), 10)

    def test_link_rows_are_distributions(self):
        data, _ = clusters()
        past = SOM(SOMConfig(3, 3, epochs=10)).fit(data)
        future = SOM(SOMConfig(2, 2, epochs=10)).fit(-data)
        link = link_matrix(past, future, data, -data)
        self.assertEqual(link.shape, (9, 4))
        np.testing.assert_allclose(link.sum(1), 1)
        self.assertTrue((link > 0).all())

    def test_weights_follow_the_forecast_and_alpha_one_keeps_it(self):
        prototypes = np.array([[0.0] * 4, [100.0] * 4])
        link = np.array([[0.5, 0.5]])
        w = scenario_weights(link, np.array([0]), np.array([[95.0] * 4]), prototypes, sigma=10)
        self.assertGreater(w[0, 1], 0.99)
        np.testing.assert_allclose(scenario_weights(link, np.array([0]), np.zeros((1, 4)), prototypes, np.inf), link)
        forecast = np.array([[400.0, 410, 420, 430]])
        kept = post_process(forecast, np.array([400.0]), w, prototypes, np.ones(4))
        np.testing.assert_allclose(kept, forecast)
        moved = post_process(forecast, np.array([400.0]), w, prototypes, np.zeros(4))
        np.testing.assert_allclose(moved, 400 + w @ prototypes)
        np.testing.assert_allclose(alpha_curve(1, 0, 3), [1, 0.5, 0])

    def test_windows_are_causal_and_relative(self):
        index = pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC")
        speed = pd.Series(np.arange(400, dtype=float), index=index)
        origin = pd.DatetimeIndex([index[200]])
        past, last, ok = past_windows(speed, origin, latency_hours=1, window=72)
        self.assertTrue(ok[0])
        self.assertEqual(last[0], 199)  # the cut-off hour t0 − 1 h, never t0
        np.testing.assert_allclose(past[0], np.arange(128, 200) - 199)
        # Changing anything at or after t0 leaves the past window unchanged.
        changed = speed.copy()
        changed.iloc[200:] = 9999
        np.testing.assert_allclose(past_windows(changed, origin)[0], past)
        future, ok_future = future_windows(speed, origin, last, horizon=72)
        self.assertTrue(ok_future[0])
        np.testing.assert_allclose(future[0], np.arange(201, 273) - 199)

    def test_recurrence_window_uses_only_data_before_the_cutoff(self):
        index = pd.date_range("2020-01-01", periods=2000, freq="h", tz="UTC")
        speed = pd.Series(np.sin(np.arange(2000) / 50) * 100 + 400, index=index)
        origin = pd.DatetimeIndex([index[1500]])
        last = np.array([400.0])
        window = recurrence_windows(speed, origin, last)
        changed = speed.copy()
        changed.iloc[1499:] = 0  # everything from the cut-off t0 − 1 h on
        np.testing.assert_allclose(recurrence_windows(changed, origin, last), window)
        self.assertEqual(window.shape, (1, 72))

    def test_medoids_are_real_members_and_crps_matches_mae_for_one_member(self):
        members = np.array([[0.0, 0], [10, 10], [11, 9], [100, 100]])
        prototypes = np.array([[10.5, 9.5], [90.0, 90]])
        assigned = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(medoids(prototypes, members, assigned), [1, 3])
        y = np.array([[1.0, 2, 3]])
        one = np.array([[[2.0, 2, 5]]])
        self.assertAlmostEqual(crps_ensemble(y, one), np.mean(np.abs(one[0, 0] - y[0])))
        # A wider ensemble centred on the truth scores better than a single offset member.
        spread = np.array([[[0.0, 1, 2], [2.0, 3, 4]]])
        self.assertLess(crps_ensemble(y, spread), crps_ensemble(y, one))

    def test_coronal_hole_block_is_scaled_to_the_speed_block(self):
        rng = np.random.default_rng(2)
        past = rng.normal(0, 80, (500, 72))
        holes = rng.uniform(0, 0.3, (500, 36))
        holes[0, :12] = np.nan  # one missing image
        params = fit_scaling(past, holes)
        x = combine_inputs(past, holes, params, ch_weight=1.0)
        self.assertEqual(x.shape, (500, 108))
        # Equal total variance in both blocks when the weight is 1; a missing image counts as average.
        self.assertAlmostEqual(x[:, :72].var(0).sum() / x[:, 72:].var(0).sum(), 1, delta=0.1)
        np.testing.assert_allclose(x[0, 72:84], 0)
        np.testing.assert_array_equal(combine_inputs(past, None, params, None), past)

    def test_gaps_mark_windows_unusable(self):
        index = pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC")
        speed = pd.Series(400.0, index=index)
        speed.iloc[100:199] = np.nan
        _, _, ok = past_windows(speed, pd.DatetimeIndex([index[200]]))
        self.assertFalse(ok[0])

    def test_bootstrap_and_mutual_information(self):
        origins = pd.date_range("2024-01-01", periods=400, freq="6h", tz="UTC")
        rng = np.random.default_rng(1)
        y = rng.normal(400, 50, (400, 72))
        gain = block_bootstrap_gain(origins, y, y + 20, y + 10, draws=200)
        self.assertAlmostEqual(gain["gain_km_s"], 10, places=6)
        self.assertLessEqual(gain["ci95"][0], gain["gain_km_s"])
        mi = mutual_information(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(mi["bits"], 1)
        self.assertAlmostEqual(mi["share_of_future_entropy"], 1)
        independent = mutual_information(rng.integers(0, 10, 300), rng.integers(0, 10, 300), shuffles=20)
        self.assertLess(abs(independent["excess_bits"]), 0.1)


if __name__ == "__main__":
    unittest.main()
