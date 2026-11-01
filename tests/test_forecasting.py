from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchlab_forecasting import ForecastConfig, ProphetForecaster, backtest, load_series
from benchlab_forecasting.cli import main
from benchlab_forecasting.data import history_at
from benchlab_forecasting.models import baseline


def sample(days=70):
    dates = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    return pd.DataFrame({"ds": dates,
                         "y": 420 + 60 * np.sin(np.arange(len(dates)) * 2 * np.pi / (27 * 24))})


class ForecastTests(unittest.TestCase):
    def test_clean_flags_without_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
                          "speed_km_s": [400, -9999, 450, 500, float("inf")],
                          "status": [0, 0, 9, 0, 0],
                          "is_valid": [True, True, True, False, True]}).to_csv(path, index=False)
            frame = load_series(path)
            self.assertEqual(frame.y.notna().sum(), 1)
            self.assertEqual(str(frame.ds.dt.tz), "UTC")
            bad = pd.read_csv(path)
            bad.loc[1, "timestamp_utc"] = bad.loc[0, "timestamp_utc"]
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "únicos"):
                load_series(path)

    def test_publication_time_and_assumed_latency(self):
        data = sample(3)
        origin = data.ds.iloc[-1]
        config = ForecastConfig(latency_minutes=120)
        history = history_at(data, origin, config)
        self.assertEqual(history.ds.max(), origin - pd.Timedelta(hours=2))
        data["available_at"] = data.ds + pd.Timedelta(hours=1)
        data.loc[data.index[-4:], "available_at"] = origin + pd.Timedelta(hours=1)
        self.assertEqual(history_at(data, origin, config).ds.max(), data.ds.iloc[-5])
        with self.assertRaisesRegex(ValueError, "idade"):
            history_at(data, origin, replace(config, max_age_hours=2))

    def test_recurrence_is_aligned_to_each_target(self):
        data = sample()
        data["y"] = np.arange(len(data), dtype=float)
        origin = data.ds.iloc[-1]
        result = baseline(data, origin, ForecastConfig(), "recurrence")
        expected = data.set_index("ds").y.reindex(result.timestamp_utc - pd.Timedelta(days=27))
        np.testing.assert_array_equal(result.speed_km_s, expected)

    def test_prophet_future_is_excluded_and_json_roundtrip(self):
        data = sample(6)
        config = ForecastConfig(train_days=5, horizon=72)
        origin = data.ds.iloc[95]
        model = ProphetForecaster(config).fit(data, origin)
        predicted = model.predict()
        changed = data.copy()
        changed.loc[changed.ds >= origin, "y"] = 100000
        other = ProphetForecaster(config).fit(changed, origin).predict()
        np.testing.assert_allclose(predicted.speed_km_s, other.speed_km_s)
        self.assertEqual(len(predicted), 72)
        self.assertEqual(predicted.timestamp_utc.iloc[0], origin + pd.Timedelta(hours=1))
        self.assertEqual(predicted.timestamp_utc.iloc[-1], origin + pd.Timedelta(hours=72))
        self.assertTrue(np.isfinite(predicted.speed_km_s).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = ProphetForecaster.load(path).predict()
            np.testing.assert_allclose(restored.speed_km_s, predicted.speed_km_s)

    def test_rotation_and_linear_growth(self):
        model = ProphetForecaster(ForecastConfig(rotation_days=27, growth="linear"))
        data = sample()
        result = model.fit(data, data.ds.iloc[-1]).predict()
        self.assertTrue(np.isfinite(result.speed_km_s).all())
        self.assertIn("solar_rotation", model.model.seasonalities)
        with self.assertRaisesRegex(ValueError, "duas rotações"):
            model.fit(sample(3), "2024-01-04")

    def test_backtest_missing_labels_common_metrics_and_failures(self):
        data = sample(40)
        origin = data.ds.iloc[32 * 24]
        data.loc[data.ds == origin + pd.Timedelta(hours=2), "y"] = np.nan
        recurrence_time = origin + pd.Timedelta(hours=3) - pd.Timedelta(days=27)
        data.loc[data.ds == recurrence_time, "y"] = np.nan
        frame, report = backtest(data, origin, origin, ForecastConfig(horizon=6),
                                 models=("persistence", "recurrence"))
        self.assertEqual(len(frame), 6)
        self.assertEqual(report["valid_targets"], 5)
        self.assertEqual(report["common_pairs"], 4)
        for name in ("persistence", "recurrence"):
            self.assertEqual(report["metrics"][name]["common"]["n"], 4)
        _, failed = backtest(data, origin, origin, ForecastConfig(min_points=10000),
                             models=("prophet", "persistence"))
        self.assertEqual(len(failed["failures"]), 1)
        self.assertEqual(failed["metrics"]["prophet"]["prediction_coverage"], 0)
        self.assertEqual(failed["common_pairs"], 0)

    def test_cli_forecast_and_backtest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.parquet"
            data = sample(8)
            data.rename(columns={"ds": "timestamp_utc", "y": "speed_km_s"}).to_parquet(path)
            output = root / "forecast"
            self.assertEqual(main(["forecast", "--data", str(path), "--output-dir", str(output),
                                   "--origin", "2024-01-06T00:00:00Z"]), 0)
            self.assertTrue((output / "model.json").exists())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["data_sha256"]), 64)
            self.assertEqual(len(pd.read_csv(output / "forecast.csv")), 72)
            output = root / "backtest"
            self.assertEqual(main(["backtest", "--data", str(path), "--output-dir", str(output),
                                   "--start", "2024-01-04T00:00:00Z",
                                   "--end", "2024-01-04T06:00:00Z", "--models",
                                   "prophet", "persistence"]), 0)
            self.assertEqual(len(pd.read_parquet(output / "predictions.parquet")), 144)


if __name__ == "__main__":
    unittest.main()
