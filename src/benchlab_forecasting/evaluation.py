"""Walk-forward refits with explicit failures and common-target comparisons."""

import numpy as np
import pandas as pd

from .data import utc_hour
from .models import ForecastConfig, ProphetForecaster, baseline, future_grid
from .validation.reporting import evaluate_predictions, scores


def backtest(series, start, end, config=None, step_hours=6,
             models=("prophet", "persistence", "recurrence"), regressors=None):
    config = config or ForecastConfig()
    start, end = utc_hour(start), utc_hour(end)
    if not isinstance(step_hours, int) or step_hours < 1 or start > end:
        raise ValueError("Passo deve ser positivo e início deve ser <= fim")
    if not models or len(set(models)) != len(models) or set(models) - {"prophet", "persistence", "recurrence"}:
        raise ValueError("Modelos devem ser únicos: prophet, persistence, recurrence")
    if end + pd.Timedelta(hours=config.horizon) > series.ds.max():
        raise ValueError("O fim do backtest mais o horizonte ultrapassa os dados")
    labels = series.set_index("ds").y
    rows, failures = [], []
    origins = pd.date_range(start, end, freq=f"{step_hours}h")
    for origin in origins:
        dates = future_grid(origin, config.horizon)
        block = pd.DataFrame({"origin_utc": origin, "timestamp_utc": dates,
                              "horizon": np.arange(1, config.horizon + 1),
                              "y": labels.reindex(dates).to_numpy()})
        for name in models:
            try:
                forecast = (ProphetForecaster(config).fit(series, origin, regressors).predict()
                            if name == "prophet" else baseline(series, origin, config, name))
                block[name] = forecast.speed_km_s.to_numpy()
            except Exception as exc:
                block[name] = np.nan
                failures.append({"origin_utc": origin.isoformat(), "model": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
        rows.append(block)
    frame = pd.concat(rows, ignore_index=True)
    report = evaluate_predictions(frame, models, horizon=config.horizon, expected_origins=origins)
    report["failures"] = failures
    return frame, report
