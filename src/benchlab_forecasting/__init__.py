"""Hourly solar-wind forecasts and chronological evaluation."""

# Imported on first use, so light modules (hybrid, coronal_holes) load without Prophet.
_EXPORTS = {"load_series": ".data", "ForecastConfig": ".models", "ProphetForecaster": ".models",
            "backtest": ".evaluation"}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module

        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
