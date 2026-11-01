"""Prophet and causal reference forecasts."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import history_at, utc_hour


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 72
    train_days: int = 180
    latency_minutes: int = 60
    max_age_hours: int = 24
    min_points: int = 48
    growth: str = "flat"
    rotation_days: float = 0
    fourier_order: int = 3
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    regressor_prior_scale: float = 1.0
    exogenous_lag_hours: int = 96
    exogenous_fill_hours: int = 6
    seed: int = 42

    def __post_init__(self):
        for name in ("horizon", "train_days", "max_age_hours", "min_points", "fourier_order"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} deve ser inteiro positivo")
        if self.horizon > 72 or self.min_points < 2:
            raise ValueError("Horizonte máximo: 72 horas; mínimo: 2 observações")
        if not isinstance(self.latency_minutes, int) or self.latency_minutes < 0:
            raise ValueError("latency_minutes deve ser inteiro não negativo")
        if not isinstance(self.seed, int) or not 1 <= self.seed < 2**31:
            raise ValueError("seed deve estar entre 1 e 2**31 - 1")
        if self.growth not in {"flat", "linear"}:
            raise ValueError("growth deve ser flat ou linear")
        if not math.isfinite(self.rotation_days) or self.rotation_days < 0:
            raise ValueError("rotation_days deve ser finito e não negativo")
        if not math.isfinite(self.changepoint_prior_scale) or self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale deve ser finito e positivo")
        for name in ("seasonality_prior_scale", "regressor_prior_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("exogenous_lag_hours", "exogenous_fill_hours"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def future_grid(origin, horizon):
    return pd.date_range(utc_hour(origin) + pd.Timedelta(hours=1), periods=horizon, freq="h")


class ProphetForecaster:
    def __init__(self, config=None):
        self.config = config or ForecastConfig()
        self.model = None

    def fit(self, series, origin, regressors=None):
        from prophet import Prophet

        origin = utc_hour(origin)
        self.model = None
        self.future_regressors = None
        history = history_at(series, origin, self.config)
        if len(history) < self.config.min_points:
            raise ValueError(f"Prophet requer pelo menos {self.config.min_points} pontos válidos")
        if self.config.rotation_days and (
            history.ds.iloc[-1] - history.ds.iloc[0]
        ).total_seconds() / 86400 < 2 * self.config.rotation_days:
            raise ValueError("Sazonalidade solar requer ao menos duas rotações no treino")
        model = Prophet(
            growth=self.config.growth, daily_seasonality=False,
            weekly_seasonality=False, yearly_seasonality=False,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            uncertainty_samples=0,
        )
        if self.config.rotation_days:
            model.add_seasonality(name="solar_rotation", period=self.config.rotation_days,
                                  fourier_order=self.config.fourier_order)
        fit_frame = history.assign(ds=history.ds.dt.tz_localize(None))
        feature_metadata = None
        if regressors is not None:
            from .features import prepare_regressors

            train_x, future_x, feature_metadata = prepare_regressors(
                regressors, history.ds, future_grid(origin, self.config.horizon), origin, self.config)
            for name in train_x:
                model.add_regressor(name, prior_scale=self.config.regressor_prior_scale,
                                    standardize=True, mode="additive")
                fit_frame[name] = train_x[name].to_numpy()
            self.future_regressors = future_x
        model.fit(fit_frame, seed=self.config.seed)
        self.model = model
        self.origin = origin
        self.training = {"start": history.ds.iloc[0].isoformat(),
                         "end": history.ds.iloc[-1].isoformat(), "points": len(history),
                         "availability": "available_at" if "available_at" in series
                         else "assumed_latency"}
        if feature_metadata is not None:
            self.training["exogenous"] = feature_metadata
        return self

    def predict(self):
        if self.model is None:
            raise ValueError("Execute fit antes de predict")
        dates = future_grid(self.origin, self.config.horizon)
        future = pd.DataFrame({"ds": dates.tz_localize(None)})
        if self.future_regressors is not None:
            for name in self.future_regressors:
                future[name] = self.future_regressors[name].to_numpy()
        prediction = self.model.predict(future)
        values = prediction.yhat.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Prophet produziu previsões não finitas")
        return pd.DataFrame({"timestamp_utc": dates, "speed_km_s": np.maximum(values, 0)})

    def save(self, path):
        from prophet.serialize import model_to_json

        if self.model is None:
            raise ValueError("Execute fit antes de save")
        payload = {"schema_version": 2, "config": asdict(self.config),
                   "origin": self.origin.isoformat(), "training": self.training,
                   "future_regressors": (self.future_regressors.to_dict(orient="list")
                                         if self.future_regressors is not None else None),
                   "model": json.loads(model_to_json(self.model))}
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        from prophet.serialize import model_from_json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema_version"] not in (1, 2):
            raise ValueError("Versão de modelo não suportada")
        instance = cls(ForecastConfig(**payload["config"]))
        instance.origin = utc_hour(payload["origin"])
        instance.training = payload["training"]
        instance.future_regressors = (pd.DataFrame(payload["future_regressors"])
                                     if payload.get("future_regressors") is not None else None)
        instance.model = model_from_json(json.dumps(payload["model"]))
        return instance


def baseline(series, origin, config, name):
    history = history_at(series, origin, config)
    dates = future_grid(origin, config.horizon)
    if name == "persistence":
        values = np.full(len(dates), history.y.iloc[-1])
    elif name == "recurrence":
        # Exact historical timestamps; missing recurrence stays missing.
        values = history.set_index("ds").y.reindex(dates - pd.Timedelta(days=27)).to_numpy()
    else:
        raise ValueError(f"Baseline desconhecido: {name}")
    return pd.DataFrame({"timestamp_utc": dates, "speed_km_s": values})
