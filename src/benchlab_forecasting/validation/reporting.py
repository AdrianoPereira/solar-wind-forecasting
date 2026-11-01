"""Evaluate saved forecasts independently of models and training."""

import numpy as np
import pandas as pd

from .metrics import CHALLENGE, regression_metrics, timse, timse_proxy


def validate_predictions(frame, models, horizon=72, expected_origins=None):
    """Validate alignment and rebuild the complete expected origin/lead grid.

    Missing rows become missing predictions. Pass expected_origins to account for
    runs absent from the input entirely. Never infer a success from target coverage.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 72:
        raise ValueError("horizon deve estar entre 1 e 72")
    models = list(models)
    reserved = {"origin_utc", "timestamp_utc", "horizon", "y"}
    if not models or len(set(models)) != len(models) or set(models) & reserved:
        raise ValueError("Informe nomes de modelos únicos, sem colunas reservadas")
    required = reserved | set(models)
    if required - set(frame.columns):
        raise ValueError(f"Colunas ausentes: {sorted(required - set(frame.columns))}")
    data = frame[list(required)].copy()
    for key in ("origin_utc", "timestamp_utc"):
        data[key] = pd.to_datetime(data[key], utc=True, errors="raise")
        if data[key].isna().any() or (data[key] != data[key].dt.floor("h")).any():
            raise ValueError(f"{key} deve conter horários inteiros válidos")
    lead = pd.to_numeric(data.horizon, errors="raise")
    if (~lead.isin(range(1, horizon + 1))).any():
        raise ValueError("Horizonte inválido")
    data["horizon"] = lead.astype(int)
    if (data.timestamp_utc != data.origin_utc + pd.to_timedelta(data.horizon, unit="h")).any():
        raise ValueError("timestamp_utc deve ser origin_utc + horizon horas")
    if data.duplicated(["origin_utc", "horizon"]).any():
        raise ValueError("Par emissão/horizonte duplicado")
    for key in ("y", *models):
        data[key] = pd.to_numeric(data[key], errors="raise").astype(float)
        if np.isinf(data[key]).any() or (data[key].dropna() < 0).any():
            raise ValueError(f"{key} contém velocidade negativa ou infinita; use NaN para ausência")
    if data.groupby("timestamp_utc").y.nunique().gt(1).any():
        raise ValueError("Alvos conflitantes para o mesmo timestamp")
    # One observation per target timestamp, including across overlapping forecasts.
    targets = data.groupby("timestamp_utc").y.first()
    origins = pd.DatetimeIndex(pd.to_datetime(
        expected_origins if expected_origins is not None else data.origin_utc.unique(), utc=True))
    if origins.empty or origins.hasnans or not origins.is_unique or not (origins == origins.floor("h")).all():
        raise ValueError("Emissões esperadas devem ser únicas, não vazias e horárias")
    if not data.origin_utc.isin(origins).all():
        raise ValueError("Há emissões fora do calendário esperado")
    index = pd.MultiIndex.from_product([origins.sort_values(), range(1, horizon + 1)],
                                       names=["origin_utc", "horizon"])
    data = data.set_index(["origin_utc", "horizon"]).reindex(index).reset_index()
    data["timestamp_utc"] = data.origin_utc + pd.to_timedelta(data.horizon, unit="h")
    data["y"] = data.timestamp_utc.map(targets)
    return data


def scores(frame, prediction):
    result = regression_metrics(frame.y, frame[prediction])
    result["timse"] = timse(frame.y, frame[prediction])
    result["timse_degenerate"] = bool(frame.y.dropna().nunique() == 1)
    return result


def evaluate_predictions(frame, models, *, horizon=72, expected_origins=None):
    models = tuple(models)
    data = validate_predictions(frame, models, horizon, expected_origins)
    common = data.y.notna() & data[list(models)].notna().all(axis=1)
    metrics = {}
    for name in models:
        complete = data[name].notna().groupby(data.origin_utc).all()
        metrics[name] = {
            "available": scores(data, name), "common": scores(data.loc[common], name),
            "prediction_coverage": float(data[name].notna().mean()),
            "complete_origins": int(complete.sum()),
            "complete_origin_fraction": float(complete.mean()),
            "timse": timse(data.y, data[name]),
            "timse_common": timse(data.loc[common, "y"], data.loc[common, name]),
            "common_by_horizon": {str(h): scores(data.loc[common & data.horizon.eq(h)], name)
                                  for h in range(1, horizon + 1)},
            "common_by_band": {f"{lo}-{hi}": scores(data.loc[common & data.horizon.between(lo, hi)], name)
                               for lo, hi in ((1, 24), (25, 48), (49, 72))},
            "common_by_month": {str(month): scores(group, name) for month, group in
                                data.loc[common].groupby(data.loc[common].timestamp_utc.dt.strftime("%Y-%m"))},
            "timse_proxy": timse_proxy(data.y, data[name]),
            "timse_proxy_common": timse_proxy(data.loc[common, "y"], data.loc[common, name]),
        }
    return {
        "origins": int(data.origin_utc.nunique()), "forecast_pairs": len(data),
        "valid_targets": int(data.y.notna().sum()), "common_pairs": int(common.sum()),
        "aggregation": "forecast-origin/lead pairs; overlaps retained; local convention",
        "challenge": dict(CHALLENGE),
        "timse_definition": "sum_j (unique_y[j+1]-unique_y[j]) * mean_{i: y_i >= unique_y[j]} (prediction_i-y_i)^2",
        "timse_units": "(km/s)^3",
        "timse_missing_policy": "local strict: null if any prediction is missing at a finite target; common scores use identical valid pairs",
        "proxy_definition": "mean_j mean_{i: y_i >= y_j} (prediction_i-y_i)^2; empirical thresholds including ties; not official",
        "coverage_note": "complete_origin_fraction measures finite forecasts, not platform uptime; no score multiplier applied",
        "metrics": metrics,
    }
