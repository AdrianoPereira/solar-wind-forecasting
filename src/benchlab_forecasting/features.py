"""Quality-controlled, causal exogenous inputs for hourly Prophet forecasts."""

import numpy as np
import pandas as pd

from .data import utc_hour


GROUPS = {"plasma": ["density_cm3", "temperature_k"],
          "plasma_mag": ["density_cm3", "temperature_k", "bx_gsm_nt",
                         "by_gsm_nt", "bz_gsm_nt", "bt_nt"]}


def _read_source(path, columns):
    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    dates = pd.to_datetime(frame.timestamp_utc, utc=True)
    if dates.isna().any() or dates.duplicated().any() or (dates != dates.dt.floor("h")).any():
        raise ValueError("Exogenous timestamps must be unique, valid and hourly")
    values = frame[columns].apply(pd.to_numeric, errors="raise").astype(float)
    values = values.where(np.isfinite(values))
    if "status" in frame:
        values.loc[pd.to_numeric(frame.status, errors="raise").ne(0), :] = np.nan
    for column in columns:
        if column in {"density_cm3", "temperature_k", "bt_nt"}:
            values[column] = values[column].where(values[column] >= 0)
        else:
            values[column] = values[column].mask(values[column].eq(-999.9))
    # Keep source-specific availability; these are not collection timestamps.
    if "available_at" in frame:
        available = pd.to_datetime(frame.available_at, utc=True)
        if (available < dates).any():
            raise ValueError("Exogenous availability precedes observation")
        for column in columns:
            values[f"{column}__available_at"] = available
    values.index = pd.DatetimeIndex(dates, name="ds")
    return values.sort_index()


def load_regressors(plasma_path, mag_path=None, group="plasma"):
    if group not in GROUPS:
        raise ValueError(f"Unknown exogenous group: {group}")
    plasma = _read_source(plasma_path, GROUPS["plasma"])
    if group == "plasma_mag":
        if mag_path is None:
            raise ValueError("plasma_mag requires a MAG file")
        mag = _read_source(mag_path, GROUPS["plasma_mag"][2:])
        plasma = plasma.join(mag, how="outer")
    return plasma.reset_index()


def lagged_features(regressors, dates, origin, config):
    """Use t-lag for target t, with bounded forward fill and no future values.

    Each source value must be available by source_time + lag - horizon.
    This conservative condition is also imposed on historical training features.
    """
    if config.exogenous_lag_hours < config.horizon + int(np.ceil(config.latency_minutes / 60)):
        raise ValueError("Exogenous lag must cover horizon plus assumed latency")
    origin = utc_hour(origin)
    dates = pd.DatetimeIndex(dates)
    raw = regressors.copy()
    raw["ds"] = pd.to_datetime(raw.ds, utc=True)
    if raw.ds.isna().any() or raw.ds.duplicated().any() or (raw.ds != raw.ds.dt.floor("h")).any():
        raise ValueError("Exogenous timestamps must be unique, valid and hourly")
    columns = [c for c in raw if c != "ds" and not c.endswith("__available_at")]
    if not columns:
        raise ValueError("No exogenous columns supplied")
    lookup = dates - pd.Timedelta(hours=config.exogenous_lag_hours)
    raw = raw.set_index("ds").sort_index()
    window = pd.date_range(lookup.min() - pd.Timedelta(hours=config.exogenous_fill_hours),
                           lookup.max(), freq="h")
    raw = raw.reindex(window)
    values = pd.DataFrame(index=window)
    for column in columns:
        key = f"{column}__available_at"
        available = (raw[key] if key in raw else
                     pd.Series(window + pd.Timedelta(minutes=config.latency_minutes), index=window))
        deadline = window + pd.Timedelta(hours=config.exogenous_lag_hours - config.horizon)
        valid = available.le(origin) & available.le(deadline)
        values[column] = pd.to_numeric(raw[column], errors="raise").where(valid)
    values = values.where(np.isfinite(values))
    if config.exogenous_fill_hours:
        values = values.ffill(limit=config.exogenous_fill_hours)
    features = values.reindex(lookup).reset_index(drop=True)
    features.columns = [f"{c}_lag{config.exogenous_lag_hours}h" for c in columns]
    features.index = dates
    return features


def prepare_regressors(regressors, train_dates, future_dates, origin, config):
    """Fit medians on training rows only; Prophet fits its own scaling there."""
    train = lagged_features(regressors, train_dates, origin, config)
    future = lagged_features(regressors, future_dates, origin, config)
    medians = train.median()
    if medians.isna().any():
        raise ValueError(f"No training observations for regressors: {list(medians[medians.isna()].index)}")
    missing = {"train_missing_fraction": train.isna().mean().to_dict(),
               "future_missing_fraction": future.isna().mean().to_dict(),
               "imputation_medians": medians.to_dict()}
    train_flags = train.isna().astype(float).add_suffix("_missing")
    future_flags = future.isna().astype(float).add_suffix("_missing")
    train = pd.concat([train.fillna(medians), train_flags], axis=1)
    future = pd.concat([future.fillna(medians), future_flags], axis=1)
    kept = train.columns[train.nunique() > 1]
    if not len(kept):
        raise ValueError("All training regressors are constant")
    missing["retained_columns"] = list(kept)
    return train[kept].reset_index(drop=True), future[kept].reset_index(drop=True), missing
