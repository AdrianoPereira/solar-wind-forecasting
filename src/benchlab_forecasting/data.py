"""Read cleaned ACE data without interpolating observations or labels."""

from pathlib import Path

import numpy as np
import pandas as pd


def utc_hour(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Timestamp ausente")
    stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    if stamp != stamp.floor("h"):
        raise ValueError("O timestamp deve estar na grade horária")
    return stamp


def load_series(path, time_column="timestamp_utc", target_column="speed_km_s",
                available_column=None):
    """Naive timestamps mean UTC; optional availability overrides assumed latency."""
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = pd.DataFrame({
        "ds": pd.to_datetime(frame[time_column], utc=True, errors="raise"),
        "y": pd.to_numeric(frame[target_column], errors="raise"),
    })
    if result.empty or result.ds.isna().any():
        raise ValueError("Série vazia ou timestamps ausentes")
    if result.ds.duplicated().any() or (result.ds != result.ds.dt.floor("h")).any():
        raise ValueError("Os timestamps devem ser únicos e horários")
    valid = np.isfinite(result.y) & (result.y >= 0)
    if "status" in frame:
        valid &= pd.to_numeric(frame.status, errors="raise").eq(0)
    if "is_valid" in frame:
        flags = frame.is_valid.astype("string").str.lower()
        if not flags.dropna().isin(["true", "false", "1", "0"]).all():
            raise ValueError("is_valid deve conter booleanos ou 0/1")
        valid &= flags.isin(["true", "1"])
    result.loc[~valid, "y"] = np.nan
    if available_column:
        result["available_at"] = pd.to_datetime(frame[available_column], utc=True,
                                                errors="raise")
        if (result.available_at < result.ds).any():
            raise ValueError("Disponibilidade anterior à observação")
    return result.sort_values("ds").reset_index(drop=True)


def history_at(series, origin, config):
    origin = utc_hour(origin)
    available = (series.available_at if "available_at" in series else
                 series.ds + pd.Timedelta(minutes=config.latency_minutes))
    mask = ((series.ds <= origin) & (available <= origin)
            & (series.ds > origin - pd.Timedelta(days=config.train_days)))
    history = series.loc[mask, ["ds", "y"]].dropna().copy()
    if history.empty:
        raise ValueError("Nenhuma observação válida disponível na janela de treino")
    history = history.sort_values("ds")
    age = (origin - history.ds.iloc[-1]).total_seconds() / 3600
    if age > config.max_age_hours:
        raise ValueError(f"Última observação válida tem {age:g} horas de idade")
    return history
