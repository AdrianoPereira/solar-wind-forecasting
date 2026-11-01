"""Chronological splits for supervised samples indexed by forecast origin."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import utc_hour


@dataclass(frozen=True)
class TemporalFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def temporal_splits(origins, validation_start, *, strategy="expanding", n_splits=3,
                    validation_hours=168, step_hours=None, horizon=72,
                    train_hours=None, gap_hours=0):
    """Half-open validation blocks; purge train labels reaching validation.

    Indices refer to input positions. Expanding grows from the first origin;
    rolling limits the training-origin window relative to validation start;
    holdout returns exactly one split. No random shuffling is supported.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Emissões devem ser não vazias, únicas e ordenadas")
    if not (dates == dates.floor("h")).all():
        raise ValueError("Emissões devem estar na grade horária")
    if strategy not in {"holdout", "expanding", "rolling"}:
        raise ValueError("strategy deve ser holdout, expanding ou rolling")
    for name, value in (("n_splits", n_splits), ("validation_hours", validation_hours),
                        ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} deve ser inteiro positivo")
    if isinstance(gap_hours, bool) or not isinstance(gap_hours, int) or gap_hours < 0:
        raise ValueError("gap_hours deve ser inteiro não negativo")
    step_hours = validation_hours if step_hours is None else step_hours
    if not isinstance(step_hours, int) or step_hours < validation_hours:
        raise ValueError("step_hours deve ser >= validation_hours para evitar blocos sobrepostos")
    if train_hours is not None and (not isinstance(train_hours, int) or train_hours < 1):
        raise ValueError("train_hours deve ser inteiro positivo")
    if strategy == "rolling" and train_hours is None:
        raise ValueError("rolling exige train_hours")
    if strategy != "rolling" and train_hours is not None:
        raise ValueError("train_hours aplica-se somente a rolling")
    start = utc_hour(validation_start)
    folds = []
    for k in range(1 if strategy == "holdout" else n_splits):
        left = start + pd.Timedelta(hours=k * step_hours)
        right = left + pd.Timedelta(hours=validation_hours)
        train = dates + pd.Timedelta(hours=horizon + gap_hours) < left
        if strategy == "rolling":
            train &= dates >= left - pd.Timedelta(hours=train_hours)
        validation = (dates >= left) & (dates < right)
        if not train.any() or not validation.any():
            raise ValueError(f"Fold {k}: treino ou validação vazio após corte temporal")
        folds.append(TemporalFold(np.flatnonzero(train), np.flatnonzero(validation), left, right))
    return folds
