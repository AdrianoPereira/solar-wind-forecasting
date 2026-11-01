"""LSTM with causal multivariate windows and direct multi-hour forecasts."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_series, utc_hour
from .features import GROUPS, load_regressors
from .models import future_grid


@dataclass(frozen=True)
class LSTMConfig:
    horizon: int = 72
    lookback_hours: int = 168
    hidden_size: int = 32
    epochs: int = 60
    patience: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1
    latency_minutes: int = 60
    fill_hours: int = 6
    max_age_hours: int = 24
    seed: int = 42
    threads: int = 2
    device: str = "cpu"
    input_group: str = "plasma_mag"

    def __post_init__(self):
        for key in ("horizon", "lookback_hours", "hidden_size", "epochs", "patience",
                    "batch_size", "max_age_hours", "threads", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.horizon > 72:
            raise ValueError("Maximum forecast horizon is 72 hours")
        for key in ("latency_minutes", "fill_hours"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.input_group not in ("speed", *GROUPS):
            raise ValueError("input_group must be speed, plasma or plasma_mag")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")


def load_lstm_data(plasma_path, mag_path=None, group="plasma_mag"):
    raw = (pd.read_parquet(plasma_path) if str(plasma_path).endswith(".parquet")
           else pd.read_csv(plasma_path))
    series = load_series(plasma_path, available_column="available_at" if "available_at" in raw else None)
    series = series.rename(columns={"available_at": "y__available_at"})
    if group != "speed":
        series = series.merge(load_regressors(plasma_path, mag_path, group), on="ds", how="left",
                              validate="one_to_one")
    return series


def indexed_frame(frame):
    result = frame.copy()
    result["ds"] = pd.to_datetime(result.ds, utc=True)
    if result.ds.isna().any() or result.ds.duplicated().any() or (result.ds != result.ds.dt.floor("h")).any():
        raise ValueError("Input timestamps must be unique, valid and hourly")
    return result.set_index("ds").sort_index()


def causal_window(frame, origin, columns, config):
    """Return filled inputs, original missing flags, and latest available speed."""
    origin = utc_hour(origin)
    grid = pd.date_range(end=origin, periods=config.lookback_hours, freq="h")
    block = frame.reindex(grid)
    values = block[columns].astype(float).copy()
    for column in columns:
        key = f"{column}__available_at"
        available = (pd.to_datetime(block[key], utc=True) if key in block else
                     pd.Series(grid + pd.Timedelta(minutes=config.latency_minutes), index=grid))
        values[column] = values[column].where(available.le(origin) & np.isfinite(values[column]))
    valid_speed = values.y.dropna()
    if valid_speed.empty or (origin - valid_speed.index[-1]).total_seconds() > config.max_age_hours * 3600:
        raise ValueError("No sufficiently recent available speed observation")
    anchor = float(valid_speed.iloc[-1])
    missing = values.isna().to_numpy(dtype=np.float32)
    if config.fill_hours:
        values = values.ffill(limit=config.fill_hours)
    return values.to_numpy(dtype=np.float32), missing, anchor


def supervised_samples(frame, origins, columns, config, label_cutoff=None):
    arrays, flags, anchors, targets, kept, failures = [], [], [], [], [], []
    for origin in origins:
        try:
            values, missing, anchor = causal_window(frame, origin, columns, config)
            dates = future_grid(origin, config.horizon)
            labels = frame.reindex(dates).y.astype(float)
            if "y__available_at" in frame:
                labels = labels.where(frame.reindex(dates)["y__available_at"].notna())
            if label_cutoff is not None:
                key = "y__available_at"
                available = (pd.to_datetime(frame.reindex(dates)[key], utc=True) if key in frame else
                             pd.Series(dates + pd.Timedelta(minutes=config.latency_minutes), index=dates))
                labels = labels.where(available.lt(label_cutoff))
            if not np.isfinite(labels).any():
                raise ValueError("No valid target observations")
            arrays.append(values)
            flags.append(missing)
            anchors.append(anchor)
            targets.append(labels.to_numpy(dtype=np.float32))
            kept.append(origin.isoformat())
        except ValueError as exc:
            failures.append({"origin_utc": origin.isoformat(), "error": str(exc)})
    if not arrays:
        raise ValueError("No usable supervised samples")
    return (np.stack(arrays), np.stack(flags), np.asarray(anchors, dtype=np.float32),
            np.stack(targets), kept, failures)


class DirectLSTM(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.encoder = nn.LSTM(features, config.hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.horizon))
        # Begin at persistence; learn corrections for each lead time.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1])


def masked_mse(prediction, target):
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("Loss requires at least one observed target")
    return (prediction[valid] - target[valid]).square().mean()


class LSTMForecaster:
    architecture = "lstm"
    config_class = LSTMConfig
    network_class = DirectLSTM

    def __init__(self, config=None):
        self.config = config or self.config_class()
        self.model = None
        self.columns = ["y", *GROUPS.get(self.config.input_group, [])]

    def transform(self, values, flags):
        filled = np.where(np.isfinite(values), values, self.medians)
        normalized = (filled - self.means) / self.scales
        return torch.from_numpy(np.concatenate([normalized, flags], axis=-1).astype(np.float32))

    def fit(self, frame, train_origins, validation_origins):
        config = self.config
        self.model = None
        torch.set_num_threads(config.threads)
        if config.device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; install a CUDA-enabled PyTorch and check the GPU driver")
        device = torch.device(config.device)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        train_origins = pd.DatetimeIndex([utc_hour(t) for t in train_origins]).sort_values()
        validation_origins = pd.DatetimeIndex([utc_hour(t) for t in validation_origins]).sort_values()
        if train_origins.empty or validation_origins.empty or not train_origins.is_unique or not validation_origins.is_unique:
            raise ValueError("Training and validation origins must be nonempty and unique")
        if train_origins.max() + pd.Timedelta(hours=config.horizon) >= validation_origins.min():
            raise ValueError("Training targets must end before validation begins")
        data = indexed_frame(frame)
        train = supervised_samples(data, train_origins, self.columns, config, validation_origins.min())
        validation = supervised_samples(data, validation_origins, self.columns, config)
        # Do not select an epoch by silently omitting difficult validation origins.
        if validation[-1]:
            raise ValueError(f"Validation has unusable origins: {validation[-1][:3]}")
        x, flags, anchors, y, kept, failures = train
        if len(kept) < 16:
            raise ValueError("At least 16 training origins are required")
        flat = x.reshape(-1, len(self.columns))
        self.medians = np.nanmedian(flat, axis=0)
        if not np.isfinite(self.medians).all():
            raise ValueError("A feature has no finite training observations")
        filled = np.where(np.isfinite(flat), flat, self.medians)
        self.means = filled.mean(axis=0)
        self.scales = np.maximum(filled.std(axis=0), 1e-6)
        self.target_scale = max(float(np.nanstd(y)), 1.0)
        train_x = self.transform(x, flags)
        train_y = torch.from_numpy((y - anchors[:, None]) / self.target_scale)
        vx, vf, va, vy, _, _ = validation
        validation_x = self.transform(vx, vf)
        validation_y = torch.from_numpy((vy - va[:, None]) / self.target_scale)
        network = self.network_class(2 * len(self.columns), config).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        generator = torch.Generator().manual_seed(config.seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                             batch_size=config.batch_size, shuffle=True, generator=generator)
        best_loss, best_weights, stale, history = float("inf"), None, 0, []
        start = time.monotonic()
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.monotonic()
            network.train()
            squared_error, count = 0.0, 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = masked_mse(network(batch_x), batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                n = int(torch.isfinite(batch_y).sum())
                squared_error += float(loss.detach()) * n
                count += n
            network.eval()
            with torch.inference_mode():
                # Batched evaluation also bounds memory on large validation blocks.
                squared_validation, validation_count = 0.0, 0
                for part_x, part_y in zip(validation_x.split(config.batch_size),
                                          validation_y.split(config.batch_size)):
                    part_y = part_y.to(device)
                    valid = torch.isfinite(part_y)
                    n = int(valid.sum())
                    if not n:
                        continue
                    prediction = network(part_x.to(device))
                    squared_validation += float((prediction[valid] - part_y[valid]).square().sum())
                    validation_count += n
                val_loss = squared_validation / validation_count
            if not math.isfinite(val_loss):
                raise ValueError("Nonfinite validation loss")
            row = {"epoch": epoch, "train_rmse_km_s": math.sqrt(squared_error / count) * self.target_scale,
                   "validation_rmse_km_s": math.sqrt(val_loss) * self.target_scale,
                   "seconds": time.monotonic() - epoch_start}
            history.append(row)
            print(f"Epoch {epoch}: train RMSE={row['train_rmse_km_s']:.2f}, "
                  f"validation RMSE={row['validation_rmse_km_s']:.2f} km/s ({row['seconds']:.1f}s)", flush=True)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_weights = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                best_epoch = epoch
            else:
                stale += 1
                if stale >= config.patience:
                    break
        network.load_state_dict(best_weights)
        network.eval()
        self.model = network
        # A selected checkpoint cannot be used before validation labels were available.
        last_label = validation_origins.max() + pd.Timedelta(hours=config.horizon)
        self.available_at = last_label + pd.Timedelta(minutes=config.latency_minutes)
        if "y__available_at" in data:
            observed = data.loc[validation_origins.min():last_label, "y__available_at"]
            latest = pd.to_datetime(observed, utc=True).max()
            if pd.notna(latest):
                self.available_at = max(self.available_at, latest)
        self.training = {"best_epoch": best_epoch, "history": history, "train_origins": kept,
                         "validation_origins": validation[4], "dropped_train_origins": failures,
                         "train_pairs": int(np.isfinite(y).sum()),
                         "validation_pairs": int(np.isfinite(vy).sum()),
                         "train_last_target": (train_origins.max() + pd.Timedelta(hours=config.horizon)).isoformat(),
                         "training_seconds": time.monotonic() - start,
                         "parameters": sum(p.numel() for p in network.parameters()),
                         "selection": "Minimum validation MSE; test targets never used"}
        return self

    def predict(self, frame, origin):
        if self.model is None:
            raise ValueError("Fit or load a model before prediction")
        origin = utc_hour(origin)
        if origin < self.available_at:
            raise ValueError("Origin precedes availability of the selected model")
        values, flags, anchor = causal_window(indexed_frame(frame), origin, self.columns, self.config)
        with torch.inference_mode():
            device = next(self.model.parameters()).device
            correction = self.model(self.transform(values[None], flags[None]).to(device)).cpu().numpy()[0]
        result = np.maximum(anchor + correction * self.target_scale, 0)
        if not np.isfinite(result).all():
            raise ValueError(f"{self.architecture.upper()} produced nonfinite forecasts")
        return pd.DataFrame({"timestamp_utc": future_grid(origin, self.config.horizon), "speed_km_s": result})

    def save(self, path):
        if self.model is None:
            raise ValueError("Fit a model before saving")
        torch.save({"schema_version": 1, "architecture": self.architecture,
                    "config": asdict(self.config), "columns": self.columns,
                    "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "medians": self.medians.tolist(),
                    "means": self.means.tolist(), "scales": self.scales.tolist(),
                    "target_scale": self.target_scale, "available_at": self.available_at.isoformat(),
                    "training": self.training}, Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported recurrent checkpoint schema")
        if payload.get("architecture", "lstm") != cls.architecture:
            raise ValueError(f"Checkpoint architecture does not match {cls.architecture}")
        instance = cls(cls.config_class(**payload["config"]))
        torch.set_num_threads(instance.config.threads)
        if payload["columns"] != instance.columns:
            raise ValueError("Checkpoint feature schema mismatch")
        for name in ("medians", "means", "scales"):
            setattr(instance, name, np.array(payload[name], dtype=np.float32))
        instance.target_scale = payload["target_scale"]
        instance.available_at = pd.Timestamp(payload["available_at"])
        instance.training = payload["training"]
        instance.model = instance.network_class(2 * len(instance.columns), instance.config)
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.eval()
        return instance
