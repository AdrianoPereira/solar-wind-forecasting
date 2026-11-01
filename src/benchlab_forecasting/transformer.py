"""Attention-only encoders: hourly Transformer and 24-hour-patch Transformer."""

from dataclasses import dataclass

import torch
from torch import nn

from .attention_gru import AttentionGRUConfig
from .lstm import LSTMForecaster, load_lstm_data


@dataclass(frozen=True)
class TransformerConfig(AttentionGRUConfig):
    layers: int = 2
    feedforward_size: int = 64
    patch_hours: int = 1

    def __post_init__(self):
        super().__post_init__()
        for key in ("layers", "feedforward_size", "patch_hours"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.lookback_hours % self.patch_hours:
            raise ValueError("lookback_hours must be divisible by patch_hours")


@dataclass(frozen=True)
class PatchTransformerConfig(TransformerConfig):
    """Thirty days of daily patches, so attention can reach the 27-day recurrence."""

    lookback_hours: int = 720
    patch_hours: int = 24


class DirectTransformer(nn.Module):
    """Self-attention over history tokens, then one learned query per forecast hour."""

    def __init__(self, features, config):
        super().__init__()
        self.patch_hours = config.patch_hours
        self.residual_scale = config.residual_scale
        tokens = config.lookback_hours // config.patch_hours
        # Each token holds every feature of one patch; the last token ends at the origin.
        self.embedding = nn.Linear(features * config.patch_hours, config.hidden_size)
        self.positions = nn.Embedding(tokens, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size, config.attention_heads, config.feedforward_size,
            config.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.horizon_queries = nn.Embedding(config.horizon, config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.attention_heads, dropout=config.dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_size, 1))
        # The forecaster adds the last available speed to these corrections.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        batch, hours, features = x.shape
        tokens = x.reshape(batch, hours // self.patch_hours, self.patch_hours * features)
        positions = torch.arange(tokens.shape[1], device=x.device)
        history = self.norm(self.encoder(self.embedding(tokens) + self.positions(positions)))
        last = history[:, -1]
        horizons = torch.arange(self.horizon_queries.num_embeddings, device=x.device)
        queries = self.horizon_queries(horizons).unsqueeze(0) + last.unsqueeze(1)
        context, _ = self.attention(queries, history, history, need_weights=False)
        context = context + self.residual_scale * last.unsqueeze(1)
        return self.head(context).squeeze(-1)


class TransformerForecaster(LSTMForecaster):
    architecture = "transformer"
    config_class = TransformerConfig
    network_class = DirectTransformer


class PatchTransformerForecaster(LSTMForecaster):
    architecture = "patch_transformer"
    config_class = PatchTransformerConfig
    network_class = DirectTransformer


load_transformer_data = load_lstm_data
