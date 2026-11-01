"""GRU with horizon-specific cross-attention over causal hourly history."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .lstm import LSTMConfig, LSTMForecaster, load_lstm_data


@dataclass(frozen=True)
class AttentionGRUConfig(LSTMConfig):
    attention_heads: int = 4
    residual_scale: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.attention_heads, bool) or
                not isinstance(self.attention_heads, int) or self.attention_heads < 1):
            raise ValueError("attention_heads must be a positive integer")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")


class DirectAttentionGRU(nn.Module):
    """One learned query per forecast hour attends to all historical GRU states."""

    encoder_class = nn.GRU

    def __init__(self, features, config):
        super().__init__()
        self.residual_scale = config.residual_scale
        self.encoder = self.encoder_class(features, config.hidden_size, batch_first=True)
        self.horizon_queries = nn.Embedding(config.horizon, config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.attention_heads, dropout=config.dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_size, 1))
        # The forecaster adds the last available speed to these corrections.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, x):
        """Return every historical state and the final summary state."""
        history, hidden = self.encoder(x)
        return history, hidden[-1]

    def forward(self, x):
        history, last = self.encode(x)
        horizons = torch.arange(self.horizon_queries.num_embeddings, device=x.device)
        queries = self.horizon_queries(horizons).unsqueeze(0) + last.unsqueeze(1)
        queries = queries.expand(x.shape[0], -1, -1)
        context, _ = self.attention(queries, history, history, need_weights=False)
        context = context + self.residual_scale * last.unsqueeze(1)
        return self.head(context).squeeze(-1)


class AttentionGRUForecaster(LSTMForecaster):
    architecture = "attention_gru"
    config_class = AttentionGRUConfig
    network_class = DirectAttentionGRU


load_attention_gru_data = load_lstm_data
