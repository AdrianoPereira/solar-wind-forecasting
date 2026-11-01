"""LSTM with horizon-specific cross-attention over causal hourly history."""

from dataclasses import dataclass

from torch import nn

from .attention_gru import AttentionGRUConfig, DirectAttentionGRU
from .lstm import LSTMForecaster, load_lstm_data


@dataclass(frozen=True)
class AttentionLSTMConfig(AttentionGRUConfig):
    """Same hyperparameters and defaults as the attention GRU."""


class DirectAttentionLSTM(DirectAttentionGRU):
    """The attention GRU's horizon queries over LSTM states instead of GRU states."""

    encoder_class = nn.LSTM

    def encode(self, x):
        history, (hidden, _) = self.encoder(x)
        return history, hidden[-1]


class AttentionLSTMForecaster(LSTMForecaster):
    architecture = "attention_lstm"
    config_class = AttentionLSTMConfig
    network_class = DirectAttentionLSTM


load_attention_lstm_data = load_lstm_data
