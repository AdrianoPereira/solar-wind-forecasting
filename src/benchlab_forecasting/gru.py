"""Direct CPU GRU, sharing the LSTM's causal preprocessing and training loop."""

from dataclasses import dataclass

from torch import nn

from .lstm import LSTMConfig, LSTMForecaster, load_lstm_data


@dataclass(frozen=True)
class GRUConfig(LSTMConfig):
    """GRU hyperparameters, with the same defaults as the reference LSTM."""


class DirectGRU(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.encoder = nn.GRU(features, config.hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.horizon))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, hidden = self.encoder(x)
        return self.head(hidden[-1])


class GRUForecaster(LSTMForecaster):
    architecture = "gru"
    config_class = GRUConfig
    network_class = DirectGRU


load_gru_data = load_lstm_data
