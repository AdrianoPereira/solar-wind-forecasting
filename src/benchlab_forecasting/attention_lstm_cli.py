"""Train, compare and run a causal LSTM with horizon-specific attention."""

from .attention_lstm import AttentionLSTMConfig, AttentionLSTMForecaster
from .lstm_cli import main as recurrent_main


def main(argv=None):
    return recurrent_main(argv, forecaster_class=AttentionLSTMForecaster,
                          config_class=AttentionLSTMConfig)


if __name__ == "__main__":
    raise SystemExit(main())
