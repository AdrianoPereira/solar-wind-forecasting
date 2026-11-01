"""Train, compare and run a causal GRU with horizon-specific attention."""

from .attention_gru import AttentionGRUConfig, AttentionGRUForecaster
from .lstm_cli import main as recurrent_main


def main(argv=None):
    return recurrent_main(argv, forecaster_class=AttentionGRUForecaster,
                          config_class=AttentionGRUConfig)


if __name__ == "__main__":
    raise SystemExit(main())
