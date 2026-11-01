"""Train, compare and run a causal hourly Transformer encoder."""

from .transformer import TransformerConfig, TransformerForecaster
from .lstm_cli import main as recurrent_main


def main(argv=None):
    return recurrent_main(argv, forecaster_class=TransformerForecaster,
                          config_class=TransformerConfig)


if __name__ == "__main__":
    raise SystemExit(main())
