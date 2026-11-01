"""Train, compare and run a causal Transformer over 24-hour patches of 30 days."""

from .transformer import PatchTransformerConfig, PatchTransformerForecaster
from .lstm_cli import main as recurrent_main


def main(argv=None):
    return recurrent_main(argv, forecaster_class=PatchTransformerForecaster,
                          config_class=PatchTransformerConfig)


if __name__ == "__main__":
    raise SystemExit(main())
