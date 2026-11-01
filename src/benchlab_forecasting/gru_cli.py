"""GRU training, comparison and inference using the shared recurrent workflow."""

from .gru import GRUConfig, GRUForecaster
from .lstm_cli import main as recurrent_main


def main(argv=None):
    return recurrent_main(argv, forecaster_class=GRUForecaster, config_class=GRUConfig)


if __name__ == "__main__":
    raise SystemExit(main())
