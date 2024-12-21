"""Utility functions and classes."""

from .data_handler import load_data, split_data
from .logging_config import setup_logger
from .metrics import Metrics
from .plotting import (
    plot_evolution_metrics,
    plot_operator_distribution,
    plot_prediction_analysis,
    plot_variable_importance,
)


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed (int): The seed value to use
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


__all__: list[str] = [
    "load_data",
    "split_data",
    "setup_logger",
    "Metrics",
    "plot_evolution_metrics",
    "plot_operator_distribution",
    "plot_prediction_analysis",
    "plot_variable_importance",
    "set_global_seed",
]
