from typing import NamedTuple

import numpy as np

from symb_regression.core.tree import Node
from symb_regression.operators.definitions import SymbolicConfig


class Metrics(NamedTuple):
    # Generation tracking
    generation: int
    # execution_time: float

    # Fitness metrics
    best_fitness: np.float64
    avg_fitness: np.float64
    worst_fitness: np.float64
    fitness_std: np.float64

    # Solution representation
    best_expression: str

    # Diversity metrics
    # population_diversity: float  # Ratio of unique trees
    # operator_distribution: dict  # Distribution of operators used

    # Complexity metrics
    avg_tree_size: np.float64  # Average nodes per tree
    avg_tree_depth: np.float64  # Average depth of trees
    # min_tree_size: int
    # max_tree_size: int

    # Performance metrics
    # eval_time: float  # Time spent on fitness evaluations
    # evolution_time: float  # Time spent on evolution operations


def mse(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    config: SymbolicConfig = SymbolicConfig.create(),
) -> np.float64:
    y_pred = expression.evaluate(x, config)
    return np.mean((y - y_pred) ** 2).astype(np.float64)


def r2(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    config: SymbolicConfig = SymbolicConfig.create(),
) -> np.float64:
    y_pred = expression.evaluate(x, config)
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)).astype(
        np.float64
    )


def calculate_score(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    config: SymbolicConfig = SymbolicConfig.create(),
) -> tuple[np.float64, ...]:
    return r2(expression, x, y, config), mse(expression, x, y, config)
