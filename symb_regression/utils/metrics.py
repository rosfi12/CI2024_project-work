from typing import NamedTuple
import numpy as np


class Metrics(NamedTuple):
    # Generation tracking
    generation: int
    execution_time: float

    # Fitness metrics
    best_fitness: np.float64
    avg_fitness: np.float64
    worst_fitness: np.float64
    fitness_std: np.float64

    # Solution representation
    best_expression: str

    # Diversity metrics
    population_diversity: float  # Ratio of unique trees
    operator_distribution: dict  # Distribution of operators used

    # Complexity metrics
    avg_tree_size: np.float64  # Average nodes per tree
    avg_tree_depth: np.float64  # Average depth of trees
    min_tree_size: int
    max_tree_size: int

    # Performance metrics
    eval_time: float  # Time spent on fitness evaluations
    evolution_time: float  # Time spent on evolution operations
