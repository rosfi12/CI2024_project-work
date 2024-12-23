from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


@dataclass
class GeneticParams:
    # Selection parameters
    tournament_size: int = 3
    elitism_count: int = 2

    # Evolution probabilities
    mutation_prob: float = 0.3
    crossover_prob: float = 0.9

    # Population parameters
    population_size: int = 200
    generations: int = 100

    # Tree constraints
    maximum_tree_depth: int = 6
    minimum_tree_depth: int = 2
    max_tree_size: int = 100

    parsimony_coefficient: float = 0.1  # Controls size penalty weight
    depth_penalty_threshold: int = 5  # Depth at which penalties start
    size_penalty_threshold: int = 50  # Size at which penalties start

    unused_var_coefficient: float = 0.1  # Coefficient for unused variable penalty

    def __post_init__(self) -> None:
        # Validate parameters
        if self.tournament_size > self.population_size:
            raise ValueError("Tournament size cannot exceed population size")
        if self.elitism_count > self.population_size:
            raise ValueError("Elitism count cannot exceed population size")
        if not (0 <= self.mutation_prob <= 1):
            raise ValueError("Mutation probability must be between 0 and 1")
        if not (0 <= self.crossover_prob <= 1):
            raise ValueError("Crossover probability must be between 0 and 1")
        if self.minimum_tree_depth > self.maximum_tree_depth:
            raise ValueError("Minimum depth cannot exceed maximum depth")
        if not (0 <= self.parsimony_coefficient <= 1):
            raise ValueError("Parsimony coefficient must be between 0 and 1")
        if self.depth_penalty_threshold > self.maximum_tree_depth:
            raise ValueError("Depth penalty threshold cannot exceed maximum depth")
        if self.size_penalty_threshold > self.max_tree_size:
            raise ValueError("Size penalty threshold cannot exceed maximum tree size")
        if not (0 <= self.unused_var_coefficient <= 1):
            raise ValueError("Unused variable coefficient must be between 0 and 1")


class MutationType(Enum):
    """Enum for different types of mutations"""

    SUBTREE = "subtree"
    OPERATOR = "operator"
    CONSTANT = "constant"
    SIMPLIFY = "simplify"


@dataclass(frozen=True)
class TreeConfig:
    """Configuration for tree creation"""

    # Node type probabilities
    OPERATOR_PROBABILITY: float = 0.8
    UNARY_OPERATOR_PROBABILITY: float = 0.3

    # Variable placement
    ROOT_VARIABLE_SIDE_PROBABILITY: float = 0.5
    VARIABLE_PROBABILITY: float = 0.7

    # Value ranges
    CONSTANT_RANGE: Tuple[float, float] = (-5.0, 5.0)

    # Tree structure
    MIN_DEPTH: int = 1
    MAX_DEPTH: int = 5
    DEFAULT_N_VARIABLES: int = 1


@dataclass(frozen=True)
class MutationConfig:
    """Configuration for mutation parameters"""

    MUTATION_DECAY = 0.9
    VALUE_STEP_FACTOR: float = 0.1
    MUTATION_WEIGHTS: Dict[MutationType, float] = field(
        default_factory=lambda: {
            MutationType.SUBTREE: 0.2,
            MutationType.OPERATOR: 0.4,
            MutationType.CONSTANT: 0.2,
            MutationType.SIMPLIFY: 0.2,
        }
    )
    SUBTREE_DEPTH_RANGE: Tuple[int, int] = (1, 3)
