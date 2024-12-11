from dataclasses import dataclass


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
    max_depth: int = 6
    min_depth: int = 2

    def __post_init__(self):
        # Validate parameters
        if self.tournament_size > self.population_size:
            raise ValueError("Tournament size cannot exceed population size")
        if self.elitism_count > self.population_size:
            raise ValueError("Elitism count cannot exceed population size")
        if not (0 <= self.mutation_prob <= 1):
            raise ValueError("Mutation probability must be between 0 and 1")
        if not (0 <= self.crossover_prob <= 1):
            raise ValueError("Crossover probability must be between 0 and 1")
        if self.min_depth > self.max_depth:
            raise ValueError("Minimum depth cannot exceed maximum depth")
