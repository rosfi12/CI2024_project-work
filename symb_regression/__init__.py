"""
# Symbolic regression package for genetic programming.

This package provides a genetic programming framework for symbolic regression.

The main class is `GeneticProgram`, which is used to evolve a population of
expression trees to fit a given dataset.

The package also provides utility functions for setting the global random seed.

Example:
    ```python
    from symb_regression import GeneticProgram, set_global_seed

    set_global_seed(42)

    gp = GeneticProgram()
    best_solution, history = gp.evolve(x, y)
    ```

Attributes:
    GeneticProgram: The main class for genetic programming.
    set_global_seed: Function to set the global random seed.




"""

from .core.genetic_programming import GeneticProgram
from .utils.random import set_global_seed

__all__: list[str] = [
    "GeneticProgram",
    "set_global_seed",
]
