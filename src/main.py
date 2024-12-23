import logging
import os
import time
import winsound
from logging import Logger
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from symb_regression.config import GeneticParams
from symb_regression.core import GeneticProgram
from symb_regression.core.tree import Node
from symb_regression.operators.definitions import SymbolicConfig
from symb_regression.utils.data_handler import load_data, split_data
from symb_regression.utils.metrics import Metrics, calculate_score
from symb_regression.utils.plotting import (
    plot_evolution_metrics,
    plot_expression_tree,
    plot_prediction_analysis,
)
from symb_regression.utils.random import set_global_seed


def print_section_header(title: str, logger: Logger | None = None):
    if logger is not None:
        logger.info("=" * 50)
        logger.info(f" {title} ".center(50, "="))
        logger.info("=" * 50)
        return
    else:
        print("\n" + "=" * 50)
        print(f" {title} ".center(50, "="))
        print("=" * 50)


def print_section_footer(logger: Logger | None = None) -> None:
    if logger is not None:
        logger.info("=" * 50)
    else:
        print("=" * 50)


def run_symbolic_regression(
    x: np.ndarray,
    y: np.ndarray,
    params: GeneticParams | None = None,
    debug: bool = False,
    play_sound: bool = False,
) -> tuple[Node, List[Metrics]]:
    logger: Logger = logging.getLogger("symb_regression")

    if params is None:
        params = GeneticParams(
            tournament_size=7,
<<<<<<< HEAD:src/test.py
            mutation_prob=0.4,
            crossover_prob=0.7,
            elitism_count=10,
            population_size=1000,
            generations=300,
            max_depth=5,
            min_depth=2,
=======
            mutation_prob=0.6,
            crossover_prob=0.9,
            elitism_count=7,
            population_size=1000,
            generations=300,
            maximum_tree_depth=7,
            minimum_tree_depth=2,
            depth_penalty_threshold=5,  # Depth at which penalties start
            max_tree_size=15,
            size_penalty_threshold=5,  # Size at which penalties start
            parsimony_coefficient=0.1,  # Controls size penalty weight
>>>>>>> origin/odd-problems:src/main.py
        )

    if debug:
        logger.debug("Genetic Programming Parameters:")
        for key, value in params.__dict__.items():
            logger.debug(f"{key}: {value}")

    gp = GeneticProgram(params)

    logger.info("Starting evolution...")
    start_time = time.perf_counter()

    try:
        best_solution, history = gp.evolve(x, y)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if play_sound:
            # Play Windows default "SystemExclamation" sound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

        # Use print for better visibility of results
        print_section_header(f"SYMBOLIC REGRESSION RESULTS - {PROBLEM}")
        print(f"Best Expression Found: {best_solution}")
        print(f"Final Fitness: {gp.calculate_fitness(best_solution, x, y):g}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Generations: {len(history)}")
        print_section_footer()

        # Plot the evolution progress

        _, axs = plt.subplots(1, 2, figsize=(12, 6))
        plot_evolution_metrics(history, ax=axs[0])

        mse, r2 = calculate_score(best_solution, x, y)
        plot_evolution_metrics(history, ax=axs[0])
        plt.tight_layout()
        plt.show()

        print("Performance Metrics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"R² Score: {r2:.6f} ({r2*100:.1f}% of variance explained)")
        print_section_footer()
        # plot_expression_tree(best_solution)

        return best_solution, history

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


# Set random seed for reproducibility
# set_global_seed(42)

# Load and process data
PROBLEM_DIR = os.getcwd()
<<<<<<< HEAD:src/test.py
DATA_DIR = os.path.join(PROBLEM_DIR, "src/data")

x, y = load_data(DATA_DIR, "problem_2")

=======
DATA_DIR = os.path.join(PROBLEM_DIR, "data")
PROBLEM = "problem_3"
x, y = load_data(DATA_DIR, PROBLEM, show_stats=True)
x_train, x_val, y_train, y_val = split_data(x, y, train_size=0.1)
print("Done")
>>>>>>> origin/odd-problems:src/main.py
# Run symbolic regression
run_symbolic_regression(x, y, play_sound=True)
