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
from symb_regression.utils.data_handler import load_data
from symb_regression.utils.metrics import Metrics
from symb_regression.utils.plotting import (
    plot_evolution_metrics,
    plot_expression_tree,
    plot_prediction_analysis,
)


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


def print_section_footer(logger: Logger | None = None):
    if logger is not None:
        logger.info("=" * 50 + "\n")
    else:
        print("=" * 50 + "\n")


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
            mutation_prob=0.4,
            crossover_prob=0.8,
            elitism_count=5,
            population_size=1000,
            generations=300,
            max_depth=4,
            min_depth=2,
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

        mse, r2 = plot_prediction_analysis(best_solution, x, y, ax=axs[1])

        plt.tight_layout()
        plt.show()

        print("Performance Metrics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"R² Score: {r2:.6f} ({r2*100:.1f}% of variance explained)")

        # plot_expression_tree(best_solution)
        if play_sound:
            # Play Windows default "SystemExclamation" sound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

        return best_solution, history

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


# Set random seed for reproducibility

# Load and process data
PROBLEM_DIR = os.getcwd()
DATA_DIR = os.path.join(PROBLEM_DIR, "data")
PROBLEM = "problem_1"
x, y = load_data(DATA_DIR, PROBLEM)


# Run symbolic regression
run_symbolic_regression(x, y, play_sound=True)
