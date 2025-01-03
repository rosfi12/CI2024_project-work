import json
import logging
import os
import time
import winsound
from datetime import datetime
from logging import Logger
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from symb_regression.config import GeneticParams
from symb_regression.core import GeneticProgram
from symb_regression.core.tree import Node
from symb_regression.utils.data_handler import load_data, sort_and_filter_data
from symb_regression.utils.metrics import Metrics, calculate_score
from symb_regression.utils.plotting import (
    plot_3d,
<<<<<<< HEAD
    plot_evolution_metrics,
    plot_expression_tree,
    plot_prediction_analysis,
=======
    plot_regression_data,
>>>>>>> 1f2aa69c018bd9c1975076bd6776ea0c8ec6c8fe
)


def save_and_print(message: str, file_handle) -> None:
    """Print message to both console and file"""
    print(message)
    file_handle.write(message + "\n")


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

    # Create results directory if it doesn't exist
    base_results_dir = os.path.join(os.getcwd(), "results")
    problem_dir = os.path.join(base_results_dir, PROBLEM)
    os.makedirs(problem_dir, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(problem_dir, f"results_{timestamp}.txt")

    if params is None:
        params = GeneticParams(
            tournament_size=7,
            mutation_prob=0.9,
            crossover_prob=0.9,
            elitism_count=5,
            population_size=1000,
            generations=300,
            minimum_tree_depth=3,
            depth_penalty_threshold=7,  # Depth at which penalties start
            maximum_tree_depth=10,
            size_penalty_threshold=10,  # Size at which penalties start
            max_tree_size=20,
            parsimony_coefficient=0.9,
            unused_var_coefficient=0.4,  # Coefficient for unused variable penalty
            injection_diversity=0.9,  # Coefficient for injection diversity
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
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

        fitness, metrics = gp.calculate_fitness(best_solution, x, y)

        # Write results to both console and file
        with open(results_file, "w") as f:
            save_and_print(f"SYMBOLIC REGRESSION RESULTS - {PROBLEM}", f)
            save_and_print("=" * 50, f)
            save_and_print(f"Best Expression Found: {best_solution}", f)
            save_and_print("\nPerformance Metrics:", f)
            save_and_print(f"Final Fitness: {fitness:g}", f)
            save_and_print(f"Mean Squared Error: {metrics['mse']:.6f}", f)
            save_and_print(
                f"R² Score: {metrics['r2']:.6f} ({metrics['r2']:.2%} of variance explained)",
                f,
            )
            save_and_print(f"Execution Time: {execution_time:.2f} seconds", f)
            save_and_print(f"Generations: {len(history)}", f)
            save_and_print("\nGenetic Algorithm Parameters:", f)
            save_and_print(json.dumps(params.__dict__, indent=4), f)
            save_and_print("=" * 50, f)

        logger.info(f"Results saved to: {results_file}")

        # Plot the evolution progress
        plot(x, y, best_solution, history)

        plot_regression_data(x, y, best_solution)
        return best_solution, history

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_global_seed(42)

    # Load and process data
    PROBLEM_DIR = os.getcwd()
    DATA_DIR = os.path.join(PROBLEM_DIR, "data")
    PROBLEM = "problem_1"
    x, y = load_data(DATA_DIR, PROBLEM, show_stats=True)

    plot_3d(x, y)

<<<<<<< HEAD
# raise
# Run symbolic regression
run_symbolic_regression(x[:, 1].reshape(-1, 1), y, play_sound=True)

=======
    # Run symbolic regression
    run_symbolic_regression(x, y, play_sound=True)
>>>>>>> 1f2aa69c018bd9c1975076bd6776ea0c8ec6c8fe
