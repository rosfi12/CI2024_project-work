import argparse
import logging
import os
import time
from logging import Logger
from typing import List

import numpy as np

from symb_regression import GeneticProgram, set_global_seed
from symb_regression.config import GeneticParams
from symb_regression.core.tree import Node
from symb_regression.utils.data_handler import load_data
from symb_regression.utils.logging_config import setup_logger
from symb_regression.utils.metrics import Metrics

# set_global_seed(42)


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
) -> tuple[Node, List[Metrics]]:
    logger: Logger = logging.getLogger("symb_regression")

    if params is None:
        params = GeneticParams(
            tournament_size=7,
            mutation_prob=0.4,
            crossover_prob=0.8,
            elitism_count=5,
            population_size=1000,
            generations=350,
            max_depth=5,
            min_depth=1,
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
        print_section_header("SYMBOLIC REGRESSION RESULTS")
        print(f"Best Expression Found: {best_solution}")
        print(f"Final Fitness: {gp.calculate_fitness(best_solution, x, y):g}")
        # print(f"Expression Size: {gp.count_nodes(best_solution)} nodes")
        # print(f"Expression Depth: {gp.get_depth(best_solution)} levels")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Generations: {len(history)}")
        print_section_footer()

        # Plot the evolution progress
        # plot_evolution_metrics(history)

        # FIXME operation value not shown in the expression tree
        # from symb_regression.utils.plotting import plot_expression_tree

        # plot_expression_tree(best_solution)

        print("\nAnalyzing solution...")
        # mse, r2 = plot_prediction_analysis(best_solution, x, y)
        # plot_variable_importance(best_solution, x, y)

        print("Performance Metrics:")
        # print(f"Mean Squared Error: {mse:.6f}")
        # print(f"R² Score: {r2:.6f} ({r2*100:.1f}% of variance explained)")

        return best_solution, history

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Symbolic Regression using Genetic Programming"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(debug=args.debug, log_file=args.log_file)
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load and process data
    PROBLEM_DIR = os.getcwd()
    DATA_DIR = os.path.join(PROBLEM_DIR, "data")

    try:
        logger.info("Loading data...")
        x, y = load_data(DATA_DIR, "problem_0")

        # Run symbolic regression
        run_symbolic_regression(x, y)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
