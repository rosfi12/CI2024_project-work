import argparse
import json
import logging
import os
import time
from datetime import datetime

from symb_regression.config import GeneticParams
from symb_regression.core import GeneticProgram
from symb_regression.utils.data_handler import load_data


def get_problem_name() -> str:
    parser = argparse.ArgumentParser(
        description="Run symbolic regression on specified problem"
    )
    parser.add_argument(
        "problem_number", type=int, help="Problem number (e.g., 1 for problem_1)"
    )
    args = parser.parse_args()
    return f"problem_{args.problem_number}"


def continuous_regression(max_runs: int = 100, min_r2_threshold: float = 0.0):
    """Run symbolic regression continuously and save best results."""

    # Setup logging and directories
    logger = logging.getLogger("symb_regression")
    base_results_dir = os.path.join(os.getcwd(), "results")
    problem_dir = os.path.join(base_results_dir, PROBLEM)
    os.makedirs(problem_dir, exist_ok=True)

    # File to store all runs
    runs_file = os.path.join(
        problem_dir, f"continuous_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # Initialize best solution tracking
    best_r2 = min_r2_threshold
    best_solution = None
    best_metrics = None
    best_params = None

    # Genetic algorithm parameters
    params = GeneticParams(
        tournament_size=14,
        mutation_prob=0.9,
        crossover_prob=0.9,
        elitism_count=5,
        population_size=1000,
        generations=300,
        minimum_tree_depth=2,
        depth_penalty_threshold=5,
        maximum_tree_depth=10,
        size_penalty_threshold=7,
        max_tree_size=10,
        parsimony_coefficient=0.1,
        unused_var_coefficient=0.1,
        injection_diversity=0.4,
    )

    # Load data once
    x, y = load_data(os.path.join(os.getcwd(), "data"), PROBLEM, show_stats=False)

    for run in range(max_runs):
        try:
            # Initialize new GP instance
            gp = GeneticProgram(params)

            # Run evolution
            start_time = time.perf_counter()
            solution, history = gp.evolve(x, y, show_progress=False)
            execution_time = time.perf_counter() - start_time

            # Calculate fitness and metrics
            fitness, metrics = gp.calculate_fitness(solution, x, y)

            # Save run results to file
            with open(runs_file, "a") as f:
                f.write(f"\nRun {run + 1}/{max_runs}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Expression: {solution}\n")
                f.write(f"Fitness: {fitness:g}\n")
                f.write(f"MSE: {metrics['mse']:.6f}\n")
                f.write(f"R² Score: {metrics['r2']:.6f} ({metrics['r2']:.2%})\n")
                f.write(f"Execution Time: {execution_time:.2f}s\n")
                f.write(f"Generations: {len(history)}\n")
                f.write("=" * 50 + "\n")

            # Update best solution if better R² found
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_solution = str(solution)
                best_metrics = metrics
                best_params = params

                # Save best solution to separate file
                best_file = os.path.join(problem_dir, "best_solution.txt")
                with open(best_file, "w") as f:
                    f.write(f"Best Solution Found (Run {run + 1})\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Expression: {best_solution}\n")
                    f.write(f"R² Score: {best_r2:.6f} ({best_r2:.2%})\n")
                    f.write(f"MSE: {best_metrics['mse']:.6f}\n")
                    f.write("\nGenetic Algorithm Parameters:\n")
                    f.write(json.dumps(best_params.__dict__, indent=4))

                logger.info(f"New best solution found! R² = {best_r2:.2%}")

        except Exception as e:
            logger.error(f"Run {run + 1} failed: {e}")
            continue


if __name__ == "__main__":
    PROBLEM = get_problem_name()
    continuous_regression(
        max_runs=100, min_r2_threshold=0.5
    )  # Run 100 times, looking for R² > 0.5
