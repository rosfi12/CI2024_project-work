import itertools
import json
import logging
import os
from datetime import datetime
from multiprocessing import Manager, Pool
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm.rich import tqdm

from symb_regression.config import GeneticParams
from symb_regression.core import GeneticProgram
from symb_regression.utils import set_global_seed
from symb_regression.utils.data_handler import load_data

# Set random seed for reproducibility
PROBLEM_DIR = os.getcwd()
DATA_DIR = os.path.join(PROBLEM_DIR, "data")


def save_results(
    results: Dict[str, Tuple[str, Dict[str, Any], float, float]],
    save_dir: str = "results",
):
    """Save optimization results to JSON file"""
    # Create results directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Format results for saving
    formatted_results = {}
    for pid, (_, params, fitness, time) in results.items():
        formatted_results[f"problem_{pid}"] = {
            "parameters": params,
            "fitness": float(fitness),
            "execution_time": float(time),
            "timestamp": datetime.now().isoformat(),
        }

    # Save to JSON file
    timestamp = datetime.now().isoformat()
    results_file = os.path.join(save_dir, f"hyperparameter_results_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(formatted_results, f, indent=4)

    print(f"\nResults saved to {results_file}")
    return results_file


def load_best_params(problem_id: str, results_file: str) -> Dict[str, Any]:
    """Load best parameters for a specific problem"""
    with open(results_file, "r") as f:
        results = json.load(f)
    return results[f"problem_{problem_id}"]["parameters"]


def evaluate_params(
    problem_id: str, params: Dict, x: np.ndarray, y: np.ndarray, n_runs: int = 1
) -> Tuple[str, Dict, float, float]:
    """Evaluate a parameter configuration multiple times and return average fitness"""

    # Temporarily disable logging
    logging.getLogger("symb_regression").setLevel(logging.ERROR)

    genetic_params = GeneticParams(
        tournament_size=params["tournament_size"],
        mutation_prob=params["mutation_prob"],
        crossover_prob=params["crossover_prob"],
        elitism_count=params["elitism_count"],
        population_size=params["population_size"],
        generations=params["generations"],
        max_depth=params["max_depth"],
        min_depth=params["min_depth"],
    )

    fitnesses = []
    times = []

    for _ in range(n_runs):
        gp = GeneticProgram(genetic_params)
        start = perf_counter()
        # Disable progress bar in evolve
        best_solution, _ = gp.evolve(x, y, show_progress=False)
        end = perf_counter()

        fitness = gp.calculate_fitness(best_solution, x, y)
        fitnesses.append(fitness)
        times.append(end - start)

    # Reset logging level
    logging.getLogger("symb_regression").setLevel(logging.INFO)

    avg_fitness = np.mean(fitnesses).astype(float)
    avg_time = np.mean(times).astype(float)

    return problem_id, params, avg_fitness, avg_time


def evaluate_params_with_progress(args) -> Tuple[str, Dict[str, Any], float, float]:
    """Wrapper function that handles progress updates"""
    problem_id, params, x, y, counter, total = args
    n_runs = 3  # Default number of runs per configuration
    result = evaluate_params(problem_id, params, x, y, n_runs)
    counter.value += 1
    return result


def optimize_parameters(problem_ids: List[str], save_dir: str = "results"):
    # Parameter grid setup
    param_grid: Dict[str, list[int | float]] = {
        "tournament_size": [3, 5, 7],
        "mutation_prob": [0.2, 0.4, 0.6],
        "crossover_prob": [0.7, 0.8, 0.9],
        "elitism_count": [5],
        "population_size": [200],
        "generations": [200],
        "max_depth": [4, 5, 6],
        "min_depth": [2],
    }

    # Generate combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    # Load problem data
    problem_data = {pid: load_data(DATA_DIR, f"problem_{pid}") for pid in problem_ids}

    # Setup multiprocessing manager
    with Manager() as manager:
        counter = manager.Value("i", 0)
        total_tasks = len(problem_ids) * len(param_combinations)

        # Create task arguments (removed n_runs from tuple)
        tasks = [
            (pid, params, *problem_data[pid], counter, total_tasks)
            for pid in problem_ids
            for params in param_combinations
        ]

        # Run parallel evaluation with progress bar
        with tqdm(total=total_tasks, desc="Optimizing parameters") as pbar:
            with Pool() as pool:
                results = []
                for result in pool.imap_unordered(evaluate_params_with_progress, tasks):
                    results.append(result)
                    pbar.update(1)

    best_results = process_results(results, problem_ids)

    # Save results
    _ = save_results(best_results, save_dir)

    return best_results


def process_results(
    results: List[Tuple[str, Dict[str, Any], float, float]], problem_ids: List[str]
) -> Dict[str, Tuple[str, Dict[str, Any], float, float]]:
    """Process and return best results for each problem"""
    best_results = {}
    for pid in problem_ids:
        problem_results = [r for r in results if r[0] == pid]
        best_idx = np.argmax([r[2] for r in problem_results])
        best_results[pid] = problem_results[best_idx]
    return best_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_global_seed(42)

    # Problems to optimize
    problem_ids: List[str] = ["1", "3", "5"]

    print("Starting parameter optimization...")
    print(f"Optimizing for problems: {problem_ids}")

    best_results = optimize_parameters(problem_ids)
