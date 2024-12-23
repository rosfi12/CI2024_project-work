import itertools
import json
import os
from datetime import datetime
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import toml
from tqdm.rich import tqdm

from symb_regression.config import GeneticParams
from symb_regression.core import GeneticProgram
from symb_regression.utils import set_global_seed
from symb_regression.utils.data_handler import load_data, split_data

# Set random seed for reproducibility
PROBLEM_DIR = os.getcwd()
DATA_DIR = os.path.join(PROBLEM_DIR, "data")


def save_results(
    results: Dict[str, Tuple[str, Dict[str, Any], float, str]],
    save_dir: str = "results",
):
    """Save optimization results to JSON and print to stdout"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Print results to stdout first
    print("\n=== Optimization Results ===")
    formatted_results = {}
    for pid, (_, params, fitness, expression) in results.items():
        print(f"\nProblem {pid}:")
        print(f"Best fitness: {fitness:.4f}")
        print(f"Best expression: {expression}")
        print("Parameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")

        formatted_results[f"problem_{pid}"] = {
            "parameters": params,
            "fitness": float(fitness),
            "best_expression": expression,
            "timestamp": datetime.now().isoformat(),
        }

    # Save to TOML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"hyperparameter_results_{timestamp}.toml")

    with open(results_file, "w") as f:
        toml.dump(formatted_results, f)

    print(f"\nResults saved to {results_file}")
    return results_file


def load_best_params(problem_id: str, results_file: str) -> Dict[str, Any]:
    """Load best parameters for a specific problem"""
    with open(results_file, "r") as f:
        results = json.load(f)
    return results[f"problem_{problem_id}"]["parameters"]


def evaluate_params(
    problem_id: str, params: Dict, x: np.ndarray, y: np.ndarray, n_runs: int = 3
) -> Tuple[str, Dict, float, str]:
    genetic_params = GeneticParams(**params)

    fitnesses = []
    best_expression = ""
    best_fitness = float("-inf")

    for _ in range(n_runs):
        gp = GeneticProgram(genetic_params)
        best_solution, _ = gp.evolve(x, y, show_progress=False, collect_history=False)
        fitness = gp.calculate_fitness(best_solution, x, y)

        if fitness > best_fitness:
            best_fitness = fitness
            best_expression = str(best_solution)

        fitnesses.append(fitness)

    avg_fitness = np.mean(fitnesses).astype(float)
    return problem_id, params, avg_fitness, best_expression


def evaluate_params_with_progress(args) -> Tuple[str, Dict, float, str]:
    """Wrapper function that handles progress updates"""
    problem_id, params, x_train, x_val, y_train, y_val, counter, total = args
    result = evaluate_params(problem_id, params, x_train, y_train)
    counter.value += 1
    return result


def optimize_parameters(
    problem_ids: List[str], save_dir: str = "results", train_ratio: float = 0.01
):
    # Parameter grid setup
    param_grid: Dict[str, list[int | float]] = {
        "tournament_size": [5, 7, 8],
        "mutation_prob": [0.4, 0.6, 0.8],
        "crossover_prob": [0.8, 0.9],
        "elitism_count": [2, 4],
        "population_size": [2000],
        "generations": [400],
        "max_depth": [
            5,
            6,
        ],
        "min_depth": [1],
    }

    # Generate combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    # Load problem data
    problem_data = {}
    for pid in problem_ids:
        x, y = load_data(DATA_DIR, f"problem_{pid}", show_stats=True)
        x_train, x_val, y_train, y_val = split_data(x, y, train_size=train_ratio)
        problem_data[pid] = (x_train, x_val, y_train, y_val)

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
    results: Tuple[str, Dict, float, str], problem_ids: List[str]
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
    problem_ids: List[str] = ["3"]

    print("Starting parameter optimization...")
    print(f"Optimizing for problems: {problem_ids}")

    best_results = optimize_parameters(problem_ids)
