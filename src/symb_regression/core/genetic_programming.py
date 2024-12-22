import logging
import random
import time
import warnings
from collections import defaultdict
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from tqdm.std import TqdmExperimentalWarning

from symb_regression.config.settings import GeneticParams
from symb_regression.core.tree import Node
from symb_regression.operators.crossover import crossover
from symb_regression.operators.definitions import BINARY_OPS, UNARY_OPS, SymbolicConfig
from symb_regression.operators.mutation import create_random_tree, mutate
from symb_regression.utils.metrics import Metrics

logger: Logger = logging.getLogger("symb_regression")

# Suppress TQDM experimental warning when using rich progress bars
# 🌟Nice Progress Bar!🌟
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class GeneticProgram:
    def __init__(
        self,
        params: GeneticParams = GeneticParams(),
        config: SymbolicConfig = SymbolicConfig.create(),
    ) -> None:
        self.params: GeneticParams = params
        self.population: List[Node] = []
        self.best_solution: Optional[Node] = None
        self.metrics_history: List[Metrics] = []
        self.config: SymbolicConfig = config
        self.operator_success: defaultdict = defaultdict(
            lambda: {"uses": 0, "improvements": 0}
        )

    def calculate_population_diversity(self) -> float:
        """Calculate population diversity using expression structure."""
        unique_structures = set()
        for tree in self.population:
            structure = self.get_tree_structure(tree)
            unique_structures.add(structure)
        return len(unique_structures) / len(self.population)

    def get_tree_structure(
        self, node: Optional[Node]
    ) -> Dict[str, Union[str, int, float]]:
        """Get a dictionary representation of tree structure."""
        if node is None:
            return {}

        structure: Dict[str, Union[str, int, float]] = {
            "op": node.op if node.op else str(node.value),
            "depth": node.depth(),
            "size": len(node),
        }
        return structure

    def maintain_diversity(self, min_diversity: float = 0.3):
        """Maintain population diversity by replacing similar individuals."""
        current_diversity = self.calculate_population_diversity()
        if current_diversity < min_diversity:
            # Group similar individuals
            structures = defaultdict(list)
            for i, tree in enumerate(self.population):
                structures[self.get_tree_structure(tree)].append(i)

            # Replace excess similar individuals
            for indices in structures.values():
                if len(indices) > 2:  # Keep at most 2 of each structure
                    for idx in indices[2:]:
                        self.population[idx] = self.create_random_tree(
                            random.randint(
                                self.params.minimum_tree_depth,
                                self.params.maximum_tree_depth,
                            )
                        )

    def create_random_tree(self, depth: int) -> Node:
        return create_random_tree(
            depth, self.params.maximum_tree_depth, self.config.n_variables
        )

    def mutate(self, node: Node) -> Node:
        # Create default weights if no history yet
        unary_weights = {op: 1.0 / len(UNARY_OPS) for op in UNARY_OPS}
        binary_weights = {op: 1.0 / len(BINARY_OPS) for op in BINARY_OPS}

        return mutate(
            node=node,
            mutation_prob=self.params.mutation_prob,
            max_depth=self.params.maximum_tree_depth,
            n_variables=self.config.n_variables,
            unary_weights=unary_weights,
            binary_weights=binary_weights,
        )

    def calculate_fitness(
        self, tree: Node, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> np.float64:
        try:
            pred = tree.evaluate(x, self.config)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return np.float64(-np.inf)

            # Accuracy metrics with improved scaling
            mse: np.float64 = np.mean((pred - y) ** 2).astype(np.float64)
            rmse: np.float64 = np.sqrt(mse)

            # R² calculation with bounds
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - pred) ** 2)
            r2: np.float64 = np.maximum(0, 1 - (ss_res / ss_tot))

            # Tree complexity metrics
            tree_size = len(tree)
            depth = tree.depth()

            # Size penalty using configured thresholds
            if tree_size > self.params.size_penalty_threshold:
                size_penalty = (
                    np.exp(
                        (tree_size - self.params.size_penalty_threshold)
                        / self.params.max_tree_size
                    )
                    - 1
                )
            else:
                size_penalty = 0

            # Depth penalty using configured thresholds
            if depth > self.params.depth_penalty_threshold:
                depth_penalty = (
                    np.exp(
                        (depth - self.params.depth_penalty_threshold)
                        / self.params.maximum_tree_depth
                    )
                    - 1
                )
            else:
                depth_penalty = 0

            # Operator complexity with weighted penalties
            op_complexity = sum(2 if node.op in BINARY_OPS else 1 for node in tree)
            op_penalty = self.params.parsimony_coefficient * (op_complexity / tree_size)

            # Combined penalties with configurable weights
            total_penalty = (
                0.4 * size_penalty + 0.4 * depth_penalty + 0.2 * op_penalty
            ) * self.params.parsimony_coefficient

            # Accuracy score with improved balance
            accuracy_score = 0.7 * r2 + 0.3 / (1 + rmse)

            # Final fitness with adaptive penalty
            fitness = accuracy_score / (1 + total_penalty)

            return np.float64(np.clip(fitness, 0, 1))

        except (ValueError, RuntimeWarning, OverflowError):
            return np.float64(-np.inf)

    def tournament_selection(self, scores: List[np.float64]) -> Node:
        indices = np.array(
            random.sample(range(len(self.population)), self.params.tournament_size)
        )
        tournament_scores: npt.NDArray[np.float64] = np.array(scores)[indices]
        winner_idx = indices[np.argmax(tournament_scores)]
        return self.population[winner_idx]

    def evolve(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        show_progress: bool = True,
        collect_history: bool = True,
    ) -> Tuple[Node, List[Metrics]]:
        """
        Evolve the population to find a solution.

        Args:
            x: Input data
            y: Target values
            show_progress: Whether to show progress bar
            collect_history: Whether to collect history metrics

        Returns:
            Tuple of (best solution, evolution history)
        """
        self._initialize_evolution(x)
        self._create_initial_population()

        return self._run_evolution_loop(x, y, show_progress, collect_history)

    def _initialize_evolution(self, x: npt.NDArray[np.float64]) -> None:
        self.config.n_variables = x.shape[1] if x.ndim > 1 else 1
        logger.info("Initializing population...")
        self.best_solution = None

    def _create_initial_population(self) -> None:
        self.population = []

        while len(self.population) < self.params.population_size:
            try:
                tree = self.create_random_tree(3)
                if tree.validate():
                    self.population.append(tree)
            except Exception as e:
                logger.debug(f"Error creating tree: {e}")
                continue
            logger.debug(f"Current population size: {len(self.population)}")
        logger.debug(f"Initial population size: {len(self.population)}")

    def _evaluate_population(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Tuple[List[np.float64], List[Node], float]:
        eval_start = time.perf_counter()
        scores = []
        valid_population = []

        for tree in self.population:
            try:
                fitness = self.calculate_fitness(tree, x, y)
                if not np.isneginf(fitness):
                    scores.append(fitness)
                    valid_population.append(tree)
            except Exception as e:
                print(f"Error evaluating tree: {e}")
                continue

        eval_time = time.perf_counter() - eval_start
        return scores, valid_population, eval_time

    def _apply_elitism(
        self, scores: List[np.float64], new_population: List[Node]
    ) -> None:
        sorted_indices = np.argsort(scores)[::-1]
        for i in range(min(self.params.elitism_count, len(sorted_indices))):
            new_population.append(self.population[sorted_indices[i]].copy())

    def _create_offspring(
        self, scores: List[np.float64], new_population: List[Node]
    ) -> None:
        attempt_count = 0
        max_attempts = self.params.population_size * 2

        while (
            len(new_population) < self.params.population_size
            and attempt_count < max_attempts
        ):
            attempt_count += 1
            try:
                parent1 = self.tournament_selection(scores)
                parent2 = self.tournament_selection(scores)

                child1, child2 = self._crossover_and_mutate(parent1, parent2)

                if child1.validate():
                    new_population.append(child1)
                if (
                    len(new_population) < self.params.population_size
                    and child2.validate()
                ):
                    new_population.append(child2)
            except Exception as e:
                print(f"Error in evolution step: {e}")
                continue

    def _crossover_and_mutate(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        if random.random() < self.params.crossover_prob:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        if random.random() < self.params.mutation_prob:
            child1 = self.mutate(child1)
        if random.random() < self.params.mutation_prob:
            child2 = self.mutate(child2)

        return child1, child2

    def _inject_diversity(self) -> None:
        num_random = self.params.population_size // 4
        for _ in range(num_random):
            idx = random.randrange(self.params.elitism_count, len(self.population))
            self.population[idx] = self.create_random_tree(
                random.randint(2, self.params.maximum_tree_depth)
            )

    def _run_evolution_loop(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        show_progress: bool = True,
        collect_history: bool = True,
    ) -> Tuple[Node, List[Metrics]]:
        best_fitness: np.float64 = np.float64(-np.inf)
        generations_without_improvement = 0
        start_time = time.perf_counter()

        pbar = tqdm(
            range(self.params.generations),
            desc="Evolution progress",
            unit="gen",
            leave=True,
        )

        for gen in pbar:
            gen_start_time = time.perf_counter()
            logger.debug(f"Generation {gen + 1}/{self.params.generations}")

            scores, valid_population, eval_time = self._evaluate_population(x, y)
            self.population = valid_population

            if not scores:
                pbar.set_postfix_str("Reinitializing population...")
                continue

            best_fitness, generations_without_improvement = self._update_best_solution(
                scores,
                best_fitness,
                generations_without_improvement,
                pbar,
            )

            if generations_without_improvement > 10:
                pbar.write("No improvement for 10 generations, injecting diversity...")
                self._inject_diversity()
                generations_without_improvement = 0

            new_population: list[Node] = []
            self._apply_elitism(scores, new_population)
            self._create_offspring(scores, new_population)

            self.population = new_population[: self.params.population_size]

            if collect_history:
                self._update_metrics(gen, start_time, scores, gen_start_time, eval_time)

        if self.best_solution is None:
            raise ValueError("No solution found")

        return self.best_solution, self.metrics_history

    def _update_best_solution(
        self,
        scores: List[np.float64],
        best_fitness: np.float64,
        generations_without_improvement: int,
        pbar: tqdm,
    ) -> Tuple[np.float64, int]:
        current_best = np.max(scores)
        best_idx = scores.index(current_best)

        if current_best > best_fitness:
            best_fitness = current_best
            self.best_solution = self.population[best_idx].copy()
            generations_without_improvement = 0
            pbar.set_postfix_str(f"Best fitness: {best_fitness:.4f}")
            pbar.write(
                f"Best fitness: {best_fitness:.4f} - Expression: {self.best_solution}"
            )
        else:
            generations_without_improvement += 1

        return best_fitness, generations_without_improvement

    def _update_metrics(
        self,
        generation: int,
        start_time: float,
        scores: List[np.float64],
        gen_start_time: float,
        eval_time: float,
    ) -> None:
        # Fitness statistics
        fitness_array: npt.NDArray[np.float64] = np.array(scores)
        best_fitness: np.float64 = np.max(fitness_array)
        avg_fitness: np.float64 = np.mean(fitness_array, dtype=np.float64)
        worst_fitness: np.float64 = np.min(fitness_array)
        fitness_std: np.float64 = np.std(fitness_array, dtype=np.float64)

        # Tree statistics
        tree_sizes: List[int] = [len(tree) for tree in self.population]
        tree_depths: List[int] = [tree.depth() for tree in self.population]
        avg_tree_size: np.float64 = np.mean(tree_sizes, dtype=np.float64)
        avg_tree_depth: np.float64 = np.mean(tree_depths, dtype=np.float64)
        min_tree_size: int = min(tree_sizes)
        max_tree_size: int = max(tree_sizes)

        # Population diversity
        unique_expressions: int = len(set(str(tree) for tree in self.population))
        population_diversity: float = unique_expressions / len(self.population)

        # Operator distribution
        operator_counts: dict[str, int] = {}
        for tree in self.population:
            for node in tree:
                op_type: str = type(node).__name__
                operator_counts[op_type] = operator_counts.get(op_type, 0) + 1
        total_nodes = sum(operator_counts.values())
        operator_distribution = {
            op: count / total_nodes for op, count in operator_counts.items()
        }

        # Timing
        evolution_time = time.perf_counter() - gen_start_time

        # Best expression
        best_idx = scores.index(best_fitness)
        best_expression = str(self.population[best_idx])

        metrics = Metrics(
            generation=generation,
            execution_time=time.perf_counter() - start_time,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            worst_fitness=worst_fitness,
            fitness_std=fitness_std,
            best_expression=best_expression,
            population_diversity=population_diversity,
            operator_distribution=operator_distribution,
            avg_tree_size=avg_tree_size,
            avg_tree_depth=avg_tree_depth,
            min_tree_size=min_tree_size,
            max_tree_size=max_tree_size,
            eval_time=eval_time,
            evolution_time=evolution_time,
        )

        self.metrics_history.append(metrics)
