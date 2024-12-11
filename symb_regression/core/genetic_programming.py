import logging
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from symb_regression.base import get_nodes
from symb_regression.config.settings import GeneticParams
from symb_regression.core.tree import Node
from symb_regression.operators.crossover import crossover
from symb_regression.operators.definitions import BINARY_OPS, UNARY_OPS
from symb_regression.operators.mutation import create_random_tree, mutate
from symb_regression.utils.metrics import Metrics

logger = logging.getLogger("symb_regression")


class GeneticProgram:
    def __init__(self, params: GeneticParams = GeneticParams()):
        self.params = params
        self.population: List[Node] = []
        self.best_solution: Optional[Node] = None
        self.metrics_history: List[Metrics] = []
        self.n_variables: int = 1  # Will be set when evolve is called
        self.operator_success = defaultdict(
            lambda: {"uses": 0, "improvements": 0}
        )

    def update_operator_stats(self, tree: Node, improved: bool):
        """Track which operators lead to improvements."""
        for node in get_nodes(tree):
            if node.op:
                self.operator_success[node.op]["uses"] += 1
                if improved:
                    self.operator_success[node.op]["improvements"] += 1

    def get_operator_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights for operators based on their success."""
        weights = {}
        for op, stats in self.operator_success.items():
            if stats["uses"] > 0:
                success_rate = stats["improvements"] / stats["uses"]
                weights[op] = (
                    0.1 + 0.9 * success_rate
                )  # Base weight + success bonus
            else:
                weights[op] = 0.5  # Default weight for unused operators
        return weights

    def calculate_population_diversity(self) -> float:
        """Calculate population diversity using expression structure."""
        unique_structures = set()
        for tree in self.population:
            structure = self.get_tree_structure(tree)
            unique_structures.add(structure)
        return len(unique_structures) / len(self.population)

    def get_tree_structure(self, node: Node) -> str:
        """Get a string representation of tree structure."""
        if node is None:
            return ""
        if node.value is not None:
            return "C"
        if node.op and node.op.startswith("x"):
            return node.op
        return f"({node.op}{self.get_tree_structure(node.left)}{self.get_tree_structure(node.right)})"

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
                                self.params.min_depth, self.params.max_depth
                            )
                        )

    def create_random_tree(self, depth: int) -> Node:
        return create_random_tree(
            depth, self.params.max_depth, self.n_variables
        )

    def mutate(self, node: Node) -> Node:
        # Create default weights if no history yet
        unary_weights = {op: 1.0 / len(UNARY_OPS) for op in UNARY_OPS}
        binary_weights = {op: 1.0 / len(BINARY_OPS) for op in BINARY_OPS}

        return mutate(
            node=node,
            mutation_prob=self.params.mutation_prob,
            max_depth=self.params.max_depth,
            n_variables=self.n_variables,
            unary_weights=unary_weights,
            binary_weights=binary_weights,
        )

    def calculate_fitness(
        self, tree: Node, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> np.float64:
        try:
            pred = tree.evaluate(x)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return np.float64(-np.inf)

            mse = np.mean((pred - y) ** 2)

            # Count unique variables used
            used_vars = set()
            for node in get_nodes(tree):
                if node.op and node.op.startswith("x"):
                    used_vars.add(node.op)

            # Penalize not using all variables
            var_penalty = 0.5 * (self.n_variables - len(used_vars))

            # Other penalties
            if tree.op is None and tree.value is not None:
                complexity_penalty = 1.0
            else:
                complexity_penalty = 0.001 * len(tree.to_string())

            if self.count_nodes(tree) < 3:
                complexity_penalty += 0.5

            fitness = 1.0 / (1.0 + mse + complexity_penalty + var_penalty)
            return np.clip(fitness, 0, 1)

        except (ValueError, RuntimeWarning, OverflowError):
            return np.float64(-np.inf)

    def evolve(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Tuple[Node, List[Metrics]]:
        self.n_variables = x.shape[1] if x.ndim > 1 else 1
        logger.info("Initializing population...")
        start_time = time.perf_counter()

        # Initialize population
        self.population = []
        with tqdm(
            total=self.params.population_size,
            desc="Creating initial population",
        ) as pbar:
            while len(self.population) < self.params.population_size:
                try:
                    tree = self.create_random_tree(3)
                    # Verify the tree is valid before adding
                    if tree.validate():
                        self.population.append(tree)
                except Exception as e:
                    logger.debug(f"Error creating tree: {e}")
                    continue

        logger.debug(f"Initial population size: {len(self.population)}")

        self.best_solution = None
        best_fitness = -np.inf
        generations_without_improvement = 0

        # Main evolution loop
        with tqdm(
            range(self.params.generations),
            desc="Evolution progress",
            unit="gen",
            leave=True,
        ) as pbar:
            for gen in pbar:
                gen_start_time = time.perf_counter()

                # Print generation info
                logger.debug(f"Generation {gen + 1}/{self.params.generations}")

                # Evaluate population with error checking
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

                # Update population to only include valid trees
                self.population = valid_population

                if not scores:  # If no valid scores, reinitialize population
                    pbar.set_postfix_str("Reinitializing population...")
                    continue

                eval_time = time.perf_counter() - eval_start

                # Find best solution
                current_best_idx = np.argmax(scores)
                current_best_fitness = scores[current_best_idx]

                # Update best solution if improved
                if current_best_fitness > best_fitness:
                    self.update_operator_stats(
                        self.population[current_best_idx], True
                    )
                    best_fitness = current_best_fitness
                    best_expression = self.population[current_best_idx]

                    # Only update if it's a non-trivial expression
                    if best_expression.op is not None or (
                        best_expression.value is not None
                        and abs(best_expression.value) > 1e-10
                    ):
                        self.best_solution = best_expression.copy()
                        best_expression_str = (
                            self.best_solution.to_pretty_string()
                        )
                        generations_without_improvement = 0
                        pbar.set_postfix(
                            {
                                "best_fitness": f"{best_fitness:.6f}",
                                "avg_fitness": f"{np.mean(scores):.6f}",
                                "diversity": f"{len(set(tree.to_string() for tree in self.population))/len(self.population):.2f}",
                                "best_expr": best_expression_str[:30]
                                + (
                                    "..."
                                    if len(best_expression_str) > 30
                                    else ""
                                ),
                            }
                        )
                else:
                    self.update_operator_stats(
                        self.population[current_best_idx], False
                    )
                    generations_without_improvement += 1

                # If no improvement for too long, inject diversity
                if generations_without_improvement > 10:
                    pbar.write(
                        "No improvement for 10 generations, injecting diversity..."
                    )
                    num_random = self.params.population_size // 4
                    for _ in range(num_random):
                        idx = random.randrange(
                            self.params.elitism_count, len(self.population)
                        )
                        self.population[idx] = self.create_random_tree(
                            random.randint(2, self.params.max_depth)
                        )
                    generations_without_improvement = 0

                # Create new population
                new_population = []

                # Elitism
                sorted_indices = np.argsort(scores)[::-1]
                for i in range(
                    min(self.params.elitism_count, len(sorted_indices))
                ):
                    new_population.append(
                        self.population[sorted_indices[i]].copy()
                    )

                # Generate offspring
                attempt_count = 0
                max_attempts = self.params.population_size * 2

                while (
                    len(new_population) < self.params.population_size
                    and attempt_count < max_attempts
                ):
                    attempt_count += 1
                    try:
                        # Tournament selection
                        parent1 = self.tournament_selection(scores)
                        parent2 = self.tournament_selection(scores)

                        # Crossover
                        if random.random() < self.params.crossover_prob:
                            child1, child2 = crossover(parent1, parent2)
                        else:
                            child1, child2 = parent1.copy(), parent2.copy()

                        # Mutation
                        if random.random() < self.params.mutation_prob:
                            child1 = self.mutate(
                                child1,
                            )
                        if random.random() < self.params.mutation_prob:
                            child2 = self.mutate(
                                child2,
                            )

                        # Validate children before adding
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

                self.maintain_diversity()
                logger.debug(f"New population size: {len(new_population)}")

                # Ensure minimum population size
                while len(new_population) < self.params.population_size:
                    try:
                        new_tree = self.create_random_tree(3)
                        if new_tree.validate():
                            new_population.append(new_tree)
                    except Exception as e:
                        print(f"Error creating new tree: {e}")
                        continue

                # Update population
                self.population = new_population[: self.params.population_size]

                # Track metrics
                self.metrics_history.append(
                    Metrics(
                        generation=gen,
                        execution_time=time.perf_counter() - start_time,
                        best_fitness=current_best_fitness,
                        avg_fitness=np.mean(scores),
                        worst_fitness=np.min(scores),
                        fitness_std=np.std(scores),
                        best_expression=self.population[
                            current_best_idx
                        ].to_string(),
                        population_diversity=len(
                            set(tree.to_string() for tree in self.population)
                        )
                        / len(self.population),
                        operator_distribution=self.get_operator_distribution(
                            self.population
                        ),
                        avg_tree_size=np.mean(
                            [self.count_nodes(tree) for tree in self.population]
                        ),
                        avg_tree_depth=np.mean(
                            [self.get_depth(tree) for tree in self.population]
                        ),
                        min_tree_size=min(
                            self.count_nodes(tree) for tree in self.population
                        ),
                        max_tree_size=max(
                            self.count_nodes(tree) for tree in self.population
                        ),
                        eval_time=eval_time,
                        evolution_time=time.perf_counter() - gen_start_time,
                    )
                )

        if self.best_solution:
            pbar.write(
                f"\nFinal best solution: {self.best_solution.to_pretty_string()}"
            )
            pbar.write(f"Final fitness: {best_fitness:.6f}")

        if self.best_solution is None:
            raise ValueError("No solution found")

        return self.best_solution, self.metrics_history

    def count_nodes(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def get_depth(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def get_operator_distribution(
        self, population: List[Node]
    ) -> Dict[str, int]:
        distribution = defaultdict(int)
        for tree in population:

            def count_ops(node: Node):
                if node.op:
                    distribution[node.op] += 1
                if node.left:
                    count_ops(node.left)
                if node.right:
                    count_ops(node.right)

            count_ops(tree)
        return dict(distribution)

    def tournament_selection(self, scores: List[float]) -> Node:
        indices = random.sample(
            range(len(self.population)), self.params.tournament_size
        )
        winner_idx = max(indices, key=lambda i: scores[i])
        return self.population[winner_idx]
