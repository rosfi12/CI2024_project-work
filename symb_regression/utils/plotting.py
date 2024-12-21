from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from symb_regression.core.tree import Node


def plot_evolution_metrics(metrics_history: List[Any]) -> None:
    """Plot metrics related to the evolution process."""
    generations = [m.generation for m in metrics_history]
    max_gen = max(generations)
    best_fitness = [m.best_fitness for m in metrics_history]
    best_gen = generations[np.argmax(best_fitness)]
    quarter_gens = [int(max_gen * x) for x in [0, 0.25, 0.5, 0.75, 1.0]]

    fig = plt.figure(figsize=(15, 10))

    # 1. Fitness Evolution (top left)
    ax1 = plt.subplot(221)
    avg_fitness = [m.avg_fitness for m in metrics_history]
    worst_fitness = [m.worst_fitness for m in metrics_history]

    ax1.plot(generations, best_fitness, "g-", label="Best", linewidth=2)
    ax1.plot(generations, avg_fitness, "b-", label="Average", alpha=0.7)
    ax1.plot(generations, worst_fitness, "r-", label="Worst", alpha=0.5)
    ax1.axvline(x=best_gen, color="green", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (1 / (1 + MSE))")
    ax1.set_title("Fitness Evolution")
    ax1.legend()
    ax1.grid(True)

    # 2. Population Diversity (top right)
    ax2 = plt.subplot(222)
    diversity = [m.population_diversity for m in metrics_history]
    ax2.plot(generations, diversity, "b-", linewidth=2)
    for gen in quarter_gens:
        ax2.axvline(x=gen, color="gray", linestyle="--", alpha=0.3)
    ax2.axvline(x=best_gen, color="green", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity Ratio")
    ax2.set_title("Population Diversity")
    ax2.grid(True)

    # 3. Tree Complexity (bottom left)
    ax3 = plt.subplot(223)
    avg_size = [m.avg_tree_size for m in metrics_history]
    min_size = [m.min_tree_size for m in metrics_history]
    max_size = [m.max_tree_size for m in metrics_history]

    ax3.plot(generations, avg_size, "b-", label="Average", linewidth=2)
    ax3.fill_between(generations, min_size, max_size, alpha=0.2, color="blue")
    ax3.axvline(x=best_gen, color="green", linestyle="--", alpha=0.5)

    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Tree Size (nodes)")
    ax3.set_title("Expression Complexity")
    ax3.grid(True)

    # 4. Operator Distribution (bottom right)
    ax4 = plt.subplot(224)
    final_ops = metrics_history[-1].operator_distribution
    plot_operator_distribution(ax4, final_ops)

    # Add summary text
    summary_text = (
        f"Best solution at gen {best_gen} ({best_gen/max_gen*100:.1f}%)\n"
        f"Final fitness: {max(best_fitness):.6f}\n"
        f"Final diversity: {diversity[-1]:.2f}"
    )
    fig.text(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_operator_distribution(ax: Axes, operator_dist: dict) -> None:
    """Plot operator distribution on given axes."""
    ops = list(operator_dist.keys())
    frequencies = list(operator_dist.values())

    # Sort and filter operators
    sorted_indices = np.argsort(frequencies)[::-1]
    top_n = 6
    if len(ops) > top_n:
        top_ops = [ops[i] for i in sorted_indices[:top_n]]
        top_freqs = [frequencies[i] for i in sorted_indices[:top_n]]
        other_freq = sum(frequencies[i] for i in sorted_indices[top_n:])

        ops = top_ops + ["Others"]
        frequencies = top_freqs + [other_freq]
    else:
        ops = [ops[i] for i in sorted_indices]
        frequencies = [frequencies[i] for i in sorted_indices]

    # Create color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))  # type: ignore
    if "Others" in ops:
        colors[-1] = (0.7, 0.7, 0.7, 1.0)

    # Create bar plot
    bars = ax.barh(range(len(ops)), frequencies, color=colors)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops)

    # Add percentage labels
    total = sum(frequencies)
    for bar in bars:
        width = bar.get_width()
        percentage = (width / total) * 100
        if percentage >= 5:
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:.1f}%",
                ha="left",
                va="center",
                fontsize=8,
            )

    ax.set_xlabel("Frequency")
    ax.set_title("Most Used Operators")


def plot_prediction_analysis(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Expression Evaluation",
) -> Tuple[np.float64, np.float64]:
    """
    Evaluate a symbolic expression against dataset and visualize results.

    Args:
        expression: The symbolic expression to evaluate
        x: Input data
        y: True output data
        title: Title for the plot

    Returns:
        Tuple containing (mse, r2_score)
    """
    # Calculate predictions
    y_pred = expression.evaluate(x)

    # Calculate metrics
    mse = np.mean((y - y_pred) ** 2).astype(np.float64)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)).astype(
        np.float64
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot predicted vs actual
    ax1.scatter(y, y_pred, alpha=0.5, label="Predictions")

    # Plot perfect prediction line
    min_val = np.min((np.min(y), np.min(y_pred)))
    max_val = np.max((np.max(y), np.max(y_pred)))
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Perfect Prediction",
    )

    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title("Predicted vs True Values")
    ax1.grid(True)
    ax1.legend()

    # Add metrics text
    metrics_text = f"MSE: {mse:.6f}\nR²: {r2:.6f}"
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot residuals
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color="r", linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    ax2.grid(True)

    # Add residuals statistics
    resid_stats = (
        f"Mean: {np.mean(residuals):.6f}\n"
        f"Std: {np.std(residuals):.6f}\n"
        f"Max: {np.max(np.abs(residuals)):.6f}"
    )
    ax2.text(
        0.05,
        0.95,
        resid_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return mse, r2


def plot_variable_importance(
    expression: Node, x: np.ndarray, y: np.ndarray, n_samples: int = 1000
) -> None:
    """
    Analyze and visualize the importance of each variable.

    Args:
        expression: The symbolic expression to analyze
        x: Input data
        y: True output data
        n_samples: Number of samples for sensitivity analysis
    """
    n_vars = x.shape[1] if x.ndim > 1 else 1
    base_pred = expression.evaluate(x)
    sensitivities = []

    # Perform sensitivity analysis
    for i in range(n_vars):
        x_perturbed = x.copy()
        if x.ndim > 1:
            std = np.std(x[:, i])
            x_perturbed[:, i] += np.random.normal(0, std, size=len(x))
        else:
            std = np.std(x)
            x_perturbed += np.random.normal(0, std, size=len(x))

        perturbed_pred = expression.evaluate(x_perturbed)
        sensitivity = np.mean(np.abs(perturbed_pred - base_pred))
        sensitivities.append(sensitivity)

    # Plot variable importance
    plt.figure(figsize=(10, 5))
    var_names = [f"x{i}" for i in range(n_vars)]
    plt.bar(var_names, sensitivities)
    plt.title("Variable Importance Analysis")
    plt.xlabel("Variables")
    plt.ylabel("Sensitivity")
    plt.grid(True, alpha=0.3)

    # Add percentage labels
    total = sum(sensitivities)
    for i, v in enumerate(sensitivities):
        plt.text(i, v, f"{(v/total)*100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def visualize_expression_behavior(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    variable_idx: int = 0,
    n_points: int = 100,
) -> None:
    """
    Visualize how the expression behaves across the range of a specific variable.

    Args:
        expression: The symbolic expression to analyze
        x: Input data
        y: True output data
        variable_idx: Index of the variable to analyze
        n_points: Number of points for visualization
    """
    if x.ndim > 1:
        x_var = x[:, variable_idx]
    else:
        x_var = x

    # Create range of values for the selected variable
    x_range = np.linspace(np.min(x_var), np.max(x_var), n_points)

    # Create input data for prediction
    if x.ndim > 1:
        x_pred = np.tile(np.mean(x, axis=0), (n_points, 1))
        x_pred[:, variable_idx] = x_range
    else:
        x_pred = x_range

    # Calculate predictions
    y_pred = expression.evaluate(x_pred)

    # Plot
    plt.figure(figsize=(12, 6))

    # Plot actual data points
    plt.scatter(x_var, y, alpha=0.5, label="Actual Data", color="blue")

    # Plot expression behavior
    plt.plot(x_range, y_pred, "r-", label="Expression", linewidth=2)

    plt.xlabel(f"Variable x{variable_idx}")
    plt.ylabel("Output")
    plt.title(f"Expression Behavior vs Variable x{variable_idx}")
    plt.legend()
    plt.grid(True)

    # Add expression text
    plt.text(
        0.02,
        0.98,
        f"Expression: {expression}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_expression_tree(root_node: Node) -> None:
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout

    G = nx.DiGraph()
    nodes_list: list[Node] = []

    def collect_nodes(node, parent_id=None):
        current_id = id(node)
        nodes_list.append(node)

        if parent_id is not None:
            G.add_edge(parent_id, current_id)

        if node.left:
            collect_nodes(node.left, current_id)
        if node.right:
            collect_nodes(node.right, current_id)

    collect_nodes(root_node)

    pos = graphviz_layout(G, prog="dot")

    # Draw operator nodes
    operator_nodes = [
        id(n)
        for n in nodes_list
        if isinstance(n.value, str) and n.value in ["+", "-", "*", "/"]
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=operator_nodes,
        node_size=800,
        node_color="lightpink",
        node_shape="o",
    )

    # Draw variable nodes
    variable_nodes = [
        id(n)
        for n in nodes_list
        if isinstance(n.value, str) and n.value.startswith("x")
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=variable_nodes,
        node_size=500,
        node_color="lightgreen",
        node_shape="s",
    )

    # Draw constant nodes
    constant_nodes = [id(n) for n in nodes_list if isinstance(n.value, (int, float))]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=constant_nodes,
        node_size=500,
        node_color="lightblue",
        node_shape="s",
    )

    # Add labels
    labels = {
        id(n): str(n.value) if isinstance(n.value, (int, float)) else n.value
        for n in nodes_list
    }
    nx.draw_networkx_labels(G, pos, labels)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    plt.axis("off")
    plt.show()
