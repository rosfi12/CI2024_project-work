from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from symb_regression.core.tree import Node
from symb_regression.operators.definitions import SymbolicConfig
from symb_regression.utils.metrics import Metrics


def plot(x: np.ndarray, y: np.ndarray, best_solution: Node, history: List[Metrics]):
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_evolution_metrics(history, ax=axs[0])
    plot_prediction_analysis(best_solution, x, y, ax=axs[1])
    plt.tight_layout()
    plt.show()
    plot_expression_tree(best_solution)


def plot_evolution_metrics(metrics_history: List[Any], ax=None) -> None:
    """Plot metrics related to the evolution process."""
    generations = [m.generation for m in metrics_history]
    # max_gen = max(generations)
    best_fitness = [m.best_fitness for m in metrics_history]
    best_gen = generations[np.argmax(best_fitness)]
    # quarter_gens = [int(max_gen * x) for x in [0, 0.25, 0.5, 0.75, 1.0]]

    # fig = plt.figure(figsize=(15, 10))

    # 1. Fitness Evolution (top left)
    # plt = plt.subplot(221)
    if ax is None:
        ax = plt.gca()

    avg_fitness = [m.avg_fitness for m in metrics_history]
    worst_fitness = [m.worst_fitness for m in metrics_history]

    ax.plot(generations, best_fitness, "g-", label="Best", linewidth=2)
    ax.plot(generations, avg_fitness, "b-", label="Average", alpha=0.7)
    ax.plot(generations, worst_fitness, "r-", label="Worst", alpha=0.5)
    ax.axvline(x=best_gen, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (1 / (1 + MSE))")
    ax.set_title("Fitness Evolution")
    ax.legend()
    ax.grid(True)


def plot_prediction_analysis(
    expression: Node,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Expression Evaluation",
    ax=None,
    config: SymbolicConfig = SymbolicConfig.create(),
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
    y_pred = expression.evaluate(x, config)

    # Calculate metrics
    mse = np.mean((y - y_pred) ** 2).astype(np.float64)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)).astype(
        np.float64
    )

    # Create figure
    # fig, (plt, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    if ax is None:
        ax = plt.gca()

    # Plot predicted vs actual
    ax.scatter(y, y_pred, alpha=0.5, label="Predictions")

    # Plot perfect prediction line
    min_val = np.min((np.min(y), np.min(y_pred)))
    max_val = np.max((np.max(y), np.max(y_pred)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Perfect Prediction",
    )

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs True Values")
    ax.grid(True)
    ax.legend()

    # Add metrics text
    metrics_text = f"MSE: {mse:.6f}\nR²: {r2:.6f}"
    ax.text(
        4.05,
        1.50,
        metrics_text,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return mse, r2


def plot_expression_tree(root_node):
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    nodes_list = []

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

    def hierarchical_layout(
        graph, root=None, width=4.0, vert_gap=1.0, vert_loc=0, xcenter=0.5
    ):
        pos = {}  # dictionary to store node positions

        def _hierarchy_pos(
            G, root, width, vert_gap, vert_loc, xcenter, pos, parent=None, parsed=[]
        ):
            if root not in parsed:
                parsed.append(root)
                neighbors = list(G.neighbors(root))
                if not neighbors:  # leaf node
                    pos[root] = (xcenter, vert_loc)
                else:
                    dx = width / len(neighbors)
                    nextx = xcenter - width / 2 - dx / 2
                    for neighbor in neighbors:
                        nextx += dx
                        pos = _hierarchy_pos(
                            G,
                            neighbor,
                            width=dx,
                            vert_gap=vert_gap,
                            vert_loc=vert_loc - vert_gap,
                            xcenter=nextx,
                            pos=pos,
                            parent=root,
                            parsed=parsed,
                        )
                pos[root] = (xcenter, vert_loc)
            return pos

        return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter, pos)

    root_id = id(root_node)
    pos = hierarchical_layout(G, root=root_id)

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=500,
        node_color="#ADD8E6",
        edge_color="#708090",
        alpha=0.9,
    )

    # Add labels
    labels = {}
    for n in nodes_list:
        if n.op is not None:
            labels[id(n)] = str(n.op)
        elif n.value is not None:
            labels[id(n)] = str(n.value)
        else:
            labels[id(n)] = "None"

    nx.draw_networkx_labels(
        G, pos, labels, font_size=10, font_color="#4B0082", font_weight="bold"
    )
    plt.title("Expression Tree", fontsize=16, fontweight="bold")
    plt.show()


def plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    show_stats: bool = False,
) -> None:
    """Plot regression data with debug information."""

    if show_stats:
        print("Raw data inspection:")
        print(f"Y array shape: {y.shape}")
        print(f"Y actual min: {np.min(y)}")
        print(f"Y actual max: {np.max(y)}")
        print(f"Y sample values: {y[:5]}")  # Show first 5 values

    fig = plt.figure(figsize=(10, 8))

    if x.shape[1] == 2:
        ax = fig.add_subplot(111, projection="3d")

        # Plot raw data first without any transformations
        ax.scatter(x[:, 0], x[:, 1], y, c="blue", alpha=0.6, label="Data Points")

        # Print coordinates of points with y < -2 for verification
        extreme_points = y < -2
        if np.any(extreme_points):
            print("\nPoints with y < -2:")
            print("X1\tX2\tY")
            for x1, x2, y_val in zip(
                x[extreme_points, 0], x[extreme_points, 1], y[extreme_points]
            ):
                print(f"{x1:.3f}\t{x2:.3f}\t{y_val:.3f}")

        # Set axis labels
        ax.set_xlabel("X₁")
        ax.set_ylabel("X₂")
        ax.set_zlabel("Y")  # type: ignore

        # Explicitly set zlim based on actual data
        margin = (np.max(y) - np.min(y)) * 0.1
        ax.set_zlim(np.min(y) - margin, np.max(y) + margin)  # type: ignore

        plt.tight_layout()
        plt.show()


def plot_regression_data(
    x: np.ndarray,
    y: np.ndarray,
    best_solution: Optional[Node] = None,
    symb_config: SymbolicConfig = SymbolicConfig.create(),
) -> None:
    """Plot regression data with improved 3D visualization."""
    fig = plt.figure(figsize=(10, 8))

    match x.shape[1]:
        case 1:
            ax = fig.add_subplot(111)

            if best_solution is not None:
                # Plot actual data points
                scatter = ax.scatter(x, y, c="blue", alpha=0.6, label="Actual Data")

                # Plot prediction line
                x_range = np.linspace(min(x), max(x), 100)
                x_pred = x_range.reshape(-1, 1)
                y_pred = best_solution.evaluate(x_pred, symb_config)
                ax.plot(x_pred, y_pred, "r-", alpha=0.8, label="Prediction")

                ax.set_title("Regression Results with Predictions")
            else:
                scatter = ax.scatter(x, y, c="blue", alpha=0.6)
                ax.set_title("Data Distribution")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.legend()
        case 2:
            ax = fig.add_subplot(111, projection="3d")

            if best_solution is not None:
                # Plot actual data points first
                scatter = ax.scatter(
                    x[:, 0],
                    x[:, 1],
                    y,
                    c=np.abs(best_solution.evaluate(x, symb_config) - y)
                    / (np.abs(y) + 1e-10),
                    cmap="RdYlBu_r",
                    alpha=0.6,
                    label="Actual Data",
                )

                # Prediction surface
                x0_range = np.linspace(min(x[:, 0]), max(x[:, 0]), 50)
                x1_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 50)
                X0, X1 = np.meshgrid(x0_range, x1_range)
                X_pred = np.column_stack((X0.ravel(), X1.ravel()))
                Z = best_solution.evaluate(X_pred, symb_config).reshape(X0.shape)

                _ = ax.plot_surface(X0, X1, Z, alpha=0.3, cmap="viridis")  # type: ignore

                fig.colorbar(scatter, label="Relative Error")
                ax.set_title("Regression Results with Predictions")
            else:
                scatter = ax.scatter(
                    x[:, 0], x[:, 1], y, c=y, cmap="viridis", alpha=0.6
                )
                fig.colorbar(scatter, label="Target Values")
                ax.set_title("Data Distribution")

            # Set explicit z-axis limits based on actual data range
            z_min = min(y.min(), Z.min() if best_solution is not None else y.min())  # type: ignore
            z_max = max(y.max(), Z.max() if best_solution is not None else y.max())  # type: ignore
            z_padding = (z_max - z_min) * 0.1  # 10% padding
            ax.set_zlim(z_min - z_padding, z_max + z_padding)  # type: ignore

            ax.set_xlabel("X₁")
            ax.set_ylabel("X₂")
            ax.set_zlabel("Y")  # type: ignore
            ax.view_init(elev=20, azim=45)  # type: ignore

            # Add legend
            if best_solution is not None:
                ax.legend()

        case _:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    plt.tight_layout()
    plt.show()
