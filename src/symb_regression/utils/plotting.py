from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from symb_regression.core.tree import Node
from symb_regression.operators.definitions import SymbolicConfig


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
            labels[id(n)] = n.op
        elif n.value is not None:
            labels[id(n)] = str(n.value)
        else:
            labels[id(n)] = "None"

    nx.draw_networkx_labels(
        G, pos, labels, font_size=10, font_color="#4B0082", font_weight="bold"
    )
    plt.title("Expression Tree", fontsize=16, fontweight="bold")
    plt.show()


def plot_3d_data(
    x: np.ndarray,
    y: np.ndarray,
    best_solution: Optional[Node] = None,
    config: SymbolicConfig = SymbolicConfig.create(),
) -> None:
    """
    Create a 3D visualization of the regression problem.

    Args:
        x: Input data (n_samples, 2)
        y: Target values (n_samples,)
        best_solution: Optional best solution found
        ax: Optional axes to plot on
        config: Symbolic configuration used
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of actual data points
    scatter = ax.scatter(
        x[:, 0], x[:, 1], y, c=y, cmap="viridis", alpha=0.6, label="Actual Data"
    )

    # If we have a solution, plot the predicted surface
    if best_solution is not None:
        # Create a meshgrid for surface plotting
        x0_range = np.linspace(min(x[:, 0]), max(x[:, 0]), 50)
        x1_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 50)
        X0, X1 = np.meshgrid(x0_range, x1_range)

        # Prepare input for prediction
        X_pred = np.column_stack((X0.ravel(), X1.ravel()))

        # Get predictions
        Y_pred = best_solution.evaluate(X_pred, config)
        Z = Y_pred.reshape(X0.shape)

        # Plot prediction surface
        _ = ax.plot_surface(
            X0, X1, Z, cmap="viridis", alpha=0.3, label="Predicted Surface"
        )

    # Customize the plot
    ax.set_xlabel("X₁")
    ax.set_ylabel("X₂")
    ax.set_zlabel("Y")
    ax.set_title("3D Visualization of Regression Problem")

    # Add colorbar
    fig.colorbar(scatter, label="Target Values")

    # Adjust the view
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    plt.show()
    return None
