import random
from typing import Tuple

from symb_regression.core.tree import Node


def crossover(parent1: Node, parent2: Node) -> Tuple[Node, Node]:
    child1, child2 = parent1.copy(), parent2.copy()

    nodes1: list[Node] = list(child1)[:1]  # Skip root node
    nodes2: list[Node] = list(child2)[:1]  # Skip root node

    if len(nodes1) > 1 and len(nodes2) > 1:
        # Select random nodes excluding root to maintain structure
        point1: Node = random.choice(nodes1)
        point2: Node = random.choice(nodes2)

        # Find parents of selected nodes
        parent1_node = next(
            (n for n in child1 if n.left is point1 or n.right is point1), None
        )
        parent2_node = next(
            (n for n in child2 if n.left is point2 or n.right is point2), None
        )

        if parent1_node and parent2_node:
            # Determine which child (left/right) to swap
            is_left1: bool = parent1_node.left is point1
            is_left2: bool = parent2_node.left is point2

            # Swap the nodes
            if is_left1:
                parent1_node.left = point2
            else:
                parent1_node.right = point2

            if is_left2:
                parent2_node.left = point1
            else:
                parent2_node.right = point1

    return child1, child2
