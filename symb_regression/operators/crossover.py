import random
from typing import Tuple

from symb_regression.base import NodeType, get_nodes


def crossover(
    parent1: NodeType, parent2: NodeType
) -> Tuple[NodeType, NodeType]:
    child1, child2 = parent1.copy(), parent2.copy()
    nodes1 = get_nodes(child1)
    nodes2 = get_nodes(child2)

    if len(nodes1) > 1 and len(nodes2) > 1:
        # Select random nodes excluding root to maintain structure
        point1 = random.choice(nodes1[1:])  # Skip root node
        point2 = random.choice(nodes2[1:])  # Skip root node

        # Find parents of selected nodes
        parent1_node = next(
            (n for n in nodes1 if n.left is point1 or n.right is point1), None
        )
        parent2_node = next(
            (n for n in nodes2 if n.left is point2 or n.right is point2), None
        )

        if parent1_node and parent2_node:
            # Determine which child (left/right) to swap
            is_left1 = parent1_node.left is point1
            is_left2 = parent2_node.left is point2

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
