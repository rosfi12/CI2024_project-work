import random

from typing_extensions import Dict

from symb_regression.base import NodeType
from symb_regression.operators.definitions import BINARY_OPS, UNARY_OPS


def create_random_tree(
    depth: int,
    max_depth: int,
    n_variables: int,
    unary_weights: Dict[str, float] = None,
    binary_weights: Dict[str, float] = None,
) -> NodeType:
    from symb_regression.core.tree import Node
    from symb_regression.operators.definitions import get_var_name

    # Use default weights if none provided
    if unary_weights is None:
        unary_weights = {op: 1.0 / len(UNARY_OPS) for op in UNARY_OPS}
    if binary_weights is None:
        binary_weights = {op: 1.0 / len(BINARY_OPS) for op in BINARY_OPS}

    # Force at least one variable in the expression at root level
    if depth == 0:
        # Binary operator with at least one variable
        op = random.choices(
            list(binary_weights.keys()),
            weights=list(binary_weights.values()),
            k=1,
        )[0]
        node = Node(op=op)

        # Force one child to be a variable
        var_idx = random.randint(1, n_variables)
        if random.random() < 0.5:
            node.left = Node(op=get_var_name(var_idx))
            node.right = create_random_tree(
                depth + 1, max_depth, n_variables, unary_weights, binary_weights
            )
        else:
            node.left = create_random_tree(
                depth + 1, max_depth, n_variables, unary_weights, binary_weights
            )
            node.right = Node(op=get_var_name(var_idx))
        return node

    # At maximum depth, prefer variables over constants
    if depth >= max_depth:
        if random.random() < 0.7:  # 70% chance for variable
            var_idx = random.randint(1, n_variables)
            return Node(op=get_var_name(var_idx))
        else:
            return Node(value=random.uniform(-5, 5))

    # For intermediate depths
    if random.random() < 0.8:  # 80% chance for operator
        if random.random() < 0.3:  # 30% chance for unary operator
            op = random.choices(
                list(unary_weights.keys()),
                weights=list(unary_weights.values()),
                k=1,
            )[0]
            node = Node(op=op)
            node.left = create_random_tree(
                depth + 1, max_depth, n_variables, unary_weights, binary_weights
            )
        else:  # 70% chance for binary operator
            op = random.choices(
                list(binary_weights.keys()),
                weights=list(binary_weights.values()),
                k=1,
            )[0]
            node = Node(op=op)
            node.left = create_random_tree(
                depth + 1, max_depth, n_variables, unary_weights, binary_weights
            )
            node.right = create_random_tree(
                depth + 1, max_depth, n_variables, unary_weights, binary_weights
            )
        return node
    else:  # 20% chance for terminal
        if random.random() < 0.7:  # Higher chance for variable
            var_idx = random.randint(1, n_variables)
            return Node(op=get_var_name(var_idx))
        else:
            return Node(value=random.uniform(-5, 5))


def mutate(
    node: NodeType,
    mutation_prob: float,
    max_depth: int,
    n_variables: int,
    unary_weights: Dict[str, float],
    binary_weights: Dict[str, float],
) -> NodeType:
    from symb_regression.core.tree import Node

    if random.random() < mutation_prob:
        mutation_type = random.choices(
            ["subtree", "operator", "constant", "simplify"],
            weights=[0.2, 0.4, 0.2, 0.2],
        )[0]

        if mutation_type == "subtree":
            return create_random_tree(
                random.randint(1, 3),
                max_depth,
                n_variables,
                unary_weights,
                binary_weights,
            )

        elif mutation_type == "operator":
            if node.op in UNARY_OPS:
                node.op = random.choices(
                    list(unary_weights.keys()),
                    weights=list(unary_weights.values()),
                )[0]
            elif node.op in BINARY_OPS:
                node.op = random.choices(
                    list(binary_weights.keys()),
                    weights=list(binary_weights.values()),
                )[0]

        elif mutation_type == "constant":
            if node.value is not None:
                step = abs(node.value) * 0.1 if node.value != 0 else 0.1
                node.value += random.gauss(0, step)

        elif mutation_type == "simplify":
            if node.op == "*":
                if (node.left and node.left.value == 0) or (
                    node.right and node.right.value == 0
                ):
                    return Node(value=0)
                if node.left and node.left.value == 1:
                    return node.right.copy() if node.right else node
                if node.right and node.right.value == 1:
                    return node.left.copy() if node.left else node

    # Recursively mutate children with reduced probability
    if node.left:
        node.left = mutate(
            node.left,
            mutation_prob * 0.7,
            max_depth,
            n_variables,
            unary_weights,
            binary_weights,
        )
    if node.right:
        node.right = mutate(
            node.right,
            mutation_prob * 0.7,
            max_depth,
            n_variables,
            unary_weights,
            binary_weights,
        )

    return node
