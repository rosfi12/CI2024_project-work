from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from symb_regression.base import INode
from symb_regression.operators.definitions import (
    BINARY_OPS,
    OPERATOR_PRECEDENCE,
    UNARY_OPS,
)


class Node(INode):
    def __init__(self, op: Optional[str] = None, value: Optional[float] = None):
        self.op = op
        self.value = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def evaluate(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.value is not None:
            return np.full(x.shape[1] if x.ndim > 1 else len(x), self.value)
        if self.op and self.op.startswith("x"):
            # Extract variable index
            var_idx = int(self.op[1:]) - 1  # Convert to 0-based index
            return x[:, var_idx] if x.ndim > 1 else x
        if self.op in UNARY_OPS:
            if self.left is None:
                raise ValueError(f"Unary operator {self.op} missing operand")
            return UNARY_OPS[self.op](self.left.evaluate(x))
        if self.op in BINARY_OPS:
            if self.left is None or self.right is None:
                raise ValueError(f"Binary operator {self.op} missing operand(s)")
            return BINARY_OPS[self.op](self.left.evaluate(x), self.right.evaluate(x))
        raise ValueError(
            f"Invalid node configuration: op={self.op}, value={self.value}"
        )

    def copy(self) -> "Node":
        new_node = Node(op=self.op, value=self.value)
        if self.left:
            new_node.left = self.left.copy()
        if self.right:
            new_node.right = self.right.copy()
        return new_node


    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.value:.2f}"
        if self.op and self.op.startswith("x"):
            return self.op

        if self.op in UNARY_OPS:
            expr: str = str(self.left) if self.left else ""
            return f"{self.op}({expr})"  # Always use parentheses for unary ops

        # Handle binary operators
        left: str = str(self.left) if self.left else ""
        right: str = str(self.right) if self.right else ""

        # Only need to check precedence for binary operators
        left_parens = (
            self.left
            and self.left.op in OPERATOR_PRECEDENCE
            and OPERATOR_PRECEDENCE[self.left.op] < OPERATOR_PRECEDENCE[self.op]
        )
        right_parens = (
            self.right
            and self.right.op in OPERATOR_PRECEDENCE
            and (
                OPERATOR_PRECEDENCE[self.right.op] < OPERATOR_PRECEDENCE[self.op]
                or (
                    self.op in {"-", "/"}
                    and OPERATOR_PRECEDENCE[self.right.op]
                    == OPERATOR_PRECEDENCE[self.op]
                )
            )
        )

        left_expr = f"({left})" if left_parens else left
        right_expr = f"({right})" if right_parens else right

        return f"{left_expr} {self.op} {right_expr}"

    def validate(self) -> bool:
        if self.value is not None:
            return self.op is None and self.left is None and self.right is None
        if self.op and self.op.startswith("x"):
            return self.left is None and self.right is None
        if self.op in UNARY_OPS:
            return self.left is not None and self.right is None
        if self.op in BINARY_OPS:
            return self.left is not None and self.right is not None
        return False

    def size(self) -> int:
        """Count total number of nodes in this subtree."""
        total = 1  # Count self
        if self.left:
            total += self.left.size()
        if self.right:
            total += self.right.size()
        return total

    def depth(self) -> int:
        """Calculate maximum depth from this node."""
        left_depth = self.left.depth() if self.left else -1
        right_depth = self.right.depth() if self.right else -1
        return 1 + max(left_depth, right_depth)

    def nodes(self) -> Iterator["Node"]:
        """Iterate over all nodes in this subtree."""
        yield self
        if self.left:
            yield from self.left.nodes()
        if self.right:
            yield from self.right.nodes()
