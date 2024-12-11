from typing import Optional

import numpy as np
import numpy.typing as npt

from symb_regression.base import INode
from symb_regression.operators.definitions import BINARY_OPS, UNARY_OPS


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
            return UNARY_OPS[self.op][0](self.left.evaluate(x))
        if self.op in BINARY_OPS:
            if self.left is None or self.right is None:
                raise ValueError(
                    f"Binary operator {self.op} missing operand(s)"
                )
            return BINARY_OPS[self.op][0](
                self.left.evaluate(x), self.right.evaluate(x)
            )
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

    def to_string(self) -> str:
        if self.value is not None:
            return f"{self.value:.3f}"
        if self.op == "x1":
            return self.op
        if self.op in UNARY_OPS:
            return f"{self.op}({self.left.to_string() if self.left else ''})"
        return f"({self.left.to_string() if self.left else ''} {self.op} {self.right.to_string() if self.right else ''})"

    def to_pretty_string(self) -> str:
        """Convert the expression tree to a more readable string format."""
        if self.value is not None:
            return f"{self.value:.3f}"
        if self.op and self.op.startswith("x"):
            var_idx = int(self.op[1:]) - 1  # Convert to 0-based indexing
            return f"x{var_idx}"
        if self.op in UNARY_OPS:
            return f"{self.op}({self.left.to_pretty_string() if self.left else ''})"
        if self.op in BINARY_OPS:
            left = self.left.to_pretty_string() if self.left else ""
            right = self.right.to_pretty_string() if self.right else ""
            # Add parentheses only when necessary
            if self.op in ["*", "/"]:
                # Check if child operations need parentheses
                if self.left and self.left.op in ["+", "-"]:
                    left = f"({left})"
                if self.right and self.right.op in ["+", "-"]:
                    right = f"({right})"
            return f"{left} {self.op} {right}"
        return "Invalid Expression"

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
