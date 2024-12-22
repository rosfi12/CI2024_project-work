from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from symb_regression.operators.definitions import (
    BINARY_OPS,
    UNARY_OPS,
    RepresentationStyle,
    registry,
)


class Node:
    def __init__(
        self,
        op: Optional[str] = None,
        value: Optional[float] = None,
    ) -> None:
        self.op: str | None = op
        self.value: float | None = value
        self.left: Node | None = None
        self.right: Node | None = None

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

    def depth(self) -> int:
        """Calculate maximum depth from this node."""
        left_depth = self.left.depth() if self.left else -1
        right_depth = self.right.depth() if self.right else -1
        return 1 + max(left_depth, right_depth)

    def copy(self) -> "Node":
        new_node = Node(op=self.op, value=self.value)
        if self.left:
            new_node.left = self.left.copy()
        if self.right:
            new_node.right = self.right.copy()
        return new_node

    def __iter__(self) -> Iterator["Node"]:
        """Iterate over all nodes in this subtree."""
        yield self
        if self.left:
            yield from self.left
        if self.right:
            yield from self.right

    def __eq__(self, other: object) -> bool:
        """Compare two trees for equality."""
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.op == other.op
            and self.value == other.value
            and self.left == other.left
            and self.right == other.right
        )

    def __len__(self) -> int:
        """Count total number of nodes in this subtree."""
        total = 1  # Count self
        if self.left:
            total += len(self.left)
        if self.right:
            total += len(self.right)
        return total

    def __str__(self) -> str:
        """Return string representation of the expression tree."""
        if self.value is not None:
            return str(self.value)
        if self.op and self.op.startswith("x"):
            var_num = int(self.op[1:])  # Extract number after 'x'
            return f"x{var_num-1}"  # Convert to 0-based index

        if self.op not in registry._operators:
            raise ValueError(f"Unknown operator: {self.op}")

        operator = registry._operators[self.op]
        rep = operator.representation
        if rep is None:
            raise ValueError(f"Missing representation for operator {self.op}")

        # Validate operands
        if operator.is_unary and self.left is None:
            raise ValueError(f"Unary operator {self.op} missing operand")
        if not operator.is_unary and (self.left is None or self.right is None):
            raise ValueError(f"Binary operator {self.op} missing operand(s)")

        # Format based on representation style
        match rep.style:
            case RepresentationStyle.PREFIX:
                return f"{rep.symbol}({str(self.left)})"
            case RepresentationStyle.INFIX:
                return f"({str(self.left)} {rep.symbol} {str(self.right)})"
            case RepresentationStyle.FUNCTION:
                return f"{rep.symbol}({str(self.left)}, {str(self.right)})"
            case RepresentationStyle.CUSTOM:
                # Shouldn't be necessary to check for tuple, but mypy complains
                assert isinstance(rep.symbol, tuple)
                prefix, infix, suffix = rep.symbol
                return f"{prefix}{str(self.left)} {infix} {str(self.right)}{suffix}"
            case _:
                raise ValueError(f"Unknown representation style for operator {self.op}")

    def __hash__(self) -> int:
        """Hash the tree structure."""
        return hash((self.op, self.value, self.left, self.right))
