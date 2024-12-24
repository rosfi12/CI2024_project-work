from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from symb_regression.operators.definitions import (
    OperatorSpec,
    RepresentationStyle,
    SymbolicConfig,
    registry,
)


class Node:
    def __init__(
        self,
        op: Optional[str] = None,
        value: Optional[float] = None,
        variable_idx: Optional[int] = None,
    ) -> None:
        self.op: str | None = op
        self.value: float | None = value
        self.variable_idx: int | None = variable_idx
        self.left: Node | None = None
        self.right: Node | None = None

    def evaluate(
        self, x: np.ndarray, config: SymbolicConfig
    ) -> npt.NDArray[np.float64]:
        if self.value is not None:
            return np.full(x.shape[0], self.value)
        elif self.variable_idx is not None:
            if self.variable_idx >= x.shape[1]:
                raise ValueError(
                    f"Variable index {self.variable_idx} is out of bounds for input "
                    f"with {x.shape[1]} variables"
                )
            return x[:, self.variable_idx]
        elif self.op is not None:
            assert self.left is not None, "Operator node missing left child"
            op_spec: OperatorSpec = config.operators[self.op]
            # TODO: type safeguard for the function signature based on the operator is_unary property
            if op_spec.is_unary:
                return op_spec.function(self.left.evaluate(x, config))  # type: ignore

            assert self.right is not None, "Binary operator node missing right child"
            return op_spec.function(
                self.left.evaluate(x, config),
                self.right.evaluate(x, config),  # type: ignore
            )
        else:
            raise ValueError("Invalid node: missing value, variable, or operator")

    def validate(self) -> bool:
        if self.value is not None:
            return self.op is None and self.left is None and self.right is None
        if self.variable_idx is not None:
            return self.op is None and self.left is None and self.right is None
        if self.op in registry.unary_ops:
            return self.left is not None and self.right is None
        if self.op in registry.binary_ops:
            return self.left is not None and self.right is not None
        return False

    def depth(self) -> int:
        """Calculate maximum depth from this node."""
        left_depth = self.left.depth() if self.left else -1
        right_depth = self.right.depth() if self.right else -1
        return 1 + max(left_depth, right_depth)

    def copy(self) -> "Node":
        new_node = Node(op=self.op, value=self.value, variable_idx=self.variable_idx)
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
            return f"{self.value:.3f}"
        if self.variable_idx is not None:
            return f"x{self.variable_idx}"

        if self.op not in registry.operators:
            raise ValueError(f"Unknown operator: {self.op}")

        op_spec = registry.operators[self.op]
        rep = op_spec.representation
        if rep is None:
            raise ValueError(f"Missing representation for operator {self.op}")

        # Validate operands
        if op_spec.is_unary and self.left is None:
            raise ValueError(f"Unary operator {self.op} missing operand")
        if not op_spec.is_unary and (self.left is None or self.right is None):
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
                assert isinstance(
                    rep.symbol, tuple
                ), f"Operator '{self.op}' expected a tuple for custom representation"
                prefix, infix, suffix = rep.symbol
                return f"{prefix}{str(self.left)} {infix} {str(self.right)}{suffix}"
            case _:
                raise ValueError(
                    f"Unknown representation style: {rep.style} for operator {self.op}"
                )

    def __hash__(self) -> int:
        """Hash the tree structure."""
        return hash((self.op, self.value, self.left, self.right))
