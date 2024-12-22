# All the clip function ensure that no value is outside the safe range
# The safe_power function ensures that the power operation is safe
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt

MAX_EXP = 10
MAX_POWER = 5
MAX_FLOAT = 1e10
MIN_FLOAT = 1e-10

FLOAT_PRECISION = np.float64

# Type aliases for operator functions
ArrayType = TypeVar("ArrayType", bound=np.generic)

UnaryOperator = Callable[[npt.NDArray[ArrayType]], npt.NDArray[Any]]
BinaryOperator = Callable[
    [npt.NDArray[ArrayType], npt.NDArray[ArrayType]], npt.NDArray[Any]
]


# All operators in one type
OperatorFunction = Union[UnaryOperator, BinaryOperator]


class RepresentationStyle(Enum):
    INFIX = auto()  # (a op b)
    PREFIX = auto()  # op(a)
    FUNCTION = auto()  # op(a, b)
    CUSTOM = auto()  # For special cases like abs_diff -> |a - b|


@dataclass(frozen=True)
class OperatorRepresentation:
    style: RepresentationStyle
    symbol: Optional[Union[str, Tuple[str, str, str]]]

    def __post_init__(self) -> None:
        """Ensure symbol exists."""
        if self.symbol is None:
            object.__setattr__(self, "symbol", "")  # Will be set by Operator

    def __str__(self) -> str:
        """Convert representation to string format."""
        if isinstance(self.symbol, tuple):
            return self.symbol[1]  # Return infix part for display
        return str(self.symbol)


class OperatorSpec(TypedDict):
    function: OperatorFunction
    precedence: int
    is_unary: bool
    representation: NotRequired[Optional[OperatorRepresentation]]


@dataclass(frozen=True)
class Operator:
    name: str
    function: OperatorFunction
    precedence: int
    is_unary: bool
    representation: Optional[OperatorRepresentation]

    def __str__(self) -> str:
        """Convert operator to string format."""
        assert self.representation is not None

        return str(self.representation)

    def __post_init__(self) -> None:
        if self.representation is None:
            # Default representations based on operator type
            style: (
                Literal[RepresentationStyle.PREFIX] | Literal[RepresentationStyle.INFIX]
            ) = (
                RepresentationStyle.PREFIX
                if self.is_unary
                else RepresentationStyle.INFIX
            )
            rep = OperatorRepresentation(style=style, symbol=self.name)
            object.__setattr__(self, "representation", rep)


class OperatorRegistry:
    def __init__(self, precision_type: type = FLOAT_PRECISION) -> None:
        self._operators: Dict[str, Operator] = {}
        self._precision_type: type = precision_type

    def create_default_representation(
        self, name: str, is_unary: bool
    ) -> OperatorRepresentation:
        """Create default representation based on operator type."""
        style = RepresentationStyle.PREFIX if is_unary else RepresentationStyle.INFIX
        return OperatorRepresentation(style=style, symbol=name)

    def wrap_precision(
        self, func: OperatorFunction, is_unary: bool
    ) -> OperatorFunction:
        """Wrap function to ensure input and output have correct precision."""
        if is_unary:

            def wrapped_unary(x: npt.NDArray[Any]) -> npt.NDArray[FLOAT_PRECISION]:
                x_cast: npt.NDArray[FLOAT_PRECISION] = x.astype(self._precision_type)
                result: npt.NDArray[FLOAT_PRECISION] = cast(UnaryOperator, func)(x_cast)
                return result.astype(self._precision_type)

            return cast(OperatorFunction, wrapped_unary)
        else:

            def wrapped_binary(
                x: npt.NDArray[Any], y: npt.NDArray[Any]
            ) -> npt.NDArray[FLOAT_PRECISION]:
                x_cast: npt.NDArray[FLOAT_PRECISION] = x.astype(self._precision_type)
                y_cast: npt.NDArray[FLOAT_PRECISION] = y.astype(self._precision_type)
                result: npt.NDArray[FLOAT_PRECISION] = cast(BinaryOperator, func)(
                    x_cast, y_cast
                )
                return result.astype(self._precision_type)

            return cast(OperatorFunction, wrapped_binary)

    def register(self, operator: Operator) -> None:
        """Register operator with precision handling."""
        wrapped_op = Operator(
            name=operator.name,
            function=self.wrap_precision(operator.function, operator.is_unary),
            precedence=operator.precedence,
            is_unary=operator.is_unary,
            representation=operator.representation,
        )
        self._operators[operator.name] = wrapped_op

    def register_many(self, operators: Dict[str, OperatorSpec]) -> None:
        """Bulk register operators from dictionary specification"""
        for name, spec in operators.items():
            operator = Operator(
                name=name,
                function=spec["function"],
                precedence=spec["precedence"],
                is_unary=spec["is_unary"],
                representation=spec.get("representation"),
            )
            self.register(operator)

    @property
    def unary_ops(self) -> Dict[str, Callable]:
        return {
            name: op.function for name, op in self._operators.items() if op.is_unary
        }

    @property
    def binary_ops(self) -> Dict[str, Callable]:
        return {
            name: op.function for name, op in self._operators.items() if not op.is_unary
        }

    @property
    def representations(self) -> Dict[str, OperatorRepresentation]:
        return {
            name: op.representation
            for name, op in self._operators.items()
            if op.representation is not None
        }

    @property
    def precedence(self) -> Dict[str, int]:
        return {name: op.precedence for name, op in self._operators.items()}

    def validate(self) -> None:
        """Runtime validation of operator completeness"""
        for op in self._operators.values():
            if not callable(op.function):
                raise ValueError(f"Operator {op.name} has invalid function")
            if not isinstance(op.precedence, int):
                raise ValueError(f"Operator {op.name} has invalid precedence")


# Initialize registry
registry = OperatorRegistry()


def safe_power(
    x: npt.NDArray[FLOAT_PRECISION], y: npt.NDArray[FLOAT_PRECISION]
) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe power operation handling edge cases and preventing warnings."""
    # Clip inputs to safe ranges
    x_safe = np.clip(x, -MAX_FLOAT, MAX_FLOAT)
    y_safe = np.clip(y, -MAX_POWER, MAX_POWER)

    # Handle negative bases
    x_safe = np.where(x_safe < 0, MIN_FLOAT, x_safe)

    # Prevent zero division
    x_safe = np.where(np.abs(x_safe) < MIN_FLOAT, MIN_FLOAT, x_safe)

    # Compute power and handle invalid results
    result = np.power(x_safe, y_safe)
    result = np.nan_to_num(result, nan=MIN_FLOAT, posinf=MAX_FLOAT, neginf=-MAX_FLOAT)

    return np.clip(result, -MAX_FLOAT, MAX_FLOAT)


def safe_tan(x: npt.NDArray[FLOAT_PRECISION]) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe tangent function with clipping"""
    clipped = np.clip(x, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
    return np.tan(clipped)


def safe_arcsin(x: npt.NDArray[FLOAT_PRECISION]) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe arcsine function with clipping"""
    clipped = np.clip(x, -1, 1)
    return np.arcsin(clipped)


def safe_sinh(x: npt.NDArray[FLOAT_PRECISION]) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe hyperbolic sine function with clipping"""
    clipped = np.clip(x, -MAX_EXP, MAX_EXP)
    return np.sinh(clipped)


def safe_cosh(x: npt.NDArray[FLOAT_PRECISION]) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe hyperbolic cosine function with clipping"""
    clipped = np.clip(x, -MAX_EXP, MAX_EXP)
    return np.cosh(clipped)


# UNARY_OPS: Dict[str, Callable] = {
#     "sin": np.sin,
#     "cos": np.cos,
#     "tan": lambda x: np.tan(np.clip(x, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)),
#     "arcsin": lambda x: np.arcsin(np.clip(x, -1, 1)),
#     "sinh": lambda x: np.sinh(np.clip(x, -MAX_EXP, MAX_EXP)),
#     "cosh": lambda x: np.cosh(np.clip(x, -MAX_EXP, MAX_EXP)),
#     "sign": np.sign,
#     "exp": lambda x: np.exp(np.clip(x, -MAX_EXP, MAX_EXP)),
#     "log": lambda x: np.log(np.abs(x) + MIN_FLOAT),
#     "sqrt": lambda x: np.sqrt(np.maximum(0, x)),
#     "sigmoid": lambda x: 1 / (1 + np.exp(np.clip(-x, -MAX_EXP, MAX_EXP))),
#     "abs": np.abs,
#     "reciprocal": lambda x: np.divide(1, np.where(np.abs(x) < MIN_FLOAT, MIN_FLOAT, x)),
# }

# BINARY_OPS: Dict[str, Callable] = {
#     "+": lambda x, y: np.clip(x + y, -MAX_FLOAT, MAX_FLOAT),
#     "-": lambda x, y: np.clip(x - y, -MAX_FLOAT, MAX_FLOAT),
#     "*": lambda x, y: np.clip(x * y, -MAX_FLOAT, MAX_FLOAT),
#     "/": lambda x, y: np.divide(x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)),
#     "**": safe_power,
#     "%": lambda x, y: np.mod(x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)),
#     "min": lambda x, y: np.minimum(
#         np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
#     ),
#     "max": lambda x, y: np.maximum(
#         np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
#     ),
#     "abs_diff": lambda x, y: np.abs(x - y),
#     "//": lambda x, y: np.floor_divide(
#         x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
#     ),
# }

# OP_REPRESENTATIONS = {
#     "+": "+",
#     "-": "-",
#     "*": "*",
#     "/": "/",
#     "**": "**",
#     "%": "%",
#     "//": "//",
#     "min": "min",
#     "max": "max",
#     "abs_diff": ("|", "-", "|"),
#     "sin": "sin",
#     "cos": "cos",
#     "exp": "exp",
#     "log": "log",
# }

# OPERATOR_PRECEDENCE: Dict[str, int] = {
#     "+": 1,
#     "-": 1,
#     "min": 1,
#     "max": 1,
#     "*": 2,
#     "/": 2,
#     "//": 2,
#     "%": 2,
#     "abs_diff": 3,
#     "**": 4,
#     "sin": 4,
#     "cos": 4,
#     "exp": 4,
#     "log": 4,
# }

BASE_OPERATORS: Dict[str, OperatorSpec] = {
    # Unary operators - highest precedence (4)
    "sin": {
        "function": np.sin,
        "precedence": 4,
        "is_unary": True,
    },
    "cos": {
        "function": np.cos,
        "precedence": 4,
        "is_unary": True,
    },
    "tan": {
        "function": safe_tan,
        "precedence": 4,
        "is_unary": True,
    },
    "arcsin": {
        "function": safe_arcsin,
        "precedence": 4,
        "is_unary": True,
    },
    "sinh": {
        "function": safe_sinh,
        "precedence": 4,
        "is_unary": True,
    },
    "cosh": {
        "function": safe_cosh,
        "precedence": 4,
        "is_unary": True,
    },
    "exp": {
        "function": lambda x: np.exp(np.clip(x, -MAX_EXP, MAX_EXP)),
        "precedence": 4,
        "is_unary": True,
    },
    "log": {
        "function": lambda x: np.log(np.abs(x) + MIN_FLOAT),
        "precedence": 4,
        "is_unary": True,
    },
    "sqrt": {
        "function": lambda x: np.sqrt(np.maximum(0, x)),
        "precedence": 4,
        "is_unary": True,
    },
    "abs": {
        "function": np.abs,
        "precedence": 4,
        "is_unary": True,
    },
    "sign": {
        "function": np.sign,
        "precedence": 4,
        "is_unary": True,
    },
    "sigmoid": {
        "function": lambda x: 1 / (1 + np.exp(np.clip(-x, -MAX_EXP, MAX_EXP))),
        "precedence": 4,
        "is_unary": True,
    },
    "reciprocal": {
        "function": lambda x: np.divide(
            1, np.where(np.abs(x) < MIN_FLOAT, MIN_FLOAT, x)
        ),
        "precedence": 4,
        "is_unary": True,
    },
    # Binary operators
    # Power and special operations - precedence 3
    "**": {
        "function": safe_power,
        "precedence": 4,
        "is_unary": False,
    },
    "abs_diff": {
        "function": lambda x, y: np.abs(x - y),
        "precedence": 3,
        "is_unary": False,
        "representation": OperatorRepresentation(
            style=RepresentationStyle.CUSTOM, symbol=("|", "-", "|")
        ),
    },
    # Multiplicative operators - precedence 2
    "*": {
        "function": lambda x, y: np.clip(x * y, -MAX_FLOAT, MAX_FLOAT),
        "precedence": 2,
        "is_unary": False,
    },
    "/": {
        "function": lambda x, y: np.divide(
            x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
        ),
        "precedence": 2,
        "is_unary": False,
    },
    "//": {
        "function": lambda x, y: np.floor_divide(
            x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
        ),
        "precedence": 2,
        "is_unary": False,
    },
    "%": {
        "function": lambda x, y: np.mod(
            x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
        ),
        "precedence": 2,
        "is_unary": False,
    },
    # Additive operators - lowest precedence (1)
    "+": {
        "function": lambda x, y: np.clip(x + y, -MAX_FLOAT, MAX_FLOAT),
        "precedence": 1,
        "is_unary": False,
    },
    "-": {
        "function": lambda x, y: np.clip(x - y, -MAX_FLOAT, MAX_FLOAT),
        "precedence": 1,
        "is_unary": False,
    },
    "min": {
        "function": lambda x, y: np.minimum(
            np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
        ),
        "precedence": 1,
        "is_unary": False,
        "representation": OperatorRepresentation(
            style=RepresentationStyle.FUNCTION, symbol="min"
        ),
    },
    "max": {
        "function": lambda x, y: np.maximum(
            np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
        ),
        "precedence": 1,
        "is_unary": False,
        "representation": OperatorRepresentation(
            style=RepresentationStyle.FUNCTION, symbol="max"
        ),
    },
}

# At the end of the file, after BASE_OPERATORS definition:
registry = OperatorRegistry()
registry.register_many(BASE_OPERATORS)

# Export these for compatibility with existing code
UNARY_OPS = registry.unary_ops
BINARY_OPS = registry.binary_ops
OPERATOR_PRECEDENCE = registry.precedence


def get_var_name(idx: int) -> str:
    """Generate variable name for given index."""
    return f"x{idx}"
