# All the clip function ensure that no value is outside the safe range
# The safe_power function ensures that the power operation is safe
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
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


@dataclass(frozen=True)
class OperatorSpec:
    function: OperatorFunction
    precedence: int
    is_unary: bool
    representation: Optional[OperatorRepresentation] = field(default=None)


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
        self.operators: Dict[str, Operator] = {}
        self.precision_type: type = precision_type

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
                x_cast: npt.NDArray[FLOAT_PRECISION] = x.astype(self.precision_type)
                result: npt.NDArray[FLOAT_PRECISION] = cast(UnaryOperator, func)(x_cast)
                return result.astype(self.precision_type)

            return cast(OperatorFunction, wrapped_unary)
        else:

            def wrapped_binary(
                x: npt.NDArray[Any], y: npt.NDArray[Any]
            ) -> npt.NDArray[FLOAT_PRECISION]:
                x_cast: npt.NDArray[FLOAT_PRECISION] = x.astype(self.precision_type)
                y_cast: npt.NDArray[FLOAT_PRECISION] = y.astype(self.precision_type)
                result: npt.NDArray[FLOAT_PRECISION] = cast(BinaryOperator, func)(
                    x_cast, y_cast
                )
                return result.astype(self.precision_type)

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
        self.operators[operator.name] = wrapped_op

    def register_many(self, operators: Dict[str, OperatorSpec]) -> None:
        """Bulk register operators from dictionary specification"""
        for name, spec in operators.items():
            operator = Operator(
                name=name,
                function=spec.function,
                precedence=spec.precedence,
                is_unary=spec.is_unary,
                representation=spec.representation,
            )
            self.register(operator)

    @property
    def unary_ops(self) -> Dict[str, Callable]:
        return {name: op.function for name, op in self.operators.items() if op.is_unary}

    @property
    def binary_ops(self) -> Dict[str, Callable]:
        return {
            name: op.function for name, op in self.operators.items() if not op.is_unary
        }

    @property
    def representations(self) -> Dict[str, OperatorRepresentation]:
        return {
            name: op.representation
            for name, op in self.operators.items()
            if op.representation is not None
        }

    @property
    def precedence(self) -> Dict[str, int]:
        return {name: op.precedence for name, op in self.operators.items()}

    def validate(self) -> None:
        """Runtime validation of operator completeness"""
        for op in self.operators.values():
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


def safe_atan2(
    x1_clipped: npt.NDArray[FLOAT_PRECISION], x2: npt.NDArray[FLOAT_PRECISION]
) -> npt.NDArray[FLOAT_PRECISION]:
    """Safe tangent function with clipping"""
    x1_clipped = np.clip(x1_clipped, -np.pi + 1e-10, np.pi - 1e-10)
    x2_clipped = np.clip(x2, -np.pi + 1e-10, np.pi - 1e-10)
    return np.atan2(x1_clipped, x2_clipped)


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


BASE_OPERATORS: Dict[str, OperatorSpec] = {
    # Unary operators - highest precedence (4)
    "sin": OperatorSpec(
        function=np.sin,
        precedence=4,
        is_unary=True,
    ),
    "cos": OperatorSpec(
        function=np.cos,
        precedence=4,
        is_unary=True,
    ),
    "tan": OperatorSpec(
        function=safe_tan,
        precedence=4,
        is_unary=True,
    ),
    "atan2": OperatorSpec(
        function=safe_atan2,
        precedence=4,
        is_unary=False,
    ),
    "cot": OperatorSpec(
        function=lambda x: np.clip(
            1 / np.tan(np.where(np.abs(x) < MIN_FLOAT, MIN_FLOAT, x)),
            -MAX_FLOAT,
            MAX_FLOAT,
        ),
        precedence=4,
        is_unary=True,
        
    ),
    "arcsin": OperatorSpec(
        function=safe_arcsin,
        precedence=4,
        is_unary=True,
    ),
    "sinh": OperatorSpec(
        function=safe_sinh,
        precedence=4,
        is_unary=True,
    ),
    "cosh": OperatorSpec(
        function=safe_cosh,
        precedence=4,
        is_unary=True,
    ),
    "exp": OperatorSpec(
        function=lambda x: np.exp(np.clip(x, -MAX_EXP, MAX_EXP)),
        precedence=4,
        is_unary=True,
    ),
    "log": OperatorSpec(
        function=lambda x: np.log(np.abs(x) + MIN_FLOAT),
        precedence=4,
        is_unary=True,
    ),
    "log2": OperatorSpec(
        function=lambda x: np.log2(np.abs(x) + MIN_FLOAT),
        precedence=4,
        is_unary=True,
    ),
    "sqrt": OperatorSpec(
        function=lambda x: np.sqrt(np.maximum(0, x)),
        precedence=4,
        is_unary=True,
    ),
    "abs": OperatorSpec(
        function=np.abs,
        precedence=4,
        is_unary=True,
    ),
    "sign": OperatorSpec(
        function=np.sign,
        precedence=4,
        is_unary=True,
    ),
    "sigmoid": OperatorSpec(
        function=lambda x: 1 / (1 + np.exp(np.clip(-x, -MAX_EXP, MAX_EXP))),
        precedence=4,
        is_unary=True,
    ),
    "reciprocal": OperatorSpec(
        function=lambda x: np.divide(1, np.where(np.abs(x) < MIN_FLOAT, MIN_FLOAT, x)),
        precedence=4,
        is_unary=True,
    ),
    # Binary operators
    # Power and special operations - precedence 3
    "**": OperatorSpec(
        function=safe_power,
        precedence=4,
        is_unary=False,
    ),
    "abs_diff": OperatorSpec(
        function=lambda x, y: np.abs(x - y),
        precedence=3,
        is_unary=False,
        representation=OperatorRepresentation(
            style=RepresentationStyle.CUSTOM, symbol=("|", "-", "|")
        ),
    ),
    # Multiplicative operators - precedence 2
    "*": OperatorSpec(
        function=lambda x, y: np.clip(x * y, -MAX_FLOAT, MAX_FLOAT),
        precedence=2,
        is_unary=False,
    ),
    "/": OperatorSpec(
        function=lambda x, y: np.divide(
            x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
        ),
        precedence=2,
        is_unary=False,
    ),
    "//": OperatorSpec(
        function=lambda x, y: np.floor_divide(
            x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
        ),
        precedence=2,
        is_unary=False,
    ),
    "%": OperatorSpec(
        function=lambda x, y: np.mod(x, np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)),
        precedence=2,
        is_unary=False,
    ),
    # Additive operators - lowest precedence (1)
    "+": OperatorSpec(
        function=lambda x, y: np.clip(x + y, -MAX_FLOAT, MAX_FLOAT),
        precedence=1,
        is_unary=False,
    ),
    "-": OperatorSpec(
        function=lambda x, y: np.clip(x - y, -MAX_FLOAT, MAX_FLOAT),
        precedence=1,
        is_unary=False,
    ),
    "min": OperatorSpec(
        function=lambda x, y: np.minimum(
            np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
        ),
        precedence=1,
        is_unary=False,
        representation=OperatorRepresentation(
            style=RepresentationStyle.FUNCTION, symbol="min"
        ),
    ),
    "max": OperatorSpec(
        function=lambda x, y: np.maximum(
            np.clip(x, -MAX_FLOAT, MAX_FLOAT), np.clip(y, -MAX_FLOAT, MAX_FLOAT)
        ),
        precedence=1,
        is_unary=False,
        representation=OperatorRepresentation(
            style=RepresentationStyle.FUNCTION, symbol="max"
        ),
    ),
}

# At the end of the file, after BASE_OPERATORS definition:
registry = OperatorRegistry()
registry.register_many(BASE_OPERATORS)

# Export these for compatibility with existing code
UNARY_OPS = registry.unary_ops
BINARY_OPS = registry.binary_ops
OPERATOR_PRECEDENCE = registry.precedence


class OperatorSet(Enum):
    MINIMAL = auto()  # +, -, *, /
    BASIC = auto()  # MINIMAL + **
    TRIG = auto()  # BASIC + sin, cos
    ADVANCED = auto()  # All operators


OPERATOR_SETS: Dict[OperatorSet, Set[str]] = {
    OperatorSet.MINIMAL: {"+", "-", "*", "/"},
    OperatorSet.BASIC: {"+", "-", "*", "/", "**"},
    OperatorSet.TRIG: {"+", "-", "*", "/", "**", "sin", "cos"},
    OperatorSet.ADVANCED: set(BASE_OPERATORS.keys()),
}


def get_operator_set(
    operator_set: OperatorSet | Set[str] = OperatorSet.BASIC,
) -> Dict[str, OperatorSpec]:
    """Get a filtered set of operators based on configuration"""
    if isinstance(operator_set, OperatorSet):
        selected_ops = OPERATOR_SETS[operator_set]
    else:
        selected_ops = operator_set

    # Validate operators exist in BASE_OPERATORS
    invalid_ops = selected_ops - set(BASE_OPERATORS.keys())
    if invalid_ops:
        raise ValueError(f"Invalid operators: {invalid_ops}")

    # Filter BASE_OPERATORS to only include selected ops
    return {op: spec for op, spec in BASE_OPERATORS.items() if op in selected_ops}


@dataclass
class SymbolicConfig:
    operators: Dict[str, OperatorSpec]
    n_variables: int

    @classmethod
    def create_custom(
        cls,
        operator_set: OperatorSet | Set[str],
        n_variables: int = 1,
    ) -> Self:
        operators = get_operator_set(operator_set)
        return cls(operators, n_variables)

    @classmethod
    def create_minimal(
        cls,
        n_variables: int = 1,
    ) -> Self:
        operators = get_operator_set(OperatorSet.MINIMAL)
        return cls(operators, n_variables)

    @classmethod
    def create_basic(
        cls,
        n_variables: int = 1,
    ) -> Self:
        operators = get_operator_set(OperatorSet.BASIC)
        return cls(operators, n_variables)

    @classmethod
    def create_trig(
        cls,
        n_variables: int = 1,
    ) -> Self:
        operators = get_operator_set(OperatorSet.TRIG)
        return cls(operators, n_variables)

    @classmethod
    def create(
        cls,
        operator_set: OperatorSet | Set[str] = OperatorSet.ADVANCED,
        n_variables: int = 1,
    ) -> Self:
        operators = get_operator_set(operator_set)
        return cls(operators, n_variables)
