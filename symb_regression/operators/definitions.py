from typing import Callable, Dict

import numpy as np

UNARY_OPS: Dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": lambda x: np.exp(np.clip(x, -10, 10)),
    "log": lambda x: np.log(np.abs(x) + 1e-10),
}

BINARY_OPS: Dict[str, Callable] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": lambda x, y: np.divide(x, np.where(np.abs(y) < 1e-10, 1e-10, y)),
}

OPERATOR_PRECEDENCE: Dict[str, int] = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
}


def get_var_name(idx: int) -> str:
    """Generate variable name for given index."""
    return f"x{idx}"
