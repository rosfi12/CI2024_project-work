from typing import Callable, Dict, Tuple

import numpy as np

UNARY_OPS: Dict[str, Tuple[Callable, float]] = {
    "sin": (np.sin, 0.2),
    "cos": (np.cos, 0.2),
    "exp": (lambda x: np.exp(np.clip(x, -10, 10)), 0.1),
    "log": (lambda x: np.log(np.abs(x) + 1e-10), 0.1),
}

BINARY_OPS: Dict[str, Tuple[Callable, float]] = {
    "+": (np.add, 0.2),
    "-": (np.subtract, 0.1),
    "*": (np.multiply, 0.1),
    "/": (
        lambda x, y: np.divide(x, np.where(np.abs(y) < 1e-10, 1e-10, y)),
        0.1,
    ),
}


def get_var_name(idx: int) -> str:
    """Generate variable name for given index."""
    return f"x{idx}"
