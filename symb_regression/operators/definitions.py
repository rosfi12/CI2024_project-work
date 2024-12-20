from typing import Callable, Dict, Tuple

import numpy as np

UNARY_OPS: Dict[str, Tuple[Callable, float]] = {
    "sin": (np.sin, 0.2),
    "cos": (np.cos, 0.2),
    "tan": (lambda x: np.tan(np.clip(x, -np.pi/2 + 1e-10, np.pi/2 - 1e-10)), 0.1),
    "arcsin": (lambda x: np.arcsin(np.clip(x, -1, 1)), 0.1), 
    "sinh": (np.sinh, 0.1),
    "cosh": (np.cosh, 0.1),
    "sign": (np.sign, 0.1),
    "exp": (lambda x: np.exp(np.clip(x, -10, 10)), 0.1),
    "log": (lambda x: np.log(np.abs(x) + 1e-10), 0.1),
    "sqrt": (lambda x: np.sqrt(np.maximum(0, x)), 0.1),
    "sigmoid": (lambda x: 1 / (1 + np.exp(np.clip(-x, -10, 10))), 0.1),
    "abs": (np.abs, 0.1),
    "reciprocal": (lambda x: np.divide(1, np.where(np.abs(x) < 1e-10, 1e-10, x)), 0.1),
}

BINARY_OPS: Dict[str, Tuple[Callable, float]] = {
    "+": (np.add, 0.2),
    "-": (np.subtract, 0.1),
    "*": (np.multiply, 0.1),
    "/": (
        lambda x, y: np.divide(x, np.where(np.abs(y) < 1e-10, 1e-10, y)),
        0.1,
    ),
    "**": (lambda x, y: np.power(np.clip(x, -1e5, 1e5), np.clip(y, -5, 5)), 0.1),
    "%": (np.mod, 0.1),
    "min": (np.minimum, 0.1),
    "max": (np.maximum, 0.1),
    "abs_diff": (lambda x, y: np.abs(x - y), 0.1),
    "//": (np.floor_divide, 0.1),

    
}


def get_var_name(idx: int) -> str:
    """Generate variable name for given index."""
    return f"x{idx}"
