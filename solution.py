import numpy as np


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def f2(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Equation not yet implemented")


def f3(x: np.ndarray) -> np.ndarray:
    return (np.arctan2(-6.732, -3.574)) * ((np.sinh(x[1]) - np.cosh(3.019)) + x[2])


def f4(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Equation not yet implemented")


def f5(x: np.ndarray) -> np.ndarray:
    return np.log(np.sign(x))


def f6(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Equation not yet implemented")


def f7(x: np.ndarray) -> np.ndarray:
    return 1 / np.tan(
        np.arctanh(
            (np.sin(9.050) ** np.cosh(np.sinh((np.arctan2(x[1], np.abs(x[1] - x[0]))))))
        )
    )


def f8(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Equation not yet implemented")
