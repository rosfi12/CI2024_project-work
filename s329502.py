import numpy as np


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def f2(x: np.ndarray) -> np.ndarray:
    return np.sqrt((4.691 / (3.457 ** (-4.374 - x[0]))) / ((np.sign(np.abs(x[1])) * np.arctan2(x[2], x[1])) / np.maximum(x[1], x[0])))


def f3(x: np.ndarray) -> np.ndarray:
    return (np.arctan2(-6.732, -3.574)) * ((np.sinh(x[1]) - np.cosh(3.019)) + x[2])


def f4(x: np.ndarray) -> np.ndarray:
    return np.sinh((np.log(3.237) * (np.cos(x[0]) - (x[0] / x[0])))) + np.exp(np.abs(np.cos(x[0]) - np.reciprocal(-0.727)))


def f5(x: np.ndarray) -> np.ndarray:
    return np.log(np.sign(x))


def f6(x: np.ndarray) -> np.ndarray:
    raise x[1] + (np.sqrt(1/np.tan(np.log2(2.177))) * (np.sqrt(-4.419) + (x[1] - x[0])))


def f7(x: np.ndarray) -> np.ndarray:
    return 1 / np.tan(
        np.arctanh(
            (np.sin(9.050) ** np.cosh(np.sinh((np.arctan2(x[1], np.abs(x[1] - x[0]))))))
        )
    )


def f8(x: np.ndarray) -> np.ndarray:
    raise (np.sinh(4.067) // np.exp(x[5])) * -3.664
