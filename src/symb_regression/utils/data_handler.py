import os
from typing import Tuple

import numpy as np
import numpy.typing as npt


def load_data(
    data_dir: str, problem_name: str, show_stats: bool = False
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load problem data from a .npz file."""
    try:
        file_path = os.path.join(data_dir, f"{problem_name}.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.load(file_path)
        x = data["x"]  # Shape (n_variables, n_samples)
        y = data["y"]  # Shape (n_samples,)

        # Transpose x to shape (n_samples, n_variables)
        if x.ndim == 2 and x.shape[0] < x.shape[1]:
            x = x.T

        if show_stats:
            print("\nData Statistics:")
            print(f"X shape: {x.shape}, Y shape: {y.shape}")
            print(f"Number of variables: {x.shape[1] if x.ndim > 1 else 1}")

            # Print statistics for each variable
            if x.ndim > 1:
                for i in range(x.shape[1]):
                    print(f"\nVariable x{i}:")
                    print(f"  Range: [{x[:,i].min():.3f}, {x[:,i].max():.3f}]")
                    print(f"  Mean: {x[:,i].mean():.3f}")
                    print(f"  Std: {x[:,i].std():.3f}")
                    corr = np.corrcoef(x[:, i], y)[0, 1]
                    print(f"  Correlation with y: {corr:.3f}")

            print("\nTarget y:")
            print(f"  Range: [{y.min():.3f}, {y.max():.3f}]")
            print(f"  Mean: {y.mean():.3f}")
            print(f"  Std: {y.std():.3f}")

        return x, y

    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}") from e


def split_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    train_size: float = 0.8,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Split data into training and validation sets.

    Args:
        x: Input features array
        y: Target values array
        train_ratio: Ratio of data to use for training (default: 0.8)

    Returns:
        Tuple containing (x_train, x_val, y_train, y_val)
    """
    # Input validation
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input x and y must be numpy arrays")

    if train_size <= 0 or train_size >= 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Handle different input shapes
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[0] < x.shape[1]:
        x = x.T

    n_samples = x.shape[0]

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have same number of samples. Got x: {x.shape[0]}, y: {y.shape[0]}"
        )

    # Create random permutation
    idx = np.random.permutation(n_samples)
    train_size = int(n_samples * train_size)

    # Split the data
    x_train = x[idx[:train_size]]
    x_val = x[idx[train_size:]]
    y_train = y[idx[:train_size]]
    y_val = y[idx[train_size:]]

    return x_train, x_val, y_train, y_val
