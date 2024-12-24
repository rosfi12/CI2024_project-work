import os
from typing import Tuple

import numpy as np
import numpy.typing as npt


def print_stats(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    print("=" * 50)
    print("\nData Statistics:")
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
    print(f"Number of variables: {x.shape[1] if x.ndim > 1 else 1}")

    # Print statistics for each variable
    if x.ndim > 1:
        for i in range(x.shape[1]):
            print(f"\nVariable x{i}:")
            print(f"  Range: [{x[:,i].min():g}, {x[:,i].max():g}]")
            print(f"  Mean: {x[:,i].mean():g}")
            print(f"  Std: {x[:,i].std():g}")
            corr = np.corrcoef(x[:, i], y)[0, 1]
            print(f"  Correlation with y: {corr:g}")

    print("\nTarget y:")
    print(f"  Range: [{y.min():g}, {y.max():g}]")
    print(f"  Mean: {y.mean():g}")
    print(f"  Std: {y.std():g}")
    print("=" * 50)


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
            print_stats(x, y)

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

    # Create random permutation using numpy
    idx = np.random.permutation(n_samples)
    train_size_int = int(n_samples * train_size)

    # Use advanced indexing instead of multiple array creations
    mask = np.zeros(n_samples, dtype=bool)
    mask[idx[:train_size_int]] = True

    x_train = x[mask]
    y_train = y[mask]
    x_val = x[~mask]
    y_val = y[~mask]

    return x_train, x_val, y_train, y_val


def sort_and_filter_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sort_column: int = 1,
    range_limit: float | None = None,
    from_end: bool = False,
    show_stats: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Sort data by specified column and optionally filter by range.
    """
    # Ensure valid column index
    if sort_column >= x.shape[1]:
        raise ValueError(f"sort_column {sort_column} exceeds x dimensions {x.shape[1]}")

    # Simple sort by specified column
    sort_idx = np.argsort(x[:, sort_column])

    # Apply sorting
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    if show_stats:
        print("Before filtering:")
        print_stats(x_sorted, y_sorted)

    # Apply range filtering if specified
    if range_limit is not None:
        # Create mask for both columns within range
        mask = np.ones(len(x_sorted), dtype=bool)
        for col in range(x_sorted.shape[1]):
            if from_end:
                max_val = np.max(x_sorted[:, col])
                min_val = max_val - range_limit
                col_mask = (x_sorted[:, col] >= min_val) & (x_sorted[:, col] <= max_val)
            else:
                col_mask = (x_sorted[:, col] >= 0) & (x_sorted[:, col] <= range_limit)
            mask = mask & col_mask

        x_filtered = x_sorted[mask]
        y_filtered = y_sorted[mask]

        if show_stats:
            print("\nAfter filtering:")
            print_stats(x_filtered, y_filtered)

        return x_filtered, y_filtered

    if show_stats:
        print(f"Final shape: {x_sorted.shape}")

    return x_sorted, y_sorted
