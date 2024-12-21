def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed (int): The seed value to use
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
