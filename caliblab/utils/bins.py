from typing import Tuple
import numpy as np


def get_bin_boundaries(n_bins: int, strategy: str = "uniform") -> np.ndarray:
    """
    Calculates the boundaries for a set of bins.

    Args:
        n_bins (int): The number of bins.
        strategy (str): The strategy for determining bin boundaries.
                        Currently, only "uniform" is supported.

    Returns:
        np.ndarray: An array of bin boundaries of shape (n_bins + 1,).
    """
    if strategy == "uniform":
        return np.linspace(0, 1, n_bins + 1)
    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented")


def get_bin_lowers_uppers(
    n_bins: int, strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the lower and upper boundaries for a set of bins.

    Args:
        n_bins (int): The number of bins.
        strategy (str): The strategy for determining bin boundaries.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper
                                       bin boundaries.
    """
    boundaries = get_bin_boundaries(n_bins, strategy)
    return boundaries[:-1], boundaries[1:]
