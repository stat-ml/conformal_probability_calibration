from typing import Tuple
import numpy as np

import torch
from torch.utils.data import Dataset, Subset, random_split


def split_dataset(
    dataset: Dataset, cal_ratio: float, seed: int
) -> Tuple[Subset, Subset]:
    """
    Splits a dataset into a calibration set and a test set.

    Args:
        dataset (Dataset): The original dataset to split.
        cal_ratio (float): The proportion of the dataset to be used for calibration.
        seed (int): The random seed for reproducibility.

    Returns:
        Tuple[Subset, Subset]: A tuple containing the calibration and test subsets.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError("cal_ratio must be between 0 and 1.")

    num_samples = len(dataset)
    cal_size = int(cal_ratio * num_samples)
    test_size = num_samples - cal_size

    return random_split(
        dataset,
        [cal_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def split_data(
    outputs: np.ndarray, labels: np.ndarray, cal_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError("cal_ratio must be between 0 and 1.")

    num_samples = outputs.shape[0]
    cal_size = int(cal_ratio * num_samples)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    cal_indices = indices[:cal_size]
    test_indices = indices[cal_size:]

    cal_outputs, test_outputs = outputs[cal_indices], outputs[test_indices]
    cal_labels, test_labels = labels[cal_indices], labels[test_indices]

    return cal_outputs, test_outputs, cal_labels, test_labels
