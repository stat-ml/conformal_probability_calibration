from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset, random_split


def split_data(
    outputs: np.ndarray, labels: np.ndarray, cal_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError("cal_ratio must be between 0 and 1.")

    test_ratio = 1.0 - cal_ratio

    cal_outputs, test_outputs, cal_labels, test_labels = train_test_split(
        outputs, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )

    return cal_outputs, test_outputs, cal_labels, test_labels
