from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
from typing import Tuple


import numpy as np


def split_data(
    outputs: np.ndarray,
    labels: np.ndarray,
    cal_ratio: float,
    seed: int,
    subset_items: int = -1,
    do_not_stratify: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError("cal_ratio must be between 0 and 1.")

    if subset_items > 0:
        if subset_items < len(outputs):
            np.random.seed(seed)
            indices = np.random.choice(len(outputs), subset_items, replace=False)
            outputs = outputs[indices]
            labels = labels[indices]

    if outputs.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    test_ratio = 1.0 - cal_ratio

    if do_not_stratify:
        cal_outputs, test_outputs, cal_labels, test_labels = train_test_split(
            outputs, labels, test_size=test_ratio, random_state=seed
        )
    else: 
        cal_outputs, test_outputs, cal_labels, test_labels = train_test_split(
            outputs, labels, test_size=test_ratio, random_state=seed, 
            stratify=labels
        )

    return cal_outputs, test_outputs, cal_labels, test_labels
