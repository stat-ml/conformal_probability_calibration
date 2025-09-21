from enum import Enum
from typing import Set

import numpy as np

from caliblab.utils.computations import get_cumulative_mass_scores


class ScoreTypes(str, Enum):
    thr = "thr"
    APS = "aps"

    @classmethod
    def all_types(cls) -> Set[str]:
        return {item.value for item in cls}


def thr_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n, _ = probs.shape
    cal_smx = probs
    cal_labels = y_true
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    return cal_scores


def aps_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Computes the Adaptive Prediction Set (APS) scores.

    The APS score for the true class is the cumulative sum of probabilities
    of classes with higher or equal probability, including the true class itself.
    """
    return get_cumulative_mass_scores(probs, y_true)


def compute_scores_for_all_classes(
    base_probs: np.ndarray, score_type: str
) -> np.ndarray:
    raise NotImplementedError("compute_scores_for_all_classes is not implemented.")
