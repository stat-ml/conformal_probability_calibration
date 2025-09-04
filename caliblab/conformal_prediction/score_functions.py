from enum import Enum
from typing import Set

import numpy as np

from caliblab.utils.computations import get_cumulative_mass_scores


class ScoreTypes(str, Enum):
    ONE_MINUS_PROB = "one_minus_prob"
    APS = "aps"

    @classmethod
    def all_types(cls) -> Set[str]:
        return {item.value for item in cls}

def aps_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Computes the Adaptive Prediction Set (APS) scores.

    The APS score for the true class is the cumulative sum of probabilities
    of classes with higher or equal probability, including the true class itself.
    """
    return get_cumulative_mass_scores(probs, y_true)


def compute_aps_scores_for_all_classes(
    base_probs: np.ndarray, score_type: str
) -> np.ndarray:
    if score_type == ScoreTypes.ONE_MINUS_PROB.value:
        return 1 - base_probs
    elif score_type == ScoreTypes.APS.value:
        # Sort probabilities in descending order
        sorted_indices = np.argsort(-base_probs, axis=1)
        sorted_probs = np.take_along_axis(base_probs, sorted_indices, axis=1)

        # Calculate cumulative probability sums
        cum_probs = np.cumsum(sorted_probs, axis=1)

        # Get the ranks to revert the sorting
        ranks = np.argsort(sorted_indices, axis=1)

        # Map cumulative probabilities back to original class order
        scores = np.take_along_axis(cum_probs, ranks, axis=1)

        ### FATAL: This line is wrong. NORMALISATION.
        scores = scores / scores.max(axis=1, keepdims=True)

        if not np.allclose(scores.max(axis=1), 1.0, atol=1e-10, rtol=0):
            raise ValueError(f"Highest score should be 1.")

        return scores
    else:
        raise ValueError(f"Invalid score type: {score_type}")
