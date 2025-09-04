from enum import Enum
from typing import Set

import numpy as np


class ScoreTypes(str, Enum):
    ONE_MINUS_PROB = "one_minus_prob"
    APS = "aps"

    @classmethod
    def all_types(cls) -> Set[str]:
        return {item.value for item in cls}


def one_minus_prob_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n, _ = probs.shape
    cal_smx = probs
    cal_labels = y_true
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    return cal_scores


def aps_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n, _ = probs.shape
    cal_smx = probs
    cal_labels = y_true
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    return cal_scores


def compute_scores_for_all_classes(
    base_probs: np.ndarray, score_type: str
) -> np.ndarray:
    if score_type == ScoreTypes.ONE_MINUS_PROB.value:
        return 1 - base_probs
    elif score_type == ScoreTypes.APS.value:
        pi = base_probs.argsort(1)[:, ::-1]
        srt = np.take_along_axis(base_probs, pi, axis=1).cumsum(axis=1)
        scores = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)
        return scores
    else:
        raise ValueError(f"Invalid score type: {score_type}")
