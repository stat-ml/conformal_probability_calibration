from enum import Enum
from typing import Optional

import numpy as np

from .cdf_inverter_base import CDF_inverter_base
from ..score_functions import ScoreTypes


class QuantileMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"


class DiscreteQuantileInversion(CDF_inverter_base):
    def __init__(self, score_type: str, quantile_method: str = QuantileMethod.NEAREST):
        self.score_type = score_type
        if quantile_method not in [e.value for e in QuantileMethod]:
            raise ValueError(f"quantile_method must be one of {[e.value for e in QuantileMethod]}")
        self.quantile_method = quantile_method
        self.calib_scores_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> "DiscreteQuantileInversion":
        # It is assumed that the calibration scores are already sorted.
        self.calib_scores_ = scores
        return self

    def _get_quantiles_nearest(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Computes the empirical CDF of the calibration scores for each test score.
        This is the proportion of calibration scores less than or equal to the test score.
        """
        n = self.calib_scores_.shape[0]
        if n == 0:
            return np.zeros_like(test_scores)

        # Find the number of calibration scores less than or equal to the test scores.
        num_le = np.searchsorted(self.calib_scores_, test_scores, side="right")

        # The quantile is the proportion of scores <= test_score.
        quantiles = num_le / n

        return quantiles

    def _get_quantiles_linear(self, test_scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Linear interpolation is not implemented for APS scores."
        )
        n = self.calib_scores_.shape[0]

        # Find indices for interpolation
        right_indices = np.searchsorted(self.calib_scores_, test_scores, side="right")
        left_indices = right_indices - 1

        # Get calibration scores at indices, handling out-of-bounds access
        s_i = np.where(left_indices >= 0, self.calib_scores_[left_indices], -np.inf)
        s_i_plus_1 = np.where(
            right_indices < n, self.calib_scores_[right_indices], np.inf
        )

        # Quantile is defined as (index + 1) / n for a sorted array
        q_i = (left_indices + 1) / n
        q_i_plus_1 = (right_indices + 1) / n

        # Perform linear interpolation
        fraction = (test_scores - s_i) / (s_i_plus_1 - s_i + 1e-9)
        quantiles = q_i + fraction * (q_i_plus_1 - q_i)

        # Handle scores that are outside the range of calibration scores
        quantiles[test_scores < self.calib_scores_[0]] = 0.0
        quantiles[test_scores > self.calib_scores_[-1]] = 1.0

        if self.score_type == ScoreTypes.ONE_MINUS_PROB:
            quantiles = 1 - quantiles

        return quantiles

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        if self.calib_scores_ is None:
            raise RuntimeError(
                "DiscreteQuantileInversion not fitted. Call fit(...) first."
            )

        if self.quantile_method == QuantileMethod.NEAREST:
            quantiles = self._get_quantiles_nearest(test_scores)
        elif self.quantile_method == QuantileMethod.LINEAR:
            quantiles = self._get_quantiles_linear(test_scores)
        else:
            raise ValueError(f"Invalid quantile_method: {self.quantile_method}")

        return quantiles
