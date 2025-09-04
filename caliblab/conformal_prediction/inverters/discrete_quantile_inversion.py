from enum import Enum
from typing import Optional

import numpy as np

from .cdf_inverter_base import CDF_inverter_base


class QuantileMethod(str, Enum):
    NEAREST = "nearest"


class DiscreteQuantileInversion(CDF_inverter_base):
    def __init__(self, score_type: str, quantile_method: str = QuantileMethod.NEAREST):
        self.score_type = score_type
        if quantile_method not in [e.value for e in QuantileMethod]:
            raise ValueError(f"quantile_method must be one of {[e.value for e in QuantileMethod]}")
        self.quantile_method = quantile_method
        self.calib_scores_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> "DiscreteQuantileInversion":
        # It is assumed that the calibration scores are already sorted.
        self.calib_scores_ = np.sort(scores)
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

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        if self.calib_scores_ is None:
            raise RuntimeError(
                "DiscreteQuantileInversion not fitted. Call fit(...) first."
            )

        if self.quantile_method == QuantileMethod.NEAREST:
            quantiles = self._get_quantiles_nearest(test_scores)
        else:
            raise ValueError(f"Invalid quantile_method: {self.quantile_method}")

        return quantiles
