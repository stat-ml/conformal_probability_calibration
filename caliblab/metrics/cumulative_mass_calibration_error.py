from .base import LabelBasedMetricBase
import numpy as np
from typing import Optional

from ..utils.bins import get_bin_boundaries


def _compute_cumulative_mass_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int, strategy: str
) -> list:
    
    n_samples, n_classes = probs.shape

    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = -np.sort(-probs, axis=1)

    cum_probs = np.cumsum(sorted_probs, axis=1)

    true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

    ranks = np.arange(n_classes)
    coverage_matrix = true_class_ranks[:, np.newaxis] <= ranks[np.newaxis, :]

    all_cum_scores = cum_probs.flatten()
    all_coverages = coverage_matrix.flatten().astype(float)

    bin_boundaries = get_bin_boundaries(all_cum_scores, n_bins, strategy)
    
    bin_diffs = []

    for i in range(len(bin_boundaries) - 1):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        
        if i == len(bin_boundaries) - 2:
            in_bin_mask = (all_cum_scores >= lower) & (all_cum_scores <= upper)
        else:
            in_bin_mask = (all_cum_scores >= lower) & (all_cum_scores < upper)
        
        if np.sum(in_bin_mask) > 0:
            
            bin_mean_score = np.mean(all_cum_scores[in_bin_mask])
            bin_mean_coverage = np.mean(all_coverages[in_bin_mask])

            bin_weight = np.sum(in_bin_mask) / len(all_cum_scores)

            bin_diff = bin_mean_coverage - bin_mean_score
            bin_diffs.append((bin_weight, bin_diff))

    return bin_diffs


class CumulativeMassCalibrationError(LabelBasedMetricBase):
    def __init__(self, n_bins: int = 15, strategy: str = "uniform", weighted: bool = True) -> None:
        super().__init__()
        self.requires_labels = True
        self.n_bins = n_bins
        if strategy not in ["uniform", "quantile"]:
            raise ValueError("strategy must be either 'uniform' or 'quantile'")
        self.strategy = strategy
        self.weighted = weighted

    @property
    def name(self) -> str:
        return "cmce"

    def _compute(
        self, *, probs, y_true: Optional[np.ndarray], true_proba: Optional[np.ndarray]
    ):
        bin_diffs = _compute_cumulative_mass_calibration_error(probs, y_true, self.n_bins, self.strategy)
        
        if self.weighted:
            return sum(weight * np.abs(diff) for weight, diff in bin_diffs)
        else:
            return np.mean([np.abs(diff) for _, diff in bin_diffs])
