import numpy as np

from caliblab.metrics.base import LabelBasedMetricBase, MetricComputeInput
from caliblab.utils.computations import cumulative_mass_and_coverage


class CoverageAroundOneMinusAlpha(LabelBasedMetricBase):
    def __init__(self, alpha: float, eps: float):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.requires_labels = True

        if eps >= 0:
            self.lower_bound = 1.0 - self.alpha
            self.upper_bound = 1.0 - self.alpha + self.eps
        else:
            self.lower_bound = 1.0 - self.alpha + self.eps
            self.upper_bound = 1.0 - self.alpha

        # Keep bounds within [0, 1]
        self.lower_bound = max(0.0, self.lower_bound)
        self.upper_bound = min(1.0, self.upper_bound)

    @property
    def name(self) -> str:
        lb = round(self.lower_bound, 3)
        ub = round(self.upper_bound, 3)
        return f"coverage_[{lb}, {ub}]"

    def compute_from_cumsum(
        self, *, cum_probs: np.ndarray, true_rank: np.ndarray
    ) -> float:
        """
        Computes the empirical coverage from precomputed cumulative sums and true ranks.

        Args:
            cum_probs: Cumulative sums of descending-sorted probabilities, shape (n_samples, n_classes).
            true_rank: Rank of the true class in descending order, shape (n_samples,).
        """
        if cum_probs.ndim != 2:
            raise ValueError("cum_probs must be a 2D array of shape (n_samples, n_classes)")
        if true_rank.ndim != 1 or true_rank.shape[0] != cum_probs.shape[0]:
            raise ValueError("true_rank must be a 1D array aligned with cum_probs rows")

        n_samples, n_classes = cum_probs.shape
        lb, ub = self.lower_bound, self.upper_bound

        if ub < lb:
            raise ValueError(f"upper bound {ub} is less than lower bound {lb}")

        L = (cum_probs < lb).sum(axis=1)
        R = (cum_probs <= ub).sum(axis=1) - 1
        R = np.clip(R, -1, n_classes - 1)

        counts = np.maximum(R - L + 1, 0)
        total_sets = int(counts.sum())
        if total_sets == 0:
            return 0.0, 0

        start_cover = np.maximum(L, true_rank)
        covers = np.maximum(R - start_cover + 1, 0)
        total_cover = int(covers.sum())

        return float(total_cover / total_sets), total_sets

    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        """
        Empirical coverage over all top-k prediction sets whose cumulative mass
        lies in [lower_bound, upper_bound].
        """
        probs = metric_input.probs
        y_true = metric_input.y_true
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n_samples, n_classes)")
        if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
            raise ValueError("y_true must be a 1D array aligned with probs rows")

        # Use shared utility to get cumulative sums and sorting indices
        cum_probs, _coverage_matrix, sorted_idx = cumulative_mass_and_coverage(
            probs, y_true
        )
        true_rank = np.where(sorted_idx == y_true[:, np.newaxis])[1]

        return self.compute_from_cumsum(cum_probs=cum_probs, true_rank=true_rank)[0]
