import numpy as np

from caliblab.metrics.base import LabelBasedMetricBase
from caliblab.utils.computations import get_cumulative_mass_scores


class CoverageAroundOneMinusAlpha(LabelBasedMetricBase):
    def __init__(self, alpha: float, eps: float):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.requires_labels = True

    @property
    def name(self) -> str:
        return f"coverage_[{1 - self.alpha - self.eps}, {1 - self.alpha + self.eps}]"

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        """
        Computes the empirical coverage for all prediction sets whose cumulative
        probability mass falls within the range [1-alpha-eps, 1-alpha+eps].
        """
        n_samples, n_classes = probs.shape

        # Step 1: Sort probabilities and calculate cumulative mass for each sample
        sorted_indices = np.argsort(-probs, axis=1)
        sorted_probs = -np.sort(-probs, axis=1)
        cum_probs = np.cumsum(sorted_probs, axis=1)

        # Step 2: Find the rank of the true class for each sample
        true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

        # Step 3: Create a coverage matrix indicating if the true class is in sets of increasing size
        ranks = np.arange(n_classes)
        coverage_matrix = true_class_ranks[:, np.newaxis] <= ranks[np.newaxis, :]

        # Step 4: Flatten the cumulative masses and coverages to get all (p, c) pairs
        all_cum_scores = cum_probs.flatten()
        all_coverages = coverage_matrix.flatten()

        # Step 5: Filter the pairs where cumulative mass is in the desired interval
        lower_bound = 1 - self.alpha - self.eps
        upper_bound = 1 - self.alpha + self.eps
        mask = (all_cum_scores >= lower_bound) & (all_cum_scores <= upper_bound)

        if mask.sum() == 0:
            return 0.0

        # Step 6: Compute the mean coverage for the filtered subset
        subset_coverages = all_coverages[mask]
        coverage = subset_coverages.astype(float).mean()

        return coverage
