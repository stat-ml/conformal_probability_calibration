import numpy as np

from caliblab.metrics.base import LabelBasedMetricBase, MetricComputeInput
from caliblab.utils.computations import cumulative_mass_and_coverage


class AlphaSuffixCoverage(LabelBasedMetricBase):
    """
    Empirical coverage of the minimal prediction set whose cumulative probability
    mass is at least `1 - alpha`.

    For each example, we consider the smallest top-k set (in descending
    probability order) such that cumulative mass is >= (1 - alpha), and check
    whether the true label lies in this set. The metric is the average of this
    indicator over examples.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.requires_labels = True

        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

    @property
    def name(self) -> str:
        a = round(self.alpha, 3)
        return f"alpha_suffix_coverage_{a}"

    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        probs = metric_input.probs
        y_true = metric_input.y_true
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n_samples, n_classes)")
        if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
            raise ValueError("y_true must be a 1D array aligned with probs rows")

        cum_probs, _, sorted_indices = cumulative_mass_and_coverage(probs, y_true)

        # Find smallest k such that cumulative mass reaches (1 - alpha).
        mask = cum_probs >= 1 - self.alpha
        first_true_idx = mask.argmax(axis=1)

        # Numerical guard: if no threshold crossing occurs, use the full set.
        no_true_row = ~mask.any(axis=1)
        if np.any(no_true_row):
            n_classes = cum_probs.shape[1]
            first_true_idx[no_true_row] = n_classes - 1

        # Aggregate once per sample: membership in the first threshold-crossing set.
        true_rank = np.where(sorted_indices == y_true[:, np.newaxis])[1]
        selected_coverage = true_rank <= first_true_idx
        return float(selected_coverage.mean())



class AlphaSuffixCoverageDifference(AlphaSuffixCoverage):
    """
    Signed gap between alpha-suffix coverage and the target (1 - alpha).
    """

    @property
    def name(self) -> str:
        a = round(self.alpha, 3)
        return f"alpha_suffix_coverage_difference_{a}"

    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        alpha_suffix_coverage = super()._compute(metric_input=metric_input)
        return float((alpha_suffix_coverage - (1 - self.alpha)))


AlfaSuffixCoverageDifference = AlphaSuffixCoverageDifference
