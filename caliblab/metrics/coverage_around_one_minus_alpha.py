import numpy as np

from caliblab.metrics.base import LabelBasedMetricBase


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

    def compute_from_sorted(
        self, *, probs: np.ndarray, y_true: np.ndarray, sorted_idx: np.ndarray, **kwargs
    ) -> float:
        """
        Computes the empirical coverage from pre-sorted probability indices.
        """
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n_samples, n_classes)")
        if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
            raise ValueError("y_true must be a 1D array aligned with probs rows")

        n_samples, n_classes = probs.shape
        lb, ub = self.lower_bound, self.upper_bound

        if ub < lb:
            return 0.0

        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cum_probs = np.cumsum(sorted_probs, axis=1)

        inv_idx = np.empty_like(sorted_idx)
        row_ids = np.arange(n_samples)[:, None]
        inv_idx[row_ids, sorted_idx] = np.arange(n_classes)[None, :]
        true_rank = inv_idx[np.arange(n_samples), y_true.astype(int)]

        L = (cum_probs < lb).sum(axis=1)
        R = (cum_probs <= ub).sum(axis=1) - 1
        R = np.clip(R, -1, n_classes - 1)

        counts = np.maximum(R - L + 1, 0)
        total_sets = int(counts.sum())
        if total_sets == 0:
            return 0.0

        start_cover = np.maximum(L, true_rank)
        covers = np.maximum(R - start_cover + 1, 0)
        total_cover = int(covers.sum())

        return float(total_cover / total_sets)

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        """
        Empirical coverage over all top-k prediction sets whose cumulative mass
        lies in [lower_bound, upper_bound].
        """
        sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
        return self.compute_from_sorted(
            probs=probs, y_true=y_true, sorted_idx=sorted_idx, **kwargs
        )
