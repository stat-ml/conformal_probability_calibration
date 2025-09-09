from typing import Optional
import numpy as np

from .base import CalibratorBase
from ..conformal_prediction.conformal_set_helper import (
    ConformalSetHelper,
    ScoreTransformation,
)
from ..utils.computations import softmax


class ConformalTemperatureCalibrator(CalibratorBase):
    """
    Per-example temperature bisection with a *fixed* conformal set:

      For each x:
        1) Build C(x) using base probabilities and the fitted conformal threshold.
        2) Find the largest tau such that sum_{y in C(x)} softmax(logits/tau)[y] >= 1 - alpha.
           (closest-from-above; never undershoot.)
    """

    def __init__(
        self,
        score_type: str,
        alpha: float = 0.1,
        tol: float = 1e-10,
        max_iter: int = 60,
        tau_min: float = 1e-4,
        tau_max: float = 1e6,
        score_transformation: str = ScoreTransformation.IDENTITY.value,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self._conf = ConformalSetHelper(
            score_type=score_type,
            alpha=alpha,
            score_transformation=score_transformation,
        )

    @property
    def name(self) -> str:
        return f"conformal_temperature_alpha={self.alpha}"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalTemperatureCalibrator":
        self._conf.fit(logits=logits, y_true=y_true)
        self._mark_fitted()
        return self

    def _mass_in_set(
        self, logit_row: np.ndarray, mask_row: np.ndarray, tau: float
    ) -> float:
        probs_tau = softmax(logit_row / tau)
        return float((probs_tau * mask_row).sum())

    def _bisection_tau(
        self, logit_row: np.ndarray, mask_row: np.ndarray, target: float
    ) -> float:
        # Ensure we have a bracket [tau_lo, tau_hi] with r(tau_lo) >= target >= r(tau_hi).
        tau_lo = self.tau_min
        r_lo = self._mass_in_set(logit_row, mask_row, tau_lo)
        if r_lo < target - self.tol:
            # If even at extreme sharpening we can't hit target (rare), return tau_lo.
            return tau_lo

        tau_hi = 1.0
        r_hi = self._mass_in_set(logit_row, mask_row, tau_hi)
        it_expand = 0
        while r_hi > target and tau_hi < self.tau_max and it_expand < 40:
            tau_hi *= 2.0
            r_hi = self._mass_in_set(logit_row, mask_row, tau_hi)
            it_expand += 1

        if r_hi > target:
            # Could not drop below target within bounds; return cap (closest-from-above).
            return tau_hi

        # Bisection: r(tau) is strictly decreasing in tau for these HDR-style sets.
        for _ in range(self.max_iter):
            tau_mid = 0.5 * (tau_lo + tau_hi)
            r_mid = self._mass_in_set(logit_row, mask_row, tau_mid)
            if r_mid >= target:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid
            if abs(tau_hi - tau_lo) <= self.tol * max(1.0, tau_lo):
                break

        return tau_lo  # largest tau with r >= target (no shortfall)

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        L = logits
        p_base = softmax(logits)
        C = self._conf.make_mask(logits)

        n, _ = L.shape
        q = np.empty_like(p_base, dtype=np.float64)
        target = 1.0 - self.alpha

        for i in range(n):
            tau_i = self._bisection_tau(L[i], C[i], target)
            q[i] = softmax(L[i] / tau_i)

            # safety: ensure no shortfall due to floating error
            mass_i = float((q[i] * C[i]).sum())
            if mass_i + 1e-12 < target:
                tau_i = max(self.tau_min, 0.999 * tau_i)
                q[i] = softmax(L[i] / tau_i)

        return q

    def uses_conformal_set_helper(self) -> bool:
        return True

    def get_conformal_set_sizes(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        C = self._conf.make_mask(logits)  # (n, K) fixed set
        return C.sum(axis=1)
