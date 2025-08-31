from .base import CalibratorBase
from typing import Optional
import numpy as np
from dataclasses import dataclass
from typing import Set


@dataclass(frozen=True, slots=True)
class ScoreTypes:
    ONE_MINUS_PROB: str = "one_minus_prob"
    APS: str = "aps"

    @classmethod
    def all_types(cls) -> Set[str]:
        return {cls.ONE_MINUS_PROB, cls.APS}


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _to_probs(logits: Optional[np.ndarray], probs: Optional[np.ndarray]) -> np.ndarray:
    if logits is None and probs is None:
        raise ValueError("Provide either `logits` or `probs`.")
    if logits is not None:
        p = _softmax(np.asarray(logits, dtype=np.float64), axis=-1)
    else:
        p = np.asarray(probs, dtype=np.float64)
    return p


def _to_logits(logits: Optional[np.ndarray], probs: Optional[np.ndarray]) -> np.ndarray:
    if logits is None and probs is None:
        raise ValueError("Provide either `logits` or `probs`.")
    if logits is not None:
        return np.asarray(logits, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return np.log(p)


class ConformalSetHelper:
    def __init__(
        self,
        alpha: float,
        score_type: str,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        if score_type not in ScoreTypes.all_types():
            raise ValueError(f"score_type must be one of {ScoreTypes.all_types()}.")

        self.alpha = float(alpha)
        self.score_type = score_type

        self.q_hat_: Optional[float] = None

    @staticmethod
    def _discrete_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        n = scores.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        s_sorted = np.sort(scores)
        return float(s_sorted[k - 1])

    def _one_minus_prob_scores(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        n, _ = probs.shape
        cal_smx = probs
        cal_labels = y_true
        cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
        return cal_scores

    def _aps_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n, _ = probs.shape
        cal_smx = probs
        cal_labels = y_true
        cal_pi = cal_smx.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
        cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
            range(n), cal_labels
        ]
        return cal_scores

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalSetHelper":
        p = _to_probs(logits, probs)
        y_true = np.asarray(y_true)
        if p.ndim != 2:
            raise ValueError("probs/logits must be 2D: (n, K).")
        if y_true.ndim != 1 or y_true.shape[0] != p.shape[0]:
            raise ValueError("y_true must be shape (n,) and match probs/logits rows.")

        if self.score_type == ScoreTypes.ONE_MINUS_PROB.value:
            scores = self._one_minus_prob_scores(p, y_true)
        elif self.score_type == ScoreTypes.APS.value:
            scores = self._aps_scores(p, y_true)
        else:
            raise ValueError(f"Invalid score type: {self.score_type}")

        q_hat = self._discrete_conformal_quantile(scores, self.alpha)
        self.q_hat_ = q_hat
        return self

    @staticmethod
    def _topk_mask_by_cumsum(probs: np.ndarray, threshold: float) -> np.ndarray:
        n, K = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        csum = np.cumsum(sorted_p, axis=1)
        k_star = (csum >= threshold).argmax(axis=1)
        mask = np.zeros_like(probs, dtype=bool)
        for i in range(n):
            mask[i, order[i, : k_star[i] + 1]] = True
        return mask

    def make_mask(self, base_probs: np.ndarray) -> np.ndarray:
        if self.q_hat_ is None:
            raise RuntimeError("ConformalSetHelper not fitted. Call fit(...) first.")
        p = np.asarray(base_probs, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError("base_probs must be 2D: (n, K).")

        if self.score_type == ScoreTypes.ONE_MINUS_PROB.value:
            return p >= 1 - self.q_hat_
        elif self.score_type == ScoreTypes.APS.value:
            return self._topk_mask_by_cumsum(p, threshold=self.q_hat_)
        else:
            raise ValueError(f"Invalid score type: {self.score_type}")


class ConformalMassCalibrator(CalibratorBase):
    """
    Two-bucket rescaling:
        q_in  = ((1 - alpha) / P_in)  * p  on C(x)
        q_out = (alpha / P_out)       * p  on C(x)^c
    where C(x) is the *fixed* conformal set built with ConformalSetHelper from the
    base probabilities at prediction time.
    """

    def __init__(self, score_type: str, alpha: float):
        super().__init__()
        self.alpha = float(alpha)
        self._conf = ConformalSetHelper(score_type=score_type, alpha=alpha)

    @property
    def name(self) -> str:
        return "conformal_mass"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalMassCalibrator":
        self._conf.fit(logits=logits, probs=probs, y_true=y_true)
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        p = _to_probs(logits, probs)  # (n, K)
        C = self._conf.make_mask(p)  # (n, K) fixed set

        P_in = (p * C).sum(axis=1, keepdims=True)  # (n, 1)
        P_in = np.where(
            P_in == 0, 1, P_in
        )  # this could be only for the empty set. In this case do nothing
        P_out = 1.0 - P_in
        P_out = np.where(
            P_out == 0, 1, P_out
        )  # this could be only for the full set. In this case do nothing
        coverage = 1.0 - self.alpha

        s_in = coverage / P_in
        s_out = (1.0 - coverage) / P_out
        q = p * (C * s_in + (~C) * s_out)

        if np.any(q.sum(axis=-1) != 1):
            import pdb

            pdb.set_trace()
            q /= q.sum(axis=1, keepdims=True)
        return q


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
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self._conf = ConformalSetHelper(score_type=score_type, alpha=alpha)

    @property
    def name(self) -> str:
        return "conformal_temperature"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalTemperatureCalibrator":
        self._conf.fit(logits=logits, probs=probs, y_true=y_true)
        return self

    def _mass_in_set(
        self, logit_row: np.ndarray, mask_row: np.ndarray, tau: float
    ) -> float:
        probs_tau = _softmax(logit_row / tau)
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
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        L = _to_logits(logits, probs)  # (n, K)
        p_base = _softmax(L)  # (n, K)
        C = self._conf.make_mask(p_base)

        n, _ = L.shape
        q = np.empty_like(p_base, dtype=np.float64)
        target = 1.0 - self.alpha

        for i in range(n):
            tau_i = self._bisection_tau(L[i], C[i], target)
            q[i] = _softmax(L[i] / tau_i)

            # safety: ensure no shortfall due to floating error
            mass_i = float((q[i] * C[i]).sum())
            if mass_i + 1e-12 < target:
                tau_i = max(self.tau_min, 0.999 * tau_i)
                q[i] = _softmax(L[i] / tau_i)

        return q
