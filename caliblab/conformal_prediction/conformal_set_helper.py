from typing import Optional
import numpy as np

from ..utils.computations import softmax
from .score_functions import ScoreTypes, one_minus_prob_scores, aps_scores
from ..utils.computations import softmax


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

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalSetHelper":
        p = softmax(logits)
        y_true = np.asarray(y_true)
        if p.ndim != 2:
            raise ValueError("probs/logits must be 2D: (n, K).")
        if y_true.ndim != 1 or y_true.shape[0] != p.shape[0]:
            raise ValueError("y_true must be shape (n,) and match probs/logits rows.")

        if self.score_type == ScoreTypes.ONE_MINUS_PROB.value:
            scores = one_minus_prob_scores(p, y_true)
        elif self.score_type == ScoreTypes.APS.value:
            scores = aps_scores(p, y_true)
        else:
            raise ValueError(f"Invalid score type: {self.score_type}")

        q_hat = self._discrete_conformal_quantile(scores, self.alpha)
        self.q_hat_ = q_hat
        return self

    @staticmethod
    def _topk_mask_by_cumsum(probs: np.ndarray, threshold: float) -> np.ndarray:
        n, _ = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        csum = np.cumsum(sorted_p, axis=1)
        k_star = (csum >= threshold).argmax(axis=1)
        mask = np.zeros_like(probs, dtype=bool)
        for i in range(n):
            mask[i, order[i, : k_star[i] + 1]] = True
        return mask

    def make_mask(self, base_logits: np.ndarray) -> np.ndarray:
        if self.q_hat_ is None:
            raise RuntimeError("ConformalSetHelper not fitted. Call fit(...) first.")
        p = np.asarray(softmax(base_logits), dtype=np.float64)
        if p.ndim != 2:
            raise ValueError("base_probs must be 2D: (n, K).")

        if self.score_type == ScoreTypes.ONE_MINUS_PROB.value:
            return p >= 1 - self.q_hat_
        elif self.score_type == ScoreTypes.APS.value:
            return self._topk_mask_by_cumsum(p, threshold=self.q_hat_)
        else:
            raise ValueError(f"Invalid score type: {self.score_type}")
