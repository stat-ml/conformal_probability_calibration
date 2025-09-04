from typing import Optional

import numpy as np
from scipy.special import softmax

from .inverters.discrete_quantile_inversion import (
    DiscreteQuantileInversion,
    QuantileMethod,
)
from .score_functions import (
    ScoreTypes,
    aps_scores,
    compute_scores_for_all_classes,
    one_minus_prob_scores,
)


def _to_probs(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=1)


class ConformalPredictor:
    def __init__(
        self,
        score_type: str,
        quantile_method: str = QuantileMethod.NEAREST,
    ):
        if score_type not in ScoreTypes.all_types():
            raise ValueError(f"score_type must be one of {ScoreTypes.all_types()}.")

        self.score_type = score_type
        self.quantile_inversion = DiscreteQuantileInversion(
            score_type, quantile_method
        )

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalPredictor":
        p = _to_probs(logits)
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

        self.quantile_inversion.fit(np.sort(scores))
        return self

    def predict(self, base_probs: np.ndarray) -> np.ndarray:
        test_scores = compute_scores_for_all_classes(base_probs, self.score_type)
        quantiles = self.quantile_inversion.predict(test_scores)
        return quantiles
