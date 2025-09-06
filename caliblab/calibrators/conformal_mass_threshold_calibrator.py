from typing import Optional
import numpy as np

from .base import CalibratorBase, _to_logits
from ..conformal_prediction.conformal_set_helper import ConformalSetHelper
from ..utils.computations import softmax

class ConformalMassThresholdCalibrator(CalibratorBase):
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
        return f"conformal_mass_threshold_{self.alpha}"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "ConformalMassThresholdCalibrator":
        self._conf.fit(logits=logits, y_true=y_true)
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        logits = _to_logits(logits)
        C = self._conf.make_mask(logits)  # (n, K) fixed set
        p = softmax(logits)

        P_in = (p * C).sum(axis=1, keepdims=True)  # (n, 1)

        P_in = np.where(
            (np.isclose(P_in, 0)), 1, P_in
        )  # this could be only for the empty set or for full set. In this case do nothing
        P_out = 1.0 - P_in

        coverage = 1.0 - self.alpha

        s_in = coverage / P_in
        s_out = (1.0 - coverage) / P_out
        q = p * (C * s_in + (~C) * s_out)

        q_final = np.where(
            (np.isclose(P_in, 1.0)), p, q
        )  # this could be only for the empty set or for full set. In this case do nothing

        if not np.allclose(q_final.sum(axis=-1), 1.0, rtol=0, atol=1e-4):
            import pdb; pdb.set_trace()
            raise ValueError(f"Each row of q must sum to 1. Got min sum value: {q_final.sum(axis=-1).min()}")
        return q_final
