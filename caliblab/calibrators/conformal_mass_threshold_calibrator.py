from typing import Optional
import numpy as np

from .base import CalibratorBase
from ..conformal_prediction.conformal_set_helper import (
    ConformalSetHelper,
    ScoreTransformation,
)
from ..utils.computations import softmax


class ConformalMassThresholdCalibrator(CalibratorBase):
    """
    Two-bucket rescaling:
        q_in  = ((1 - alpha) / P_in)  * p  on C(x)
        q_out = (alpha / P_out)       * p  on C(x)^c
    where C(x) is the *fixed* conformal set built with ConformalSetHelper from the
    base probabilities at prediction time.
    """

    def __init__(
        self,
        score_type: str,
        alpha: float,
        score_transformation: ScoreTransformation = ScoreTransformation.IDENTITY,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self._conf = ConformalSetHelper(
            score_type=score_type,
            alpha=alpha,
            score_transformation=score_transformation,
        )

    @property
    def name(self) -> str:
        return f"cnfrml_mass_thrsh:a={self.alpha},sс.tp={self._conf.score_type[:3]},sс.trnf={self._conf.score_transformation[:4]}"

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
        p = softmax(logits)
        C = self._conf.make_mask(logits)  # (n, K) fixed set
        nontrivial_sets = np.all(C, axis=-1) ^ np.any(C, axis=-1)

        P_in = (p * C).sum(axis=1)  # (n, 1)
        P_out = 1.0 - P_in
        coverage = 1.0 - self.alpha

        s_in  = np.ones_like(P_in)
        s_out = np.ones_like(P_out)
        np.divide(coverage,        P_in,  out=s_in,  where=nontrivial_sets)
        np.divide(1.0 - coverage,  P_out, out=s_out, where=nontrivial_sets)

        s_in  = s_in[:,  None]
        s_out = s_out[:, None]

        q_final = p * (C * s_in + (~C) * s_out)

        if not np.allclose(q_final.sum(axis=-1), 1.0, rtol=0, atol=1e-4):
            raise ValueError(
                f"Each row of q must sum to 1. Got min sum value: {q_final.sum(axis=-1).min()}"
            )
        return q_final

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
