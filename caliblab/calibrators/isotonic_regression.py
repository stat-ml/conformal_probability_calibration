from typing import Optional
import numpy as np
from .base import CalibratorBase


class IsotonicCalibration(CalibratorBase):
    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "IsotonicCalibration":
        raise NotImplementedError("IsotonicCalibration.fit is not implemented.")

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        raise NotImplementedError(
            "IsotonicCalibration.predict_proba is not implemented."
        )
