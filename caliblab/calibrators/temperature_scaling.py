from typing import Optional
import numpy as np
from .base import CalibratorBase
from ..utils.computations import softmax


class TemperatureScaling(CalibratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.temperature: float = 1.0

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "TemperatureScaling":
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if probs is None:
            probs = softmax(logits)
        return probs
