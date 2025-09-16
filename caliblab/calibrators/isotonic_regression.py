from typing import Optional
from ..utils.computations import softmax

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression

from .base import CalibratorBase


class IsotonicRegression(CalibratorBase):
    def __init__(self):
        super().__init__()
        self.calibrators = None

    @property
    def name(self) -> str:
        return "isotonic"

    def fit(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "IsotonicRegression":
        if probs is None:
            if logits is None:
                raise ValueError(
                    "Either logits or probs must be provided to IsotonicRegression."
                )
            probs = softmax(logits)

        n_classes = probs.shape[1]
        self.calibrators = [
            SklearnIsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            for _ in range(n_classes)
        ]

        for k in range(n_classes):
            # Create a binary target for the current class
            y_binary = (y_true == k).astype(int)
            self.calibrators[k].fit(probs[:, k], y_binary)

        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if probs is None:
            if logits is None:
                raise ValueError(
                    "Either logits or probs must be provided to IsotonicRegression."
                )
            probs = softmax(logits)

        n_samples, n_classes = probs.shape
        calibrated_probs = np.zeros_like(probs)

        for k in range(n_classes):
            calibrated_probs[:, k] = self.calibrators[k].transform(probs[:, k])

        # Normalize probabilities to sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)

        # Using errstate to avoid warnings for division by zero, which is handled.
        with np.errstate(divide="ignore", invalid="ignore"):
            # For rows with a sum of 0, use a uniform distribution.
            normalized_probs = np.where(
                row_sums == 0,
                1.0 / calibrated_probs.shape[1],
                calibrated_probs / row_sums,
            )

        return normalized_probs


__all__ = ["IsotonicRegression"]
