from typing import Optional

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
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "IsotonicRegression":
        if probs is None:
            if logits is None:
                raise ValueError(
                    "Either logits or probs must be provided to IsotonicRegression."
                )
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

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
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if probs is None:
            if logits is None:
                raise ValueError(
                    "Either logits or probs must be provided to IsotonicRegression."
                )
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

        n_samples, n_classes = probs.shape
        calibrated_probs = np.zeros_like(probs)

        for k in range(n_classes):
            calibrated_probs[:, k] = self.calibrators[k].transform(probs[:, k])

        # Normalize probabilities to sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        normalized_probs = calibrated_probs / safe_row_sums

        return normalized_probs


__all__ = ["IsotonicRegression"]
