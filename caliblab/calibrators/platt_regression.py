from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils.computations import softmax
from .base import CalibratorBase


def _safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute elementwise logit with numerical safety.

    For values near 0 or 1, clip to avoid infinities.
    """
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped) - np.log(1.0 - p_clipped)


class PlattRegression(CalibratorBase):
    """
    One-vs-rest Platt (logistic) regression calibration.

    For each class k, we fit a logistic regression mapping from a scalar score
    to a calibrated probability for the binary event {y == k}. The scalar score
    is the raw logit for class k when logits are provided; otherwise we use the
    logit of the predicted probability for class k.

    At prediction time, we apply the per-class mappings independently and then
    renormalize the resulting scores to ensure the per-sample probabilities sum
    to 1 across classes.
    """

    def __init__(self):
        super().__init__()
        self.models: Optional[list[LogisticRegression]] = None
        self._trained_on: Optional[str] = None  # "logits" or "probs"

    @property
    def name(self) -> str:
        return "platt_regression"

    def fit(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "PlattRegression":
        if probs is None and logits is None:
            raise ValueError("Either logits or probs must be provided to PlattRegression.")

        if logits is not None:
            scores = logits.astype(np.float64)
            self._trained_on = "logits"
        else:
            probs = probs.astype(np.float64)
            scores = _safe_logit(probs)
            self._trained_on = "probs"

        n_samples, n_classes = scores.shape
        self.models = []

        for k in range(n_classes):
            x_k = scores[:, k].reshape(-1, 1)
            y_k = (y_true == k).astype(int)

            # Use a small amount of L2 regularization (default C=1.0) and lbfgs solver
            model = LogisticRegression(solver="lbfgs")
            # In degenerate cases where y_k has a single class, sklearn will error.
            # Handle by setting a constant model: probability is the base rate.
            if np.unique(y_k).size < 2:
                base_rate = float(np.mean(y_k))

                class ConstantModel:
                    def predict_proba(self, X):  # type: ignore[no-redef]
                        n = X.shape[0]
                        return np.stack([1.0 - np.full(n, base_rate), np.full(n, base_rate)], axis=1)

                self.models.append(ConstantModel())
                continue

            model.fit(x_k, y_k)
            self.models.append(model)

        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if self.models is None:
            raise RuntimeError("PlattRegression models are not initialized.")

        if logits is None and probs is None:
            raise ValueError("Either logits or probs must be provided to PlattRegression.")

        # Construct the score matrix consistent with training
        if self._trained_on == "logits":
            if logits is None:
                raise ValueError("PlattRegression was trained on logits; provide logits at prediction time.")
            scores = logits.astype(np.float64)
        else:
            if probs is None:
                if logits is None:
                    raise ValueError("Either logits or probs must be provided to PlattRegression.")
                probs = softmax(logits)
            scores = _safe_logit(probs.astype(np.float64))

        n_samples, n_classes = scores.shape
        calibrated = np.zeros((n_samples, n_classes), dtype=np.float64)

        for k in range(n_classes):
            x_k = scores[:, k].reshape(-1, 1)
            calibrated[:, k] = self.models[k].predict_proba(x_k)[:, 1]

        # Normalize to ensure probabilities sum to 1 per sample.
        row_sums = calibrated.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(row_sums == 0.0, 1.0 / n_classes, calibrated / row_sums)

        return normalized


__all__ = ["PlattRegression"]



