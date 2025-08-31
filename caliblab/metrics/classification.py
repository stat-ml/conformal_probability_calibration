import numpy as np
from sklearn.metrics import accuracy_score

from .base import LabelBasedMetricBase


class Accuracy(LabelBasedMetricBase):
    """Computes the accuracy score."""

    @property
    def name(self) -> str:
        return "accuracy"

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        y_pred = np.argmax(probs, axis=1)
        return accuracy_score(y_true, y_pred)
