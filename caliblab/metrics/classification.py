import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from .base import LabelBasedMetricBase
from ..utils.computations import make_one_hot


class Accuracy(LabelBasedMetricBase):
    """Computes the accuracy score."""

    @property
    def name(self) -> str:
        return "accuracy"

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        y_pred = np.argmax(probs, axis=1)
        return accuracy_score(y_true, y_pred)


class RocAuc(LabelBasedMetricBase):
    """Computes the Area Under the Receiver Operating Characteristic Curve."""

    @property
    def name(self) -> str:
        return "roc_auc"

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")


class PrAuc(LabelBasedMetricBase):
    """Computes the Area Under the Precision-Recall Curve."""

    @property
    def name(self) -> str:
        return "pr_auc"

    def _compute(self, *, probs: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        y_true_one_hot = make_one_hot(y_true, n_classes=probs.shape[1])
        return average_precision_score(y_true_one_hot, probs, average="macro")
