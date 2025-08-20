from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from ..utils.validation import check_probs, check_labels


class MetricBase(ABC):
    requires_labels: bool = False
    requires_true_proba: bool = False

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def __call__(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        true_proba: Optional[np.ndarray] = None,
    ) -> float:
        self._validate_inputs(
            probs=probs,
            y_true=y_true,
            true_proba=true_proba,
        )
        return self.compute(
            probs=probs,
            y_true=y_true,
            true_proba=true_proba,
        )

    def _validate_inputs(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray],
        true_proba: Optional[np.ndarray],
    ) -> None:
        check_probs(probs, name="probs")
        _, k = probs.shape
        if self.requires_labels:
            if y_true is None:
                raise ValueError(f"{self.name} requires `y_true`.")
            check_labels(y_true, n_classes=k)
        if self.requires_true_proba:
            if true_proba is None:
                raise ValueError(f"{self.name} requires `true_proba`.")
            check_probs(true_proba, name="true_proba")
            if true_proba.shape != probs.shape:
                raise ValueError("true_proba and probs must have the same shape.")

    @abstractmethod
    def compute(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray],
        true_proba: Optional[np.ndarray],
    ) -> float:
        raise NotImplementedError


class LabelBasedMetricBase(MetricBase):
    requires_labels: bool = True
    requires_true_proba: bool = False


class TrueProbMetricBase(MetricBase):
    requires_labels: bool = False
    requires_true_proba: bool = True
