from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
from ..utils.validation import check_probs, check_labels


@dataclass(frozen=True)
class MetricComputeInput:
    probs: np.ndarray
    y_true: Optional[np.ndarray] = None
    true_proba: Optional[np.ndarray] = None
    uncalibrated_probs: Optional[np.ndarray] = None


class MetricBase(ABC):
    """A base class for metrics."""

    def __init__(self):
        self.requires_labels = False
        self.requires_true_proba = False
        self.requires_uncalibrated_probs = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the metric."""
        raise NotImplementedError

    def __call__(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        true_proba: Optional[np.ndarray] = None,
        uncalibrated_probs: Optional[np.ndarray] = None,
    ) -> float:
        metric_input = MetricComputeInput(
            probs=probs,
            y_true=y_true,
            true_proba=true_proba,
            uncalibrated_probs=uncalibrated_probs,
        )
        self._validate_inputs(metric_input=metric_input)
        return self._compute(metric_input=metric_input)

    @abstractmethod
    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        """The actual metric computation."""
        raise NotImplementedError

    def _validate_inputs(
        self,
        *,
        metric_input: MetricComputeInput,
    ) -> None:
        probs = metric_input.probs
        y_true = metric_input.y_true
        true_proba = metric_input.true_proba
        uncalibrated_probs = metric_input.uncalibrated_probs

        check_probs(probs, name="probs")
        _, k = probs.shape
        if uncalibrated_probs is not None:
            check_probs(uncalibrated_probs, name="uncalibrated_probs")
            if uncalibrated_probs.shape != probs.shape:
                raise ValueError("uncalibrated_probs and probs must have the same shape.")
        if self.requires_uncalibrated_probs and uncalibrated_probs is None:
            raise ValueError(f"{self.name} requires `uncalibrated_probs`.")
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


class LabelBasedMetricBase(MetricBase, ABC):
    """Base class for metrics that require ground truth labels."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        raise NotImplementedError


class TrueProbMetricBase(MetricBase, ABC):
    """Base class for metrics that require true probabilities."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        raise NotImplementedError


class PairwiseProbMetricBase(MetricBase, ABC):
    """Base class for metrics that compare calibrated and uncalibrated probabilities."""

    def __init__(self):
        super().__init__()
        self.requires_uncalibrated_probs = True

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(self, *, metric_input: MetricComputeInput) -> float:
        raise NotImplementedError
