from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from ..utils.validation import check_probs, check_labels


class MetricBase(ABC):
    """A base class for metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the metric."""
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> float:
        """Compute the metric."""
        return self._compute(**kwargs)

    @abstractmethod
    def _compute(self, **kwargs: Any) -> float:
        """The actual metric computation."""
        raise NotImplementedError


class LabelBasedMetricBase(MetricBase, ABC):
    """Base class for metrics that require ground truth labels."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(
        self,
        *,
        probs: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError


class TrueProbMetricBase(MetricBase, ABC):
    """Base class for metrics that require true probabilities."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(
        self,
        *,
        probs: np.ndarray,
        true_proba: np.ndarray,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError
