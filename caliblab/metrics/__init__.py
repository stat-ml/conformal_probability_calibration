"""
This module provides a set of metrics for evaluating the performance of a model.
"""
from typing import Any

from .base import (
    LabelBasedMetricBase,
    MetricBase,
    TrueProbMetricBase,
)
from .calibration_errors import CWECE, ECE, MCE
from .classification import Accuracy
from .proper_scores import BrierScore, NegativeLogLikelihood


def get_metric(name: str, **kwargs: Any) -> MetricBase:
    """
    Factory function to get a metric instance by name.
    """
    name = name.lower().strip()
    if name == "accuracy":
        return Accuracy()
    elif name == "ece":
        return ECE(**kwargs)
    elif name == "mce":
        return MCE(**kwargs)
    elif name == "cwece":
        return CWECE(**kwargs)
    elif name == "nll":
        return NegativeLogLikelihood()
    elif name == "brier_score":
        return BrierScore()
    else:
        raise ValueError(f"Unknown metric: {name}")


__all__ = [
    "MetricBase",
    "LabelBasedMetricBase",
    "TrueProbMetricBase",
    "ECE",
    "MCE",
    "CWECE",
    "BrierScore",
    "NegativeLogLikelihood",
    "Accuracy",
    "get_metric",
]
