"""
This module provides a set of metrics for evaluating the performance of a model.
"""
from .base import (
    LabelBasedMetricBase,
    MetricBase,
    TrueProbMetricBase,
)
from .calibration_errors import ECE, MCE, CWECE
from .classification import Accuracy
from .getters import get_metric
from .proper_scores import BrierScore, NLL

__all__ = [
    "MetricBase",
    "LabelBasedMetricBase",
    "TrueProbMetricBase",
    "ECE",
    "MCE",
    "CWECE",
    "BrierScore",
    "NLL",
    "Accuracy",
    "get_metric",
]
