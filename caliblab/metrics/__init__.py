"""
This module provides a set of metrics for evaluating the performance of a model.
"""

from typing import Any

from .base import (
    LabelBasedMetricBase,
    MetricBase,
    TrueProbMetricBase,
)
from .calibration_errors import (
    ClasswiseExpectedCalibrationError,
    ExpectedCalibrationError,
    MaximumCalibrationError,
)
from .classification import Accuracy, RocAuc, PrAuc
from .proper_scores import BrierScore, NegativeLogLikelihood
from .coverage_around_one_minus_alpha import CoverageAroundOneMinusAlpha
from .cumulative_mass_calibration_error import CumulativeMassCalibrationError


def get_metric(name: str, **kwargs: Any) -> MetricBase:
    """
    Factory function to get a metric instance by name.
    """
    name = name.lower().strip()
    if name == "accuracy":
        return Accuracy()
    if name == "roc_auc":
        return RocAuc()
    if name == "pr_auc":
        return PrAuc()
    elif name == "ece":
        return ExpectedCalibrationError(**kwargs)
    elif name == "mce":
        return MaximumCalibrationError(**kwargs)
    elif name == "cw-ece":
        return ClasswiseExpectedCalibrationError(**kwargs)
    elif name == "nll":
        return NegativeLogLikelihood()
    elif name == "brier_score":
        return BrierScore()
    elif name == "coverage_around_one_minus_alpha":
        return CoverageAroundOneMinusAlpha(**kwargs)
    elif name == "cmce":
        return CumulativeMassCalibrationError(**kwargs)
    else:
        raise ValueError(f"Unknown metric: {name}")


__all__ = [
    "MetricBase",
    "LabelBasedMetricBase",
    "TrueProbMetricBase",
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    "ClasswiseExpectedCalibrationError",
    "BrierScore",
    "NegativeLogLikelihood",
    "Accuracy",
    "RocAuc",
    "PrAuc",
    "get_metric",
    "CoverageAroundOneMinusAlpha",
    "CumulativeMassCalibrationError",
]
