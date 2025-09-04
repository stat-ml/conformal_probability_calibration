from __future__ import annotations

from typing import Dict, Type

from .base import CalibratorBase
from .conformal_calibrator import ConformalCalibrator
from .isotonic_regression import IsotonicRegression
from .temperature_scaling import TemperatureScaling

CALIBRATORS: Dict[str, CalibratorBase] = {
    "temperature_scaling": TemperatureScaling(),
    "isotonic_regression": IsotonicRegression(),
    "conformal_aps_nearest": ConformalCalibrator(
        score_type="aps", quantile_method="nearest"
    ),
    "conformal_aps_linear": ConformalCalibrator(
        score_type="aps", quantile_method="linear"
    ),
    "conformal_one_minus_prob_nearest": ConformalCalibrator(
        score_type="one_minus_prob", quantile_method="nearest"
    ),
    "conformal_one_minus_prob_linear": ConformalCalibrator(
        score_type="one_minus_prob", quantile_method="linear"
    ),
}

__all__ = [
    "CalibratorBase",
    "TemperatureScaling",
    "IsotonicRegression",
    "ConformalCalibrator",
    "CALIBRATORS",
]
