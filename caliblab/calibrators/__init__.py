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
    )
}


def get_calibrator(name: str) -> CalibratorBase:
    if name not in CALIBRATORS:
        raise ValueError(f"Unknown calibrator: {name}")
    return CALIBRATORS[name]


__all__ = [
    "CalibratorBase",
    "TemperatureScaling",
    "IsotonicRegression",
    "ConformalCalibrator",
    "CALIBRATORS",
    "get_calibrator",
]
