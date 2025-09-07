from typing import Any

from .base import CalibratorBase
from .isotonic_regression import IsotonicRegression
from .temperature_scaling import TemperatureScaling
from .conformal_mass_threshold_calibrator import ConformalMassThresholdCalibrator
from .conformal_temperature_calibrator import ConformalTemperatureCalibrator


def get_calibrator(name: str, **kwargs: Any) -> CalibratorBase:
    """
    Factory function to get a calibrator instance by name.
    """
    name = name.lower().strip()
    if name == "temperature_scaling":
        return TemperatureScaling(**kwargs)
    elif name == "isotonic_regression":
        return IsotonicRegression(**kwargs)
    elif name == "conformal_mass_threshold":
        return ConformalMassThresholdCalibrator(**kwargs)
    elif name == "conformal_temperature":
        return ConformalTemperatureCalibrator(**kwargs)
    else:
        raise ValueError(f"Unknown calibrator: {name}")


__all__ = [
    "CalibratorBase",
    "IsotonicRegression",
    "TemperatureScaling",
    'ConformalCalibrator',
    "TemperatureScaling",
    "ConformalMassThresholdCalibrator",
    "ConformalTemperatureCalibrator",
    "get_calibrator",
]
