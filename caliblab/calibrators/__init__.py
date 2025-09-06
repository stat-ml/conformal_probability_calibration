from typing import Any

from .base import CalibratorBase
from .isotonic_regression import IsotonicRegression
from .temperature_scaling import TemperatureScaling
from .conformal_calibrator import ConformalCalibrator
from .conformal_calibrators import ConformalMassCalibrator

def get_calibrator(name: str, **kwargs: Any) -> CalibratorBase:
    """
    Factory function to get a calibrator instance by name.
    """
    name = name.lower().strip()
    if name == "temperature_scaling":
        return TemperatureScaling(**kwargs)
    elif name == "isotonic_regression":
        return IsotonicRegression(**kwargs)
    elif name == "conformal_calibrator":
        return ConformalCalibrator(**kwargs)
    elif name == "conformal_mass":
        return ConformalMassCalibrator(**kwargs)
    else:
        raise ValueError(f"Unknown calibrator: {name}")


__all__ = [
    "CalibratorBase",
    "IsotonicRegression",
    "TemperatureScaling",
    'ConformalCalibrator',
    "TemperatureScaling",
    "ConformalMassCalibrator",
    "get_calibrator",
]
