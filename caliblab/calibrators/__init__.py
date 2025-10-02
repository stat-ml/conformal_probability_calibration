from typing import Any

from .base import CalibratorBase
from .isotonic_regression import IsotonicRegression
from .temperature_scaling import TemperatureScaling
from .conformal_mass_threshold_calibrator import ConformalMassThresholdCalibrator
from .conformal_temperature_calibrator import ConformalTemperatureCalibrator
from .venn_abers import VennAbersCalibrator
from .platt_regression import PlattRegression
from .dirichlet_calibration import DirichletCalibration
from .adaptive_temperature_scaling import AdaptiveTemperatureScaling


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
    elif name == "venn_abers":
        return VennAbersCalibrator(**kwargs)
    elif name == "platt_regression":
        return PlattRegression(**kwargs)
    elif name == "dirichlet":
        return DirichletCalibration(**kwargs)
    elif name == "adaptive_temperature_scaling":
        return AdaptiveTemperatureScaling(**kwargs)
    else:
        raise ValueError(f"Unknown calibrator: {name}")


__all__ = [
    "CalibratorBase",
    "IsotonicRegression",
    "TemperatureScaling",
    "ConformalCalibrator",
    "TemperatureScaling",
    "ConformalMassThresholdCalibrator",
    "ConformalTemperatureCalibrator",
    "VennAbersCalibrator",
    "PlattRegression",
    "DirichletCalibration",
    "AdaptiveTemperatureScaling",
    "get_calibrator",
]
