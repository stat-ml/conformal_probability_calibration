from typing import Any

from ..calibrators.base import CalibratorBase
from ..calibrators.isotonic_regression import IsotonicRegression
from ..calibrators.temperature_scaling import TemperatureScaling


def get_calibrator(name: str, **kwargs: Any) -> CalibratorBase:
    """
    Factory function to get a calibrator instance by name.
    """
    name = name.lower().strip()
    if name == "temperature_scaling":
        return TemperatureScaling(**kwargs)
    elif name == "isotonic_regression":
        return IsotonicRegression(**kwargs)
    else:
        raise ValueError(f"Unknown calibrator: {name}")
