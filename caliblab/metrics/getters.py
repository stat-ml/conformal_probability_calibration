from typing import Any

from .base import MetricBase
from .calibration_errors import CWECE, ECE, MCE
from .classification import Accuracy
from .proper_scores import NLL, BrierScore


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
    elif name == "ce":
        return CE(**kwargs)
    elif name == "nll":
        return NLL()
    elif name == "brier_score":
        return BrierScore()
    else:
        raise ValueError(f"Unknown metric: {name}")
