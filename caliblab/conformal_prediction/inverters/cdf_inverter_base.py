from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

class InversionType(str, Enum):
    DISCRETE = "discrete"
    NORMALIZING_FLOW = "normalizing_flow"


class CDF_inverter_base(ABC):
    @abstractmethod
    def fit(self, scores: np.ndarray) -> "CDF_inverter_base":
        """
        Fit the inverter on calibration scores.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Produce a prediction set/mask.
        """
        raise NotImplementedError
