from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CalibratorBase(ABC):
    """A base class for calibrators."""

    def __init__(self):
        self.is_fitted_ = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the calibrator."""
        raise NotImplementedError

    def _mark_fitted(self) -> None:
        self.is_fitted_ = True

    def check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("This calibrator instance is not fitted yet.")

    @abstractmethod
    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "CalibratorBase":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError
    def uses_conformal_set_helper(self) -> bool:
        return False

    def get_conformal_set_sizes(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "This method is not applicable to the current calibrator."
        )

