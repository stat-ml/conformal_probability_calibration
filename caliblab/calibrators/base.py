from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CalibratorBase(ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
    ) -> "CalibratorBase":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    def check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
