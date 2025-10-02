from typing import Optional

import os
import sys
import numpy as np

from ..utils.computations import softmax
from .base import CalibratorBase
from .dirichlet import DirichletCalibrator


def _ensure_dirichlet_on_path() -> None:
    base_dir = os.path.dirname(__file__)
    dirichlet_parent = base_dir
    if dirichlet_parent not in sys.path:
        sys.path.insert(0, dirichlet_parent)


class DirichletCalibration(CalibratorBase):
    def __init__(self, matrix_type: str = "full", l2: float | list[float] = 0.0, comp_l2: bool | list[bool] = False, initializer: str = "identity"):
        super().__init__()
        _ensure_dirichlet_on_path()
        self.calibrator = DirichletCalibrator(matrix_type=matrix_type, l2=l2, comp_l2=comp_l2, initializer=initializer)
        self.matrix_type = matrix_type
        self.l2 = l2
        self.comp_l2 = comp_l2
        self.initializer = initializer
        self._model = None
        self._trained_on: Optional[str] = None

    @property
    def name(self) -> str:
        return "dirichlet"

    def fit(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "DirichletCalibration":
        if probs is None and logits is None:
            raise ValueError("Either logits or probs must be provided to DirichletCalibration.")

        if logits is not None:
            p = softmax(logits)
            self._trained_on = "logits"
        else:
            p = np.asarray(probs, dtype=np.float64)
            self._trained_on = "probs"

        self._model = self.calibrator.fit(p, y_true)
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if self._model is None:
            raise RuntimeError("DirichletCalibration model is not initialized.")

        if logits is None and probs is None:
            raise ValueError("Either logits or probs must be provided to DirichletCalibration.")

        if self._trained_on == "logits":
            if logits is None:
                raise ValueError("DirichletCalibration was trained on logits; provide logits at prediction time.")
            p = softmax(logits)
        else:
            if probs is None:
                if logits is None:
                    raise ValueError("Either logits or probs must be provided to DirichletCalibration.")
                p = softmax(logits)
            else:
                p = probs

        return np.asarray(self._model.predict_proba(p))


__all__ = ["DirichletCalibration"]


