from typing import Optional

import numpy as np
from venn_abers import VennAbersCalibrator as ExternalVennAbersCalibrator

from ..utils.computations import softmax
from .base import CalibratorBase


class VennAbersCalibrator(CalibratorBase):
    """
    A wrapper for the VennAbersCalibrator.
    This calibrator does not have a separate training step for the calibrator itself,
    but rather uses the calibration data at prediction time.
    """

    def __init__(self, va_type: str = "one_vs_all"):
        super().__init__()
        if va_type not in ["one_vs_one", "one_vs_all"]:
            raise ValueError(f"Unknown va_type: {va_type}")
        self.va_type = va_type
        self.calibrator = ExternalVennAbersCalibrator()
        self.p_cal: Optional[np.ndarray] = None
        self.y_cal: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"venn_abers_{self.va_type}"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "VennAbersCalibrator":
        if logits is None:
            raise ValueError("Logits must be provided.")

        self.p_cal = softmax(logits)
        self.y_cal = y_true
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if logits is None:
            raise ValueError("Logits must be provided.")
        if self.p_cal is None or self.y_cal is None:
            raise RuntimeError("Calibrator is not fitted with calibration data.")

        p_test = softmax(logits)

        calibrated_probs = self.calibrator.predict_proba(
            p_cal=self.p_cal,
            y_cal=self.y_cal,
            p_test=p_test,
            va_type=self.va_type,
        )

        return calibrated_probs
