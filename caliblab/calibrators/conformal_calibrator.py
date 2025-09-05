import numpy as np
from ..utils.computations import softmax

from ..conformal_prediction.conformal_predictor import ConformalPredictor
from .base import CalibratorBase

from ..conformal_prediction.inverters.cdf_inverter_base import InversionType


class ConformalCalibrator(CalibratorBase):
    def __init__(self, score_type: str, quantile_method: str, inversion_type: InversionType):
        super().__init__()
        self.score_type = score_type
        self.quantile_method = quantile_method
        self.inversion_type = inversion_type
        self.predictor = ConformalPredictor(
            score_type=score_type, quantile_method=quantile_method, inversion_type=inversion_type
        )

    @property
    def name(self) -> str:
        return f"conformal_{self.score_type}_{self.quantile_method}_{self.inversion_type}"

    def get_score(self) -> str:
        return self.score_type

    def get_quantile_method(self) -> str:
        return self.quantile_method

    def fit(
        self, *, probs=None, logits=None, y_true, **kwargs
    ) -> "ConformalCalibrator":
        if probs is None:
            if logits is None:
                raise ValueError(
                    "Either logits or probs must be provided to IsotonicRegression."
                )
            probs = softmax(logits)
        # ConformalPredictor's fit can handle logits directly

        self.predictor.fit(
            logits=logits, y_true=y_true, run_dir=kwargs.get("run_dir")
        )
        self._mark_fitted()
        return self

    def predict_proba(self, *, probs=None, logits=None) -> np.ndarray:
        self.check_fitted()
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided.")

        if probs is None:
            probs = softmax(logits, axis=1)

        quantiles = self.predictor.predict(base_probs=probs, logits=logits)

        # FATAL
        # Enforce that the largest element in each row is exactly 1.0
        max_indices = np.argmax(quantiles, axis=1)
        quantiles[np.arange(quantiles.shape[0]), max_indices] = 1.0

        if not np.allclose(quantiles.max(axis=1).min(), 1.0, atol=1e-4):
            raise ValueError(f"Highest quantile should be 1.")

        # Get sort order for each row based on probabilities (descending)
        order = np.argsort(-probs, axis=1)

        # Sort quantiles according to the probability order
        sorted_quantiles = np.take_along_axis(quantiles, order, axis=1)

        # The new probability mass is the difference between consecutive sorted quantiles.
        # Prepending with 0 ensures the first element's probability is its own quantile.
        calibrated_sorted = np.diff(sorted_quantiles, axis=1, prepend=0)

        if not np.allclose(calibrated_sorted[:, 0], quantiles.min(axis=1), atol=1e-4):
            raise ValueError(f"Lowest calibrated_sorted should be equal to lowest quantile.")

        # Restore the original class order
        calibrated_probs = np.take_along_axis(
            calibrated_sorted, order.argsort(axis=1), axis=1
        )

        return calibrated_probs
