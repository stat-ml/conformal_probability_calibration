import numpy as np
from scipy.special import softmax

from ..conformal_prediction.conformal_predictor import ConformalPredictor
from .base import BaseCalibrator


class ConformalCalibrator(BaseCalibrator):
    def __init__(self, score_type: str, quantile_method: str):
        super().__init__()
        self.score_type = score_type
        self.quantile_method = quantile_method
        self.predictor = ConformalPredictor(
            score_type=score_type, quantile_method=quantile_method
        )

    @property
    def name(self) -> str:
        return f"conformal_{self.score_type}_{self.quantile_method}"

    def get_score(self) -> str:
        return self.score_type

    def get_quantile_method(self) -> str:
        return self.quantile_method

    def fit(
        self, *, probs=None, logits=None, y_true, **kwargs
    ) -> "ConformalCalibrator":
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided.")
        # ConformalPredictor's fit can handle logits directly
        input_probs = probs if probs is not None else softmax(logits, axis=1)
        self.predictor.fit(logits=input_probs, y_true=y_true)
        self._mark_fitted()
        return self

    def predict_proba(self, *, probs=None, logits=None) -> np.ndarray:
        self.check_fitted()
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided.")

        if probs is None:
            probs = softmax(logits, axis=1)

        quantiles = self.predictor.predict(probs)

        # Get sort order for each row based on probabilities (descending)
        order = np.argsort(-probs, axis=1)

        # Sort quantiles according to the probability order
        sorted_quantiles = np.take_along_axis(quantiles, order, axis=1)

        # The new probability mass is the difference between consecutive sorted quantiles.
        # Prepending with 0 ensures the first element's probability is its own quantile.
        calibrated_sorted = np.diff(sorted_quantiles, axis=1, prepend=0)

        # Restore the original class order
        calibrated_probs = np.take_along_axis(
            calibrated_sorted, order.argsort(axis=1), axis=1
        )

        return calibrated_probs