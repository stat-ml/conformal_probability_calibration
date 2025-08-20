import numpy as np
from .base import LabelBasedMetricBase, TrueProbMetricBase


class BrierScore(LabelBasedMetricBase):
    def compute(self, *, probs, y_true, true_proba):
        raise NotImplementedError("BrierScore.compute is not implemented.")


class NegativeLogLikelihood(LabelBasedMetricBase):
    def compute(self, *, probs, y_true, true_proba):
        raise NotImplementedError("NegativeLogLikelihood.compute is not implemented.")


class FullBrierScore(TrueProbMetricBase):
    def compute(self, *, probs, y_true, true_proba):
        return np.mean((probs - true_proba) ** 2)
