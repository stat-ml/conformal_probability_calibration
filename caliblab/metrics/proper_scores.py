import numpy as np
from .base import MetricBase


import numpy as np

class BrierScore(MetricBase):
    def compute(self, *, probs, y_true=None, true_proba=None):
        if (y_true is None and true_proba is None) or (y_true is not None and true_proba is not None):
            raise ValueError("Provide exactly one of y_true (labels) or true_proba (probabilities).")

        if y_true is not None:
            n_samples, n_classes = probs.shape
            true_proba = np.zeros_like(probs)
            true_proba[np.arange(n_samples), y_true] = 1.0

        return np.mean((probs - true_proba) ** 2)



class NegativeLogLikelihood(MetricBase):
    def compute(self, *, probs, y_true=None, true_proba=None):
        if (y_true is None and true_proba is None) or (y_true is not None and true_proba is not None):
            raise ValueError("Provide exactly one of y_true (labels) or true_proba (probabilities).")

        if y_true is not None:
            correct_probs = probs[np.arange(len(y_true)), y_true]
            return np.mean(-np.log(correct_probs + 1e-12))  # add epsilon for stability

        else:
            if probs.shape != true_proba.shape:
                raise ValueError("When using true_proba, its shape must match probs.")
            nll = -np.sum(true_proba * np.log(probs + 1e-12), axis=1)
            return np.mean(nll)
