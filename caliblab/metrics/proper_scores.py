import numpy as np
from .base import MetricBase


import numpy as np

def make_one_hot(y_true, n_classes):
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true] = 1.0
    return one_hot

class BrierScore(MetricBase):
    def _compute(self, *, probs, y_true=None, true_proba=None):
        if (y_true is None and true_proba is None) or (y_true is not None and true_proba is not None):
            raise ValueError("Provide exactly one of y_true (labels) or true_proba (probabilities).")

        if y_true is not None:
            n_samples, n_classes = probs.shape
            true_proba = make_one_hot(y_true, n_classes)

        return np.mean(np.sum((probs - true_proba) ** 2, axis=1), axis=0)



class NegativeLogLikelihood(MetricBase):
    def _compute(self, *, probs, y_true=None, true_proba=None):
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
