import numpy as np
from .base import MetricBase, MetricComputeInput
from ..utils.computations import make_one_hot


class BrierScore(MetricBase):
    @property
    def name(self) -> str:
        return "brier_score"

    def _compute(self, *, metric_input: MetricComputeInput):
        probs = metric_input.probs
        y_true = metric_input.y_true
        true_proba = metric_input.true_proba
        if (y_true is None and true_proba is None):
            raise ValueError(
                "Provide exactly one of y_true (labels) or true_proba (probabilities)."
            )

        if true_proba is None:
            n_samples, n_classes = probs.shape
            true_proba = make_one_hot(y_true, n_classes)

        return np.mean(np.sum((probs - true_proba) ** 2, axis=1), axis=0)


class NegativeLogLikelihood(MetricBase):
    @property
    def name(self) -> str:
        return "nll"

    def _compute(self, *, metric_input: MetricComputeInput):
        probs = metric_input.probs
        y_true = metric_input.y_true
        true_proba = metric_input.true_proba
        if (y_true is None and true_proba is None):
            raise ValueError(
                "Provide exactly one of y_true (labels) or true_proba (probabilities)."
            )

        if true_proba is not None:
            if probs.shape != true_proba.shape:
                raise ValueError("When using true_proba, its shape must match probs.")
            nll = -np.sum(true_proba * np.log(probs + 1e-12), axis=1)
            return np.mean(nll)
        else:
            correct_probs = probs[np.arange(len(y_true)), y_true]
            return np.mean(-np.log(correct_probs + 1e-12))  # add epsilon for stability


__all__ = ["BrierScore", "NegativeLogLikelihood"]
