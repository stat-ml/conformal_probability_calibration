from .base import LabelBasedMetricBase, PairwiseProbMetricBase, MetricComputeInput
import numpy as np
from typing import Optional

from ..utils.bins import get_bin_boundaries


def _compute_bin_calibration_error(
    probs: np.ndarray, correct: np.ndarray, bin_boundaries: np.ndarray
) -> tuple:
    """Compute calibration error for each bin."""
    n_samples = len(probs)
    bin_errors = []

    for i in range(len(bin_boundaries) - 1):
        # Find samples in current bin
        if i == len(bin_boundaries) - 2:  # Last bin includes upper boundary
            in_bin = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        else:
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(correct[in_bin])
            bin_weight = np.sum(in_bin) / n_samples
            bin_error = np.abs(avg_confidence - avg_accuracy)
            bin_errors.append((bin_weight, bin_error))

    return bin_errors


class ExpectedCalibrationError(LabelBasedMetricBase):
    def __init__(self, n_bins: int = 15, strategy: str = "uniform") -> None:
        super().__init__()
        self.n_bins = n_bins
        self.strategy = strategy

    @property
    def name(self) -> str:
        return "ece"

    def _compute(
        self,
        *,
        metric_input: MetricComputeInput,
    ):
        probs = metric_input.probs
        y_true = metric_input.y_true
        max_probs = np.max(probs, axis=1)
        correct = (np.argmax(probs, axis=1) == y_true).astype(float)
        bin_boundaries = get_bin_boundaries(max_probs, self.n_bins, self.strategy)

        bin_errors = _compute_bin_calibration_error(max_probs, correct, bin_boundaries)
        return sum(weight * error for weight, error in bin_errors)


class MaximumCalibrationError(LabelBasedMetricBase):
    def __init__(self, n_bins: int = 15, strategy: str = "uniform") -> None:
        super().__init__()
        self.n_bins = n_bins
        self.strategy = strategy

    @property
    def name(self) -> str:
        return "mce"

    def _compute(
        self,
        *,
        metric_input: MetricComputeInput,
    ):
        probs = metric_input.probs
        y_true = metric_input.y_true
        max_probs = np.max(probs, axis=1)
        correct = (np.argmax(probs, axis=1) == y_true).astype(float)
        bin_boundaries = get_bin_boundaries(max_probs, self.n_bins, self.strategy)

        bin_errors = _compute_bin_calibration_error(max_probs, correct, bin_boundaries)
        if not bin_errors:
            raise ValueError(
                "No bins contain any samples; cannot compute MaximumCalibrationError."
            )
        return max(error for _, error in bin_errors)


class ClasswiseExpectedCalibrationError(LabelBasedMetricBase):
    def __init__(self, n_bins: int = 15, strategy: str = "uniform") -> None:
        super().__init__()
        self.n_bins = n_bins
        self.strategy = strategy

    @property
    def name(self) -> str:
        return "cw-ece"

    def _compute(
        self,
        *,
        metric_input: MetricComputeInput,
    ):
        probs = metric_input.probs
        y_true = metric_input.y_true
        bin_boundaries = get_bin_boundaries(probs, self.n_bins, self.strategy)
        class_eces = []
        n_classes = probs.shape[1]

        for class_idx in range(n_classes):
            class_probs = probs[:, class_idx]
            class_correct = (y_true == class_idx).astype(float)

            bin_errors = _compute_bin_calibration_error(
                class_probs, class_correct, bin_boundaries
            )
            class_ece = sum(weight * error for weight, error in bin_errors)
            class_eces.append(class_ece)

        return np.mean(class_eces)


def _validate_pairwise_prob_inputs(
    calibrated_probs: np.ndarray, uncalibrated_probs: Optional[np.ndarray]
) -> np.ndarray:
    if uncalibrated_probs is None:
        raise ValueError(
            "uncalibrated_probs must be provided to compute preserving ratios."
        )
    if calibrated_probs.shape != uncalibrated_probs.shape:
        raise ValueError(
            "uncalibrated_probs and calibrated probs must have the same shape."
        )
    return uncalibrated_probs


class AccuracyPreservingRatio(PairwiseProbMetricBase):
    """Ratio of samples where top-1 class is unchanged after calibration."""

    @property
    def name(self) -> str:
        return "accuracy_preserving_ratio"

    def _compute(
        self,
        *,
        metric_input: MetricComputeInput,
    ) -> float:
        probs = metric_input.probs
        uncalibrated_probs = metric_input.uncalibrated_probs
        uncalibrated_probs = _validate_pairwise_prob_inputs(probs, uncalibrated_probs)
        calibrated_top1 = np.argmax(probs, axis=1)
        uncalibrated_top1 = np.argmax(uncalibrated_probs, axis=1)
        return float(np.mean(calibrated_top1 == uncalibrated_top1))


class OrderPreservingRatio(PairwiseProbMetricBase):
    """Ratio of samples where class probability ranking is unchanged."""

    @property
    def name(self) -> str:
        return "order_preserving_ratio"

    def _compute(
        self,
        *,
        metric_input: MetricComputeInput,
    ) -> float:
        probs = metric_input.probs
        uncalibrated_probs = metric_input.uncalibrated_probs
        uncalibrated_probs = _validate_pairwise_prob_inputs(probs, uncalibrated_probs)
        calibrated_order = np.argsort(-probs, axis=1, kind="mergesort")
        uncalibrated_order = np.argsort(-uncalibrated_probs, axis=1, kind="mergesort")
        same_order = np.all(calibrated_order == uncalibrated_order, axis=1)
        return float(np.mean(same_order))


__all__ = [
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    "ClasswiseExpectedCalibrationError",
    "AccuracyPreservingRatio",
    "OrderPreservingRatio",
]
