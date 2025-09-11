from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..calibrators.base import CalibratorBase
from ..datasets.base import BaseDataset
from ..metrics.base import MetricBase
from ..models.base import ModelBase
from ..utils.computations import softmax
from .constants import EvaluationReport


class ModelEvaluator:
    """Main class for evaluating model calibration and performance with caching.

    Predictions are cached at: experiments/{dataset}_{model}/predictions.npz
    Other artifacts (plots/metrics) are also saved in the same run directory.
    """

    def __init__(
        self,
        *,
        model: ModelBase,
        metrics: List[MetricBase],
        calibrators: List[CalibratorBase],
        run_dir: Path,
        device: torch.device,
    ):
        self.model = model
        self.metrics = metrics
        self.calibrators = calibrators
        self.run_dir = run_dir
        self.device = device

    def get_predictions(
        self,
        loader: DataLoader,
        cache_path: Path,
        use_cache: bool = True,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if use_cache and not force_recompute and cache_path.exists():
            print(f"Loading cached predictions from {cache_path}")
            cached_data = np.load(cache_path)
            return cached_data["outputs"], cached_data["labels"]

        print(f"Computing predictions and saving to {cache_path}")
        outputs, labels = self.model.predict(loader, self.device)
        np.savez(cache_path, outputs=outputs, labels=labels)
        return outputs, labels

    def run_calibration_and_metrics(
        self,
        cal_outputs: np.ndarray,
        cal_labels: np.ndarray,
        test_outputs: np.ndarray,
        test_labels: np.ndarray,
    ) -> List[EvaluationReport]:
        reports = []

        uncalibrated_report = self._evaluate_calibrator(
            None, test_outputs, test_labels
        )
        reports.append(uncalibrated_report)

        for calibrator in self.calibrators:
            start_time = time.time()
            calibrator.fit(logits=cal_outputs, y_true=cal_labels)
            train_time = time.time() - start_time

            start_time = time.time()
            calibrated_test_outputs = calibrator.predict_proba(logits=test_outputs)
            predict_time = time.time() - start_time

            report = self._evaluate_calibrator(
                calibrator,
                calibrated_test_outputs,
                test_labels,
                train_time,
                predict_time,
            )
            reports.append(report)

        return reports

    def _evaluate_calibrator(
        self,
        calibrator: CalibratorBase,
        outputs: np.ndarray,
        labels: np.ndarray,
        train_time: float = 0.0,
        predict_time: float = 0.0,
    ) -> EvaluationReport:
        calibrator_name = calibrator.name if calibrator else "uncalibrated"
        metric_results = {}

        if calibrator is None:
            # For the uncalibrated case, the outputs are logits
            probs = softmax(outputs)
        else:
            # For calibrated cases, the outputs are already probabilities
            probs = outputs

        for metric in self.metrics:
            metric_results[metric.name] = metric(probs=probs, y_true=labels)

        return EvaluationReport(
            calibrator_name=calibrator_name,
            metrics=metric_results,
            n_samples=len(labels),
            n_classes=probs.shape[1],
            calibrated_probabilities=probs,
            true_labels=labels,
            train_time=train_time,
            predict_time=predict_time,
        )
