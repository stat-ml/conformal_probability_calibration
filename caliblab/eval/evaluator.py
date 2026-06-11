from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..calibrators.base import CalibratorBase
from ..datasets.base import BaseDataset
from ..metrics.base import MetricBase
from ..models.base import ModelBase
from ..utils.computations import softmax
from .constants import EvaluationReport
from ..datasets.utils import DataSplit


class ModelEvaluator:
    """Evaluate model calibration and performance for one data split."""

    def __init__(
        self,
        *,
        metrics: List[MetricBase],
        calibrators: List[CalibratorBase | None],
        run_dir: Path,
        device: torch.device,
    ):
        self.metrics = metrics
        self.calibrators = calibrators
        self.run_dir = run_dir
        self.device = device

    def run_calibration_and_metrics(
        self,
        datasplit: DataSplit
    ) -> List[EvaluationReport]:
        reports = []

        all_calibrators = [None] + self.calibrators

        for calibrator in all_calibrators:
            if calibrator is None:
                calibrated_test_outputs = datasplit.test_outputs
                train_time = 0.0
                predict_time = 0.0
            else:
                start_time = time.perf_counter()
                calibrator.fit(logits=datasplit.cal_outputs, y_true=datasplit.cal_labels)
                train_time = time.perf_counter() - start_time

                start_time = time.perf_counter()
                calibrated_test_outputs = calibrator.predict_proba(
                    logits=datasplit.test_outputs
                )
                predict_time = time.perf_counter() - start_time

            report = self._evaluate_calibrator(
                calibrator,
                calibrated_test_outputs,
                datasplit.test_labels,
                datasplit.test_outputs,
                datasplit.test_probs,
                train_time,
                predict_time,
            )
            reports.append(report)

        return reports

    def _evaluate_calibrator(
        self,
        calibrator: CalibratorBase | None,
        outputs: np.ndarray,
        labels: np.ndarray,
        uncalibrated_test_logits: np.ndarray,
        true_probs: Optional[np.ndarray] = None,
        train_time: float = 0.0,
        predict_time: float = 0.0,
    ) -> EvaluationReport:
        calibrator_name = calibrator.name if calibrator else "uncalibrated"
        print("Evaluating calibrator: ", calibrator_name)

        metric_results = {}

        if calibrator is None:
            # For the uncalibrated case, the outputs are logits
            probs = softmax(outputs)
        else:
            # For calibrated cases, the outputs are already probabilities
            probs = outputs
        uncalibrated_probs = softmax(uncalibrated_test_logits)

        for metric in self.metrics:
            metric_results[metric.name] = metric(
                probs=probs,
                y_true=labels,
                true_proba=true_probs,
                uncalibrated_probs=uncalibrated_probs,
            )

        conformal_test_sizes = None
        conformal_test_coverage = None
        if calibrator and calibrator.uses_conformal_set_helper():
            conformal_test_sizes = calibrator.get_conformal_set_sizes(
                logits=uncalibrated_test_logits
            )
            conformal_test_coverage = calibrator.get_conformal_set_coverage(
                logits=uncalibrated_test_logits,
                y_true=labels
            )
            print(f"{calibrator_name} conformal test coverage: {conformal_test_coverage}")

        return EvaluationReport(
            calibrator_name=calibrator_name,
            metrics=metric_results,
            n_samples=len(labels),
            n_classes=probs.shape[1],
            calibrated_probabilities=probs,
            true_labels=labels,
            train_time=train_time,
            predict_time=predict_time,
            conformal_set_sizes=conformal_test_sizes,
            conformal_test_coverage=conformal_test_coverage,
        )
