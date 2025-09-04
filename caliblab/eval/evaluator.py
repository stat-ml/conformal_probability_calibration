from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ..utils.computations import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..calibrators.base import CalibratorBase
from ..datasets.base import BaseDataset
from ..metrics.base import MetricBase
from ..models.base import ModelBase
from .constants import EvaluationReport


class ModelEvaluator:
    """Main class for evaluating model calibration and performance with caching.

    Predictions are cached at: experiments/{dataset}_{model}/predictions.npz
    Other artifacts (plots/metrics) are also saved in the same run directory.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        model: ModelBase,
        metrics: List[MetricBase],
        run_dir: Path,
        calibrators: Optional[List[CalibratorBase]] = None,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.model = model
        self.metrics = metrics
        self.run_dir = run_dir
        self.calibrators = calibrators if calibrators is not None else []
        if device is None:
            raise ValueError("A torch.device must be provided to the ModelEvaluator.")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.cal_loader = self.dataset.get_cal_loader(batch_size=128, num_workers=4)
        self.test_loader = self.dataset.get_test_loader(batch_size=128, num_workers=4)

    def _predict(
        self, loader: DataLoader, use_cache: bool, force_recompute: bool, cache_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions for a given data loader, with caching."""
        pred_path = self.run_dir / f"{cache_name}.npz"

        if use_cache and pred_path.exists() and not force_recompute:
            print(f"Using cached predictions at: {pred_path}")
            data = np.load(pred_path)
            if "probabilities" in data:
                return data["probabilities"], data["true_labels"]
            return data["logits"], data["true_labels"]

        print(f"Computing predictions for {cache_name}...")
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(loader):
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        logits = np.concatenate(all_logits)
        true_labels = np.concatenate(all_labels)

        if use_cache:
            np.savez(
                pred_path, logits=logits, true_labels=true_labels
            )
            print(f"Saved predictions to: {pred_path}")

        return logits, true_labels

    def evaluate(
        self, use_cache: bool = True, force_recompute: bool = False
    ) -> List[EvaluationReport]:
        """
        Run complete evaluation pipeline with multiple calibrators.

        Args:
            use_cache: If True, reuse predictions from disk if present.
            force_recompute: If True, ignore cache and recompute predictions.

        Returns:
            List[EvaluationReport]: List of evaluation reports, one per calibrator.
        """
        cal_logits, cal_labels = self._predict(
            self.cal_loader, use_cache, force_recompute, "cal_preds"
        )
        test_logits, test_labels = self._predict(
            self.test_loader, use_cache, force_recompute, "test_preds"
        )

        n_samples = len(test_labels)
        n_classes = test_logits.shape[1]
        results = []

        # Evaluate each calibrator
        for calibrator in [None] + self.calibrators:
            calibrator_name = calibrator.name if calibrator is not None else "none"
            print(f"\nEvaluating calibrator: {calibrator_name}")
            logits = deepcopy(test_logits)
            if calibrator is not None:
                calibrator.fit(
                    logits=test_logits, y_true=test_labels, run_dir=self.run_dir
                )
                final_probs = calibrator.predict_proba(logits=logits)
            else:
                final_probs = softmax(logits)

            calibrated_metrics = {}
            for metric in self.metrics:
                calibrated_metrics[metric.name] = metric(
                    probs=final_probs, y_true=test_labels
                )
            results.append(
                EvaluationReport(
                    calibrator_name=calibrator_name,
                    metrics=calibrated_metrics,
                    n_samples=n_samples,
                    n_classes=n_classes,
                    calibrated_probabilities=final_probs,
                    true_labels=test_labels,
                )
            )
            
        return results

