from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from .utils import calculate_confidence_bins


class ConfidenceVisualizer:
    def __init__(self, n_bins: int):
        self.n_bins = n_bins

    def plot(
        self,
        reports: List[EvaluationReport],
        run_dir: Path,
        dataset_name: str,
        model_name: str,
    ) -> None:
        """
        Plots a confidence calibration curve from a list of EvaluationReports.
        """
        results = {
            report.calibrator_name: (
                report.calibrated_probabilities,
                report.true_labels,
            )
            for report in reports
            if report.calibrated_probabilities is not None
            and report.true_labels is not None
        }
        title = f"Calibration Curve for {dataset_name} - {model_name}"
        output_path = run_dir / "confidence_calibration_curve.png"

        plt.figure(figsize=(9, 9))
        ax = plt.gca()

        # Plot the perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        for name, (probs, y_true) in results.items():
            (
                bin_confidences,
                bin_accuracies,
                bin_counts,
            ) = calculate_confidence_bins(probs, y_true, self.n_bins)

            non_empty_bins = bin_counts > 0
            if np.any(non_empty_bins):
                ax.plot(
                    bin_confidences[non_empty_bins],
                    bin_accuracies[non_empty_bins],
                    "o-",
                    label=name,
                )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
