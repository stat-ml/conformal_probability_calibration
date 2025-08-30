from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport


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
        }
        title = f"Calibration Curve for {dataset_name} - {model_name}"
        output_path = run_dir / "confidence_calibration_curve.png"

        plt.figure(figsize=(15, 15))
        ax = plt.gca()

        # Plot the perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        for name, (probs, y_true) in results.items():
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = (predictions == y_true).astype(float)

            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            bin_accuracies = np.zeros(self.n_bins)
            bin_confidences = np.zeros(self.n_bins)
            bin_counts = np.zeros(self.n_bins)

            for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
                in_bin = (confidences > lower) & (confidences <= upper)
                bin_counts[i] = np.sum(in_bin)

                if bin_counts[i] > 0:
                    bin_accuracies[i] = np.mean(accuracies[in_bin])
                    bin_confidences[i] = np.mean(confidences[in_bin])

            # Filter out bins with no samples
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
