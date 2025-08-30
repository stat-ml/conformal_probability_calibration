from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport


class CumulativeMassVisualizer:
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
        Plots a cumulative mass calibration curve from a list of EvaluationReports
        using a binning strategy.
        """
        plt.figure(figsize=(9, 9))
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        for report in reports:
            if report.calibrated_probabilities is None:
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name
            n_samples, n_classes = probs.shape

            # Sort probabilities and get corresponding indices
            sorted_indices = np.argsort(-probs, axis=1)
            sorted_probs = -np.sort(-probs, axis=1)

            # Get cumulative probability masses for each potential set size
            cum_probs = np.cumsum(sorted_probs, axis=1)
            all_cum_scores = cum_probs.flatten()

            # Find the rank of the true class for each sample
            true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

            # Create a matrix indicating if the true class is covered for each set size
            ranks = np.arange(n_classes)
            coverage_matrix = true_class_ranks[:, np.newaxis] <= ranks[np.newaxis, :]
            all_coverages = coverage_matrix.flatten()

            # Binning logic
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            bin_mean_scores = np.zeros(self.n_bins)
            bin_mean_coverages = np.zeros(self.n_bins)
            bin_counts = np.zeros(self.n_bins)

            for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
                in_bin_mask = (all_cum_scores > lower) & (all_cum_scores <= upper)
                bin_counts[i] = np.sum(in_bin_mask)

                if bin_counts[i] > 0:
                    bin_mean_scores[i] = np.mean(all_cum_scores[in_bin_mask])
                    bin_mean_coverages[i] = np.mean(all_coverages[in_bin_mask])

            non_empty_bins = bin_counts > 0
            if np.any(non_empty_bins):
                ax.plot(
                    bin_mean_scores[non_empty_bins],
                    bin_mean_coverages[non_empty_bins],
                    "o-",
                    label=name,
                )

        ax.set_xlabel("Cumulative Mass")
        ax.set_ylabel("Empirical Coverage")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f"Cumulative Mass Calibration for {dataset_name} - {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")

        output_path = run_dir / "cumulative_mass_calibration_curve.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
