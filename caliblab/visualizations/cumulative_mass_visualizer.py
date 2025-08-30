from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from .utils import calculate_cumulative_mass_bins


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
            if (
                report.calibrated_probabilities is None
                or report.true_labels is None
            ):
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name

            (
                bin_mean_scores,
                bin_mean_coverages,
                bin_counts,
            ) = calculate_cumulative_mass_bins(probs, y_true, self.n_bins)

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
