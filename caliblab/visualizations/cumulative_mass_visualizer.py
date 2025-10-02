from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from .utils import calculate_cumulative_mass_bins
from ..utils.legend import map_legend_label


class CumulativeMassVisualizer:
    def __init__(self, n_bins: int):
        self.bins = np.linspace(0, 1, n_bins + 1)

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
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)

        for report in reports:
            if report.calibrated_probabilities is None or report.true_labels is None:
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name
            original_name = name


            # Map legend label for display (e.g., Uncalibrated -> Base)
            display_label = map_legend_label(name)

            (
                bin_mean_scores,
                bin_mean_coverages,
                bin_counts,
            ) = calculate_cumulative_mass_bins(probs, y_true, self.bins)

            non_empty_bins = bin_counts > 0
            if np.any(non_empty_bins):
                ax.plot(
                    bin_mean_scores[non_empty_bins],
                    bin_mean_coverages[non_empty_bins],
                    "o-",
                    label=display_label,
                    linewidth=2,
                )

        ax.set_xlabel("Cumulative Mass", fontsize=18, labelpad=12)
        ax.set_ylabel("Empirical Coverage", fontsize=18, labelpad=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(
            f"Cumulative Mass Calibration for {dataset_name} - {model_name}",
            fontsize=18,
            pad=5,
        )
        ax.legend(loc="lower right", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        output_path = run_dir / "cumulative_mass_calibration_curve.png"
        print(f"Saving cumulative mass curve to {output_path}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
