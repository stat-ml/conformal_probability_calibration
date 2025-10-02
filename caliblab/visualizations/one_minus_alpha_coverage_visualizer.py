from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from ..metrics import CoverageAroundOneMinusAlpha
from ..utils.computations import cumulative_mass_and_coverage


class OneMinusAlphaCoverageVisualizer:
    def __init__(self, alpha: float, replace_naming_with_ours: bool = False, n_eps_steps: int = 10):
        self.alpha = alpha
        self.eps_values = np.linspace(-self.alpha, self.alpha, n_eps_steps)
        self.eps_values = self.eps_values[~np.isclose(self.eps_values, 0)]
        self.replace_naming_with_ours = replace_naming_with_ours
        print(f"replace_naming_with_ours: {self.replace_naming_with_ours}")

    def plot(
        self,
        reports: List[EvaluationReport],
        run_dir: Path,
        dataset_name: str,
        model_name: str,
    ) -> None:
        plt.figure(figsize=(9, 9))
        ax = plt.gca()

        for report in reports:
            if report.calibrated_probabilities is None or report.true_labels is None:
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name
            original_name = name

            # Replace naming if requested
            if self.replace_naming_with_ours and name.__contains__("cnfrml_"):
                name = "ours"

            # Determine if this is our method for coloring
            is_ours = name == "ours" or original_name.__contains__("cnfrml_")
            color_kwargs = {"color": "#9467bd"} if is_ours else {}

            # Use shared utility to precompute cum_probs and true_rank
            cum_probs, _coverage_matrix, sorted_idx = cumulative_mass_and_coverage(probs, y_true)
            true_rank = np.where(sorted_idx == y_true[:, np.newaxis])[1]

            coverage_values = []
            set_sizes = []
            for eps in self.eps_values:
                metric = CoverageAroundOneMinusAlpha(alpha=self.alpha, eps=eps)
                coverage, size = metric.compute_from_cumsum(cum_probs=cum_probs, true_rank=true_rank)
                coverage_values.append(coverage)
                set_sizes.append(size)
            
            ax.plot(
                self.eps_values,
                coverage_values,
                "-",
                label=name,
                linewidth=2,
                **color_kwargs,
            )

            sc = ax.scatter(
                self.eps_values,
                coverage_values,
                s=70,
                alpha=np.log(np.array(set_sizes) + 0.1) / np.log(np.array(set_sizes).sum() + 0.1), 
                edgecolors="none",
                **color_kwargs,
            )
        
        ax.axhline(
            y=1 - self.alpha,
            color='r',
            linestyle='--',
            label=f"1 - alpha = {1 - self.alpha}",
            linewidth=2,
        )

        ax.set_xlabel("Epsilon (eps)", fontsize=18, labelpad=12)
        ax.set_ylabel("Coverage in 1-alpha and 1-alpha+eps", fontsize=18, labelpad=12)
        ax.set_xlim([self.eps_values.min(), self.eps_values.max()])
        ax.set_title(
            f"Coverage vs. Epsilon (alpha={self.alpha})\n{dataset_name} - {model_name}",
            fontsize=18,
            pad=5,
        )
        ax.legend(loc="lower right", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.6)

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        output_path = run_dir / f"one_minus_alpha_coverage_curve_{self.alpha}.png"
        print(f"Saving coverage curve to {output_path}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
