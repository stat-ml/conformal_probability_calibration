from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from ..metrics import CoverageAroundOneMinusAlpha
from ..utils.computations import cumulative_mass_and_coverage
from .utils import pretty_matplotlib_config
from ..utils.legend import map_legend_label



class OneMinusAlphaCoverageVisualizer:
    def __init__(self, alpha: float, n_eps_steps: int = 10):
        self.alpha = alpha
        self.eps_values = np.linspace(-self.alpha, self.alpha, n_eps_steps)
        self.eps_values = self.eps_values[~np.isclose(self.eps_values, 0)]

    def plot(
        self,
        reports: List[EvaluationReport],
        run_dir: Path,
        dataset_name: str,
        model_name: str,
    ) -> None:
        # Ensure consistent matplotlib styling (same as Cumulative Mass visualizer)
        pretty_matplotlib_config(
            fontsize=35,
            legend_fontsize=20,
            axes_titlesize=40,
            axes_labelsize=35,
            tick_labelsize=35,
        )
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.spines['left'].set_position(('outward', 15))
        ax.spines['bottom'].set_position(('outward', 15))

        for report in reports:
            if report.calibrated_probabilities is None or report.true_labels is None:
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name

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
            
            display_label = map_legend_label(name)

            ax.plot(
                self.eps_values,
                coverage_values,
                "-",
                label=display_label,
                linewidth=2,
            )

            sc = ax.scatter(
                self.eps_values,
                coverage_values,
                s=70,
                alpha=np.log(np.array(set_sizes) + 0.1) / np.log(np.array(set_sizes).sum() + 0.1), 
                edgecolors="none",
            )
        
        ax.axhline(
            y=1 - self.alpha,
            color='r',
            linestyle='--',
            linewidth=2,
        )

        ax.set_xlabel("Offset ($\epsilon$)", labelpad=12)
        ax.set_ylabel("Coverage", labelpad=12)
        ax.set_xlim([self.eps_values.min(), self.eps_values.max()])
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')

        output_path = run_dir / f"one_minus_alpha_coverage_curve_{self.alpha}.png"
        output_path_pdf = run_dir / f"one_minus_alpha_coverage_curve_{self.alpha}.pdf"
        print(f"Saving coverage curve to {output_path}")
        plt.tight_layout()
        plt.savefig(output_path_pdf, dpi=300, format="pdf", bbox_inches="tight")
        plt.savefig(output_path)
        plt.close()
