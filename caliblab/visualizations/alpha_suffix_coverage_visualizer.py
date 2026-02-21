from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from ..metrics import AlphaSuffixCoverage
from ..utils.computations import cumulative_mass_and_coverage
from ..utils.legend import map_legend_label
from .utils import pretty_matplotlib_config


class AlphaSuffixCoverageVisualizer:
    def __init__(
        self,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        n_alpha_steps: int = 25,
    ) -> None:
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if not (0.0 <= self.alpha_min <= self.alpha_max <= 1.0):
            raise ValueError("Expected 0 <= alpha_min <= alpha_max <= 1.")
        if n_alpha_steps < 2:
            raise ValueError("n_alpha_steps must be at least 2.")
        self.alpha_values = np.linspace(self.alpha_min, self.alpha_max, n_alpha_steps)
        self.one_minus_alpha_values = 1.0 - self.alpha_values

    def plot(
        self,
        reports: List[EvaluationReport],
        run_dir: Path,
        dataset_name: str,
        model_name: str,
    ) -> None:
        pretty_matplotlib_config(
            fontsize=35,
            legend_fontsize=20,
            axes_titlesize=40,
            axes_labelsize=35,
            tick_labelsize=35,
        )
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.spines["left"].set_position(("outward", 15))
        ax.spines["bottom"].set_position(("outward", 15))

        for report in reports:
            if report.calibrated_probabilities is None or report.true_labels is None:
                continue

            probs = report.calibrated_probabilities
            y_true = report.true_labels
            name = report.calibrator_name

            coverage_values = []
            for alpha in self.alpha_values:
                # AlphaSuffixCoverage currently validates alpha in (0, 1], so map
                # the exact alpha=0 endpoint to the nearest positive float.
                alpha_for_metric = max(float(alpha), float(np.nextafter(0.0, 1.0)))
                metric = AlphaSuffixCoverage(alpha=alpha_for_metric)
                coverage = metric(probs=probs, y_true=y_true)
                coverage_values.append(coverage)

            display_label = map_legend_label(name)
            ax.plot(
                self.one_minus_alpha_values,
                coverage_values,
                "-",
                label=display_label,
                linewidth=2,
            )

        ax.plot(
            [self.one_minus_alpha_values.min(), self.one_minus_alpha_values.max()],
            [self.one_minus_alpha_values.min(), self.one_minus_alpha_values.max()],
            "r--",
            linewidth=2,
            label="Ideal",
        )
        ax.set_xlabel("1 - alpha", labelpad=12)
        ax.set_ylabel("Alpha-Suffix Coverage", labelpad=12)
        ax.set_xlim([self.one_minus_alpha_values.min(), self.one_minus_alpha_values.max()])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

        output_path = run_dir / "alpha_suffix_coverage_curve.png"
        output_path_pdf = run_dir / "alpha_suffix_coverage_curve.pdf"
        print(f"Saving alpha-suffix coverage curve to {output_path}")
        plt.tight_layout()
        plt.savefig(output_path_pdf, dpi=300, format="pdf", bbox_inches="tight")
        plt.savefig(output_path)
        plt.close()
