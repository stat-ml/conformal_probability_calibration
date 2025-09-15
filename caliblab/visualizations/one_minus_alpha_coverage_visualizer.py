from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport
from ..metrics import CoverageAroundOneMinusAlpha


class OneMinusAlphaCoverageVisualizer:
    def __init__(self, alpha: float, n_eps_steps: int = 25):
        self.alpha = alpha
        self.eps_values = np.linspace(-self.alpha, self.alpha, n_eps_steps)

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

            sorted_idx = np.argsort(probs, axis=1)[:, ::-1]

            coverage_values = []
            for eps in self.eps_values:
                metric = CoverageAroundOneMinusAlpha(alpha=self.alpha, eps=eps)
                coverage = metric.compute_from_sorted(
                    probs=probs, y_true=y_true, sorted_idx=sorted_idx
                )
                coverage_values.append(coverage)

            ax.plot(
                self.eps_values,
                coverage_values,
                "o-",
                label=name,
                markersize=3,
            )
        
        ax.axhline(y=1 - self.alpha, color='r', linestyle='--', label=f"1 - alpha = {1 - self.alpha}")

        ax.set_xlabel("Epsilon (eps)")
        ax.set_ylabel("Coverage in 1-alpha and 1-alpha+eps")
        ax.set_xlim([self.eps_values.min(), self.eps_values.max()])
        ax.set_title(
            f"Coverage vs. Epsilon (alpha={self.alpha}) for {dataset_name} - {model_name}"
        )
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

        output_path = run_dir / "one_minus_alpha_coverage_curve.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
