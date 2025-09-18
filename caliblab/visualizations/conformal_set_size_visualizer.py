from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..eval.constants import EvaluationReport


class ConformalSetSizeVisualizer:
    def plot(
        self,
        reports: List[EvaluationReport],
        run_dir: Path,
        dataset_name: str,
        model_name: str,
    ) -> None:
        """
        Plots a distribution of conformal set sizes from a list of EvaluationReports.
        """
        results = {
            report.calibrator_name: report.conformal_set_sizes
            for report in reports
            if report.conformal_set_sizes is not None
        }
        if not results:
            return

        title = f"Conformal Set Size Distribution for {dataset_name} - {model_name}"
        output_path = run_dir / "conformal_set_size_distribution.png"

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for name, set_sizes in results.items():
            ax.hist(
                set_sizes.clip(0, 50),
                bins=np.arange(0, 51),
                alpha=0.6,
                label=f"{name} (median: {np.median(set_sizes):.2f})",
            )

        for name, set_sizes in results.items():
            quantiles = np.percentile(set_sizes, [0, 25, 50, 75, 100])
            print(f"{name} conformal set size quantiles (min, 25%, 50%, 75%, max): {quantiles}")

        ax.set_xlabel("Conformal Set Size")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
