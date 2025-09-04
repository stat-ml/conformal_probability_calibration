from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .inverters.discrete_quantile_inversion import (
    DiscreteQuantileInversion,
    QuantileMethod,
)
from .score_functions import (
    ScoreTypes,
    aps_scores,
    compute_aps_scores_for_all_classes,
)


class ConformalPredictor:
    def __init__(
        self,
        score_type: str,
        quantile_method: str = QuantileMethod.NEAREST,
    ):
        if score_type not in ScoreTypes.all_types():
            raise ValueError(f"score_type must be one of {ScoreTypes.all_types()}.")

        self.score_type = score_type
        self.quantile_inversion = DiscreteQuantileInversion(
            score_type, quantile_method
        )

    def fit(
        self,
        *,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        run_dir: Optional[Path] = None,
    ) -> "ConformalPredictor":
        y_true = np.asarray(y_true)
        if probs.ndim != 2:
            raise ValueError("probs/logits must be 2D: (n, K).")
        if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
            raise ValueError("y_true must be shape (n,) and match probs/logits rows.")

        if self.score_type == ScoreTypes.APS.value:
            scores = aps_scores(probs, y_true)
        else:
            raise ValueError(f"Invalid score type: {self.score_type}")

        if run_dir:
            self._plot_scores_distribution(scores, run_dir, self.score_type)
        self.quantile_inversion.fit(np.sort(scores))
        return self

    def _plot_scores_distribution(
        self, scores: np.ndarray, run_dir: Path, score_type: str
    ):
        plt.figure()
        plt.hist(scores, bins=50, density=True)
        plt.title(f"Distribution of {score_type} scores")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plot_path = run_dir / f"scores_distribution_{score_type}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved scores distribution plot to {plot_path}")

    def predict(self, base_probs: np.ndarray) -> np.ndarray:
        test_scores = compute_aps_scores_for_all_classes(base_probs, self.score_type)
        if not np.allclose(test_scores.max(axis=1), 1.0, atol=1e-8, rtol=0):
            raise ValueError(f"Highest score should be 1.")
        quantiles = self.quantile_inversion.predict(test_scores)
        return quantiles
