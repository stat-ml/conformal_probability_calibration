from typing import Optional, Iterable, Dict
import numpy as np

from ..utils.computations import softmax
from .base import CalibratorBase
from caliblab.metrics.cumulative_mass_calibration_error import CumulativeMassCalibrationError  # adjust path if your metric lives elsewhere


class GridTemperatureScaling(CalibratorBase):
    """
    Temperature scaling via grid search over tau (no gradients).
    Minimizes CumulativeMassCalibrationError (CMCE).

    Parameters
    ----------
    tau_min : float
        Lower bound (inclusive) for tau grid if tau_grid is not provided.
    tau_max : float
        Upper bound (inclusive) for tau grid if tau_grid is not provided.
    num_taus : int
        Number of points in the linspace grid (including endpoints).
    tau_grid : Optional[Iterable[float]]
        Explicit grid of tau values to try. If provided, overrides (tau_min, tau_max, num_taus).
    n_bins : int
        Number of bins for CMCE.
    strategy : {"uniform", "quantile"}
        Binning strategy for CMCE.
    weighted : bool
        Whether to use weighted CMCE.
    """
    def __init__(
        self,
        *,
        tau_min: float = 1e-3,
        tau_max: float = 1e3,
        num_taus: int = 500,
        tau_grid: Optional[Iterable[float]] = None,
        n_bins: int = 25,
        strategy: str = "uniform",
        weighted: bool = True,
    ) -> None:
        super().__init__()
        if tau_grid is not None:
            self._grid = np.asarray(list(tau_grid), dtype=float)
            if self._grid.ndim != 1 or self._grid.size == 0:
                raise ValueError("tau_grid must be a non-empty 1D iterable of floats.")
        else:
            if tau_min <= 0 or tau_max <= 0:
                raise ValueError("tau_min and tau_max must be positive.")
            if not (tau_max > tau_min):
                raise ValueError("tau_max must be greater than tau_min.")
            if num_taus < 2:
                raise ValueError("num_taus must be >= 2.")
            self._grid = np.linspace(tau_min, tau_max, num=num_taus, dtype=float)

        self.temperature: float = 1.0
        self.n_bins = n_bins
        self.strategy = strategy
        self.weighted = weighted

        # diagnostic storage (optional)
        self._tau_to_score: Dict[float, float] = {}

    @property
    def name(self) -> str:
        return "temp_scaling_grid"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "GridTemperatureScaling":
        """
        Grid-search tau to minimize CMCE on provided logits and labels.

        Parameters
        ----------
        logits : np.ndarray, shape (n_samples, n_classes)
            Uncalibrated logits.
        y_true : np.ndarray, shape (n_samples,)
            Ground-truth labels.

        Returns
        -------
        self
        """
        if logits is None:
            raise ValueError("Logits must be provided.")
        if logits.ndim != 2:
            raise ValueError("logits must have shape (n_samples, n_classes).")
        if y_true.ndim != 1 or y_true.shape[0] != logits.shape[0]:
            raise ValueError("y_true must be 1D and match logits' first dimension.")

        metric = CumulativeMassCalibrationError(
            n_bins=self.n_bins, strategy=self.strategy, weighted=self.weighted
        )

        best_tau = None
        best_score = np.inf
        self._tau_to_score.clear()

        # Evaluate CMCE for each tau
        for tau in self._grid:
            # Scale logits and convert to probabilities
            probs = softmax(logits / tau)

            # The metric requires labels
            score = metric(probs=probs, y_true=y_true, true_proba=None)
            self._tau_to_score[float(tau)] = float(score)

            if score < best_score:
                best_score = score
                best_tau = float(tau)

        # Fallback (shouldn't happen with a valid grid)
        if best_tau is None:
            raise RuntimeError("Grid search failed to evaluate any tau values.")

        self.temperature = best_tau
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the selected temperature to logits and return calibrated probabilities.

        Parameters
        ----------
        logits : np.ndarray, shape (n_samples, n_classes)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Calibrated probabilities.
        """
        self.check_fitted()
        if logits is None:
            raise ValueError("Logits must be provided.")
        if logits.ndim != 2:
            raise ValueError("logits must have shape (n_samples, n_classes).")

        probs = softmax(logits / self.temperature)
        return probs

    # Optional helper to inspect the search landscape
    def get_search_scores(self) -> Dict[float, float]:
        """
        Returns a mapping tau -> CMCE score from the last fit() call.
        """
        return dict(self._tau_to_score)
