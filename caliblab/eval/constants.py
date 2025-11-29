from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class EvaluationReport:
    """A dataclass to hold the results of a single evaluation run."""

    calibrator_name: str
    metrics: Dict[str, float]
    n_samples: int
    n_classes: int
    calibrated_probabilities: Optional[np.ndarray] = field(default=None, repr=False)
    true_labels: Optional[np.ndarray] = field(default=None, repr=False)
    conformal_set_sizes: Optional[np.ndarray] = field(default=None, repr=False)
    # Pre-computed summary statistics of the conformal set sizes.
    # These are persisted in run_reports.pkl so they are available even after
    # the potentially large raw arrays are dropped.
    avg_set_size: Optional[float] = None
    set_size_q25: Optional[float] = None
    set_size_q50: Optional[float] = None
    set_size_q75: Optional[float] = None
    train_time: float = 0.0
    predict_time: float = 0.0
