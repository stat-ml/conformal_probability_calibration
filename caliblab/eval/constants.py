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
