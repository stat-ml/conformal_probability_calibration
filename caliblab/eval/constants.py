from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Dict[str, float]
    n_samples: int
    n_classes: int
