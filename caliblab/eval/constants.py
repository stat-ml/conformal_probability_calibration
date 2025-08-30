from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EvaluationReport:
    """Dataclass for storing evaluation results."""

    calibrator_name: str
    metrics: Dict[str, float]
    n_samples: int
    n_classes: int
