from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .constants import EvaluationReport
from ..metrics.base import MetricBase, LabelBasedMetricBase, TrueProbMetricBase
from ..utils.validation import check_probs, check_labels


def evaluate(
    *,
    probs_cal: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    true_proba: Optional[np.ndarray] = None,
    metrics: List[MetricBase],
) -> EvaluationReport:
    check_probs(probs_cal, name="probs_cal")
    n, k = probs_cal.shape
    if y_true is not None:
        check_labels(y_true, n_classes=k)
    if true_proba is not None:
        check_probs(true_proba, name="true_proba")

    out: Dict[str, float] = {}
    for m in metrics:
        if isinstance(m, LabelBasedMetricBase) and y_true is None:
            raise ValueError(f"Metric {m.name} requires y_true but none was provided.")
        if isinstance(m, TrueProbMetricBase) and true_proba is None:
            raise ValueError(
                f"Metric {m.name} requires true_proba but none was provided."
            )
        val = m(
            probs=probs_cal,
            y_true=y_true,
            true_proba=true_proba,
        )
        out[m.name] = val

    return EvaluationReport(metrics=out, n_samples=n, n_classes=k)
