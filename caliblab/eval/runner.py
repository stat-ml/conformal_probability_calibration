from typing import List, Optional, Tuple, Any, Dict, Set
from pathlib import Path
import torch
from tabulate import tabulate

from ..calibrators.base import CalibratorBase
from ..datasets.base import BaseDataset
from ..metrics.base import MetricBase
from ..models import ModelBase
from ..visualizations import ConfidenceVisualizer
from .constants import EvaluationReport
from .evaluator import ModelEvaluator
from ..utils.device import get_device
from .runner_utils import (
    generate_and_save_summary,
    print_and_collect_run_results,
)

EvaluationConfig = List[List[Any]]


def run_evaluations(
    *,
    configs: EvaluationConfig,
    calibrators: List[CalibratorBase],
    metrics: List[MetricBase],
    output_dir: Path,
    use_cache: bool,
    force_recompute: bool,
    visualizers: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[EvaluationReport]:
    """
    Runs a series of model evaluations based on a list of configurations.

    Args:
        configs: A list of tuples, where each tuple contains a dataset, a model,
                 and a list of metrics to compute.
        calibrators: A list of calibrators to apply to each model.
        output_dir: The root directory to save experiment results.
        use_cache: Whether to use cached predictions if available.
        force_recompute: Whether to force re-computation of predictions, ignoring cache.

    Returns:
        A list of all EvaluationReport objects generated during the runs.
    """
    all_reports: List[EvaluationReport] = []
    table_data = []
    all_metric_names: Set[str] = set()

    device = get_device(verbose=True)

    for config in configs:
        dataset, model = config
        print("-" * 80)
        print(
            f"Running evaluation for model '{model.name}' on dataset '{dataset.name}'"
        )

        # Create the run-specific directory
        run_dir = output_dir / f"{dataset.name}_{model.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator(
            dataset=dataset,
            model=model,
            metrics=metrics,
            calibrators=calibrators,
            run_dir=run_dir,
            device=device,
        )
        run_reports = evaluator.evaluate(
            use_cache=use_cache, force_recompute=force_recompute
        )

        print_and_collect_run_results(
            run_reports, dataset, model, table_data, all_metric_names
        )

        # --- Plotting ---
        if visualizers:
            for visualizer in visualizers:
                visualizer.plot(
                    reports=run_reports,
                    run_dir=run_dir,
                    dataset_name=dataset.name,
                    model_name=model.name,
                )

        all_reports.extend(run_reports)

    print("-" * 80)
    print("All evaluations complete.")

    generate_and_save_summary(table_data, all_metric_names, Path(output_dir))

    return all_reports
