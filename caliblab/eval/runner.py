from typing import List, Optional, Tuple, Any, Dict
from pathlib import Path
from tabulate import tabulate

from ..calibrators.base import CalibratorBase
from ..datasets.base import BaseDataset
from ..metrics.base import MetricBase
from ..models import ModelBase
from ..visualizations import ConfidenceVisualizer
from .constants import EvaluationReport
from .evaluator import ModelEvaluator

EvaluationConfig = Tuple[BaseDataset, ModelBase]


def run_evaluations(
    *,
    configs: List[EvaluationConfig],
    calibrators: List[CalibratorBase],
    metrics: List[MetricBase],
    output_dir: Path,
    use_cache: bool,
    force_recompute: bool,
    visualizer: Optional[ConfidenceVisualizer] = None,
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
    all_metric_names = set()

    for dataset, model in configs:
        print("-" * 80)
        print(f"Running evaluation for model '{model.name}' on dataset '{dataset.name}'")

        # Create the run-specific directory
        run_dir = output_dir / f"{dataset.name}_{model.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator(
            dataset=dataset,
            model=model,
            metrics=metrics,
            calibrators=calibrators,
            run_dir=run_dir,
        )
        run_reports = evaluator.evaluate(
            use_cache=use_cache, force_recompute=force_recompute
        )

        # --- Plotting ---
        if visualizer is not None:
            visualizer.plot(
                reports=run_reports,
                run_dir=run_dir,
                dataset_name=dataset.name,
                model_name=model.name,
            )

        # --- Per-run summary ---
        print("\n" + tabulate(
            [
                {
                    "Dataset": dataset.name,
                    "Model": model.name,
                    "Calibrator": report.calibrator_name,
                }
                for report in run_reports
            ],
            headers="keys",
            tablefmt="grid",
        ))

        # Print a summary of the results and collect data for the final table
        for report in run_reports:
            print(f"  Calibrator: {report.calibrator_name}")
            row = {
                "Dataset": dataset.name,
                "Model": model.name,
                "Calibrator": report.calibrator_name,
            }
            for metric_name, value in report.metrics.items():
                print(f"    {metric_name}: {value:.4f}")
                row[metric_name] = value
                all_metric_names.add(metric_name)
            table_data.append(row)

        all_reports.extend(run_reports)

    print("-" * 80)
    print("All evaluations complete.")

    # Print the summary table
    if table_data:
        headers = ["Dataset", "Model", "Calibrator"] + sorted(list(all_metric_names))

        # Format numbers to 4 decimal places for printing
        formatted_rows = []
        for row_dict in table_data:
            formatted_row = []
            for header in headers:
                value = row_dict.get(header)
                if isinstance(value, float):
                    formatted_row.append(f"{value:.4f}")
                else:
                    formatted_row.append(value)
            formatted_rows.append(formatted_row)

        print("\n" + "=" * 80)
        print("Summary of Results")
        print("=" * 80)
        table_string = tabulate(formatted_rows, headers=headers, tablefmt="grid")
        print(table_string)

        # Save the table to a file
        output_path = Path(output_dir) / "summary_results.txt"
        with open(output_path, "w") as f:
            f.write("Summary of Results\n")
            f.write("=" * 80 + "\n")
            f.write(table_string)
        print(f"\nResults table saved to {output_path}")

    return all_reports
