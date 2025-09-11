import pandas as pd
from typing import List, Optional, Tuple, Any, Dict, Set
from pathlib import Path
import torch
from tabulate import tabulate

from ..calibrators.base import CalibratorBase
from ..datasets import dataset_getter
from ..metrics.base import MetricBase
from ..models import get_model
from ..visualizations import ConfidenceVisualizer
from .constants import EvaluationReport
from .evaluator import ModelEvaluator
from ..utils.device import get_device
from .runner_utils import (
    generate_and_save_summary,
    print_and_collect_run_results,
)

EvaluationConfig = List[Dict[str, Any]]


def run_evaluations(
    *,
    configs: EvaluationConfig,
    calibrators: List[CalibratorBase],
    metrics: List[MetricBase],
    output_dir: Path,
    data_root: Path,
    use_cache: bool,
    force_recompute: bool,
    visualizers: Optional[List[Any]] = None,
    num_splits: int = 1,
    **kwargs: Any,
) -> List[EvaluationReport]:
    all_reports: List[EvaluationReport] = []
    table_data = []
    all_metric_names: Set[str] = set()

    device = get_device(verbose=True)
    model_cache_dir = kwargs.get("model_cache_dir")

    for split_seed in range(num_splits):
        print("*" * 80)
        print(f"Running split with seed {split_seed}")
        for config in configs:
            dataset_config = config["dataset_config"]
            dataset_name = dataset_config["name"]
            dataset_params = dataset_config.get("params", {})
            dataset_params["seed"] = split_seed
            dataset = dataset_getter(
                dataset_name,
                data_dir=str(data_root / dataset_name),
                **dataset_params,
            )

            model_config = config["model_config"]
            model_name = model_config["name"]
            model_source = model_config["source"]
            model_alias = model_config.get("alias")
            model_repo = model_config.get("repo")
            model_params = model_config.get("params", {})
            model = get_model(
                name=model_name,
                source=model_source,
                alias=model_alias,
                repo=model_repo,
                cache_dir=model_cache_dir,
                **model_params,
            )
            print("-" * 80)
            print(
                f"Running evaluation for model '{model.name}' on dataset '{dataset.name}'"
            )

            run_dir = output_dir / f"{dataset.name}_{model.name}" / f"split_{split_seed}"
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

    if num_splits > 1:
        df = pd.DataFrame(table_data)
        agg_df = (
            df.groupby(["Dataset", "Model", "Calibrator"])
            .agg(["mean", "std"])
            .reset_index()
        )
        
        for metric in all_metric_names:
            mean_col = (metric, "mean")
            std_col = (metric, "std")
            agg_df[metric] = agg_df[mean_col].apply(lambda x: f"{x:.4f}") + " ± " + agg_df[std_col].apply(lambda x: f"{x:.4f}")

        
        agg_df.columns = [" ".join(col).strip() for col in agg_df.columns.values]
        
        summary_df = agg_df[["Dataset", "Model", "Calibrator", *all_metric_names]]

        summary_table = tabulate(
            summary_df, headers="keys", tablefmt="pipe", showindex=False
        )
        summary_path = output_dir / "summary_results.txt"
        with open(summary_path, "w") as f:
            f.write(summary_table)
        print(summary_table)
    else:
        generate_and_save_summary(table_data, all_metric_names, Path(output_dir))

    return all_reports
