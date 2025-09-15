import pandas as pd
from typing import List, Optional, Tuple, Any, Dict, Set
from pathlib import Path
import torch
from tabulate import tabulate

import os

import numpy as np

from ..calibrators.base import CalibratorBase
from ..datasets import dataset_getter
from ..datasets.utils import split_data
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

    for config in configs:
        dataset_config = config["dataset_config"]
        dataset_name = dataset_config["name"]
        dataset_params = dataset_config.get("params", {})
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

        base_run_dir = output_dir / f"{dataset.name}_{model.name}"
        base_run_dir.mkdir(parents=True, exist_ok=True)

        test_loader = dataset.get_test_loader(batch_size=512)
        test_preds_path = base_run_dir / "test_preds.npz"

        pred_evaluator = ModelEvaluator(
            model=model,
            metrics=[],
            calibrators=[],
            run_dir=base_run_dir,
            device=device,
        )
        test_outputs, test_labels = pred_evaluator.get_predictions(
            test_loader, test_preds_path, use_cache, force_recompute
        )

        for split_seed in range(num_splits):
            print("*" * 80)
            print(f"Running split with seed {split_seed}")
            print("-" * 80)
            print(
                f"Running evaluation for model '{model.name}' on dataset '{dataset.name}'"
            )

            run_dir = base_run_dir / f"split_{split_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cal_ratio = dataset_params.get("cal_ratio", 0.5)
            cal_outputs, test_outputs_split, cal_labels, test_labels_split = split_data(
                test_outputs, test_labels, cal_ratio, split_seed
            )

            evaluator = ModelEvaluator(
                model=model,
                metrics=metrics,
                calibrators=calibrators,
                run_dir=run_dir,
                device=device,
            )
            run_reports = evaluator.run_calibration_and_metrics(
                cal_outputs, cal_labels, test_outputs_split, test_labels_split
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
        
        summary_data = {
            "Dataset": agg_df[("Dataset", "")],
            "Model": agg_df[("Model", "")],
            "Calibrator": agg_df[("Calibrator", "")],
        }

        for metric in all_metric_names:
            mean_col = (metric, "mean")
            std_col = (metric, "std")
            std_values = agg_df[std_col].fillna(0)
            summary_data[metric] = agg_df[mean_col].apply(
                lambda x: f"{x:.4f}"
            ) + " ± " + std_values.apply(lambda x: f"{x:.4f}")
        
        summary_df = pd.DataFrame(summary_data)

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
