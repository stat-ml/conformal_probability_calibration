import pandas as pd
from typing import List, Optional, Tuple, Any, Dict, Set
from pathlib import Path
import torch
from tabulate import tabulate

import os
import pickle

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
    get_predictions,
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
    cal_ratio: float = 0.3,
    subset_items: int = 40_000,
    **kwargs: Any,
) -> List[EvaluationReport]:
    all_reports: List[EvaluationReport] = []
    table_data = []
    all_metric_names: List[str] = [metric.name for metric in metrics]

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

        test_outputs, test_labels = get_predictions(
            model, test_loader, device, test_preds_path, use_cache, force_recompute
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

            reports_path = run_dir / "run_reports.pkl"

            from_cache = False
            if use_cache and not force_recompute and reports_path.exists():
                print(f"Loading cached run_reports from {reports_path}")
                with open(reports_path, "rb") as f:
                    run_reports = pickle.load(f)
                from_cache = True
            else:
                cal_outputs, test_outputs_split, cal_labels, test_labels_split = split_data(
                    test_outputs, test_labels, cal_ratio, split_seed, subset_items
                )

                evaluator = ModelEvaluator(
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

            if visualizers and not from_cache:
                for visualizer in visualizers:
                    visualizer.plot(
                        reports=run_reports,
                        run_dir=run_dir,
                        dataset_name=dataset.name,
                        model_name=model.name,
                    )

            # Drop large arrays before saving/aggregating to keep memory and cache light
            for report in run_reports:
                report.calibrated_probabilities = None
                report.true_labels = None
                report.conformal_set_sizes = None

            # Cache pruned run_reports for this split if freshly computed
            if not from_cache:
                try:
                    with open(reports_path, "wb") as f:
                        pickle.dump(run_reports, f)
                    print(f"Saved run_reports to {reports_path}")
                except Exception as e:
                    print(f"Warning: failed to save run_reports cache: {e}")

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
        summary_df.to_csv(output_dir / "summary_results.csv", index=False)
        summary_path = output_dir / "summary_results.txt"
        with open(summary_path, "w") as f:
            f.write(summary_table)
        print(summary_table)
    else:
        generate_and_save_summary(table_data, all_metric_names, Path(output_dir))

    return all_reports
