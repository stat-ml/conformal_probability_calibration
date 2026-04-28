from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from ..models import ModelBase
from .constants import EvaluationReport


TIMING_COLUMNS = ["Fit Time (s)", "Predict Time (s)", "Total Time (s)"]


def print_and_collect_run_results(
    run_reports: List[EvaluationReport],
    dataset_name: str,
    model_name: str,
    table_data: List[Dict[str, Any]],
    all_metric_names: List[str],
) -> None:
    """Prints per-run summary and collects data for the final table."""
    print(f"\nResults for {dataset_name} with {model_name}:")
    for report in run_reports:
        print(f"  Calibrator: {report.calibrator_name}")
        fit_time = getattr(report, "train_time", 0.0)
        predict_time = getattr(report, "predict_time", 0.0)
        total_time = fit_time + predict_time
        print(f"    fit_time: {fit_time:.4f}s")
        print(f"    predict_time: {predict_time:.4f}s")
        print(f"    total_time: {total_time:.4f}s")
        row = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Calibrator": report.calibrator_name,
            "Fit Time (s)": fit_time,
            "Predict Time (s)": predict_time,
            "Total Time (s)": total_time,
        }
        for metric_name, value in report.metrics.items():
            print(f"    {metric_name}: {value:.4f}")
            row[metric_name] = value
        table_data.append(row)


def generate_and_save_summary(
    table_data: List[Dict[str, Any]],
    all_metric_names: List[str],
    output_dir: Path,
) -> None:
    """Formats, prints, and saves the final summary table."""
    if not table_data:
        return

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

    for column in [*TIMING_COLUMNS, *all_metric_names]:
        mean_col = (column, "mean")
        std_col = (column, "std")
        std_values = agg_df[std_col].fillna(0)
        summary_data[column] = agg_df[mean_col].apply(
            lambda x: f"{x:.4f}"
        ) + " ± " + std_values.apply(lambda x: f"{x:.4f}")
    
    summary_df = pd.DataFrame(summary_data)

    summary_table = tabulate(
        summary_df, headers="keys", tablefmt="pipe", showindex=False
    )
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    print(f"Saved summary_results.csv to {output_dir / 'summary_results.csv'}")
    df.to_csv(output_dir / "results.csv", index=False)

    summary_path = output_dir / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write(summary_table)
    print(summary_table)


def get_predictions(
    model: Optional[ModelBase],
    loader: Optional[DataLoader],
    device: torch.device,
    cache_path: Path,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if use_cache and not force_recompute and cache_path.exists():
        print(f"Loading cached predictions from {cache_path}")
        cached_data = np.load(cache_path, allow_pickle=True)
        if "outputs" in cached_data and "labels" in cached_data and "probs" in cached_data:
            probs = cached_data["probs"]
            # np.savez serializes None as a 0-D object array; normalize it back.
            if isinstance(probs, np.ndarray) and probs.ndim == 0 and probs.dtype == object:
                if probs.item() is None:
                    probs = None
            return cached_data["outputs"], cached_data["labels"], probs
        elif "outputs" in cached_data and "labels" in cached_data:
            return cached_data["outputs"], cached_data["labels"], None
        elif "logits" in cached_data and "true_labels" in cached_data:
            return cached_data["logits"], cached_data["true_labels"], None
        else:
            raise KeyError(
                "Could not find 'outputs'/'labels' or 'logits'/'true_labels' in cached file."
            )

    if model is None or loader is None:
        raise RuntimeError(
            f"Prediction cache not found at {cache_path}; a model and loader are required to compute predictions."
        )

    print(f"Computing predictions and saving to {cache_path}")
    outputs, labels, probs = model.predict(loader, device)
    if probs is None:
        np.savez(cache_path, outputs=outputs, labels=labels)
    else:
        np.savez(cache_path, outputs=outputs, labels=labels, probs=probs)
    return outputs, labels, probs
