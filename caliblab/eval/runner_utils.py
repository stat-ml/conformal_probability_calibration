from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from ..datasets import BaseDataset
from ..models import ModelBase
from .constants import EvaluationReport


def print_and_collect_run_results(
    run_reports: List[EvaluationReport],
    dataset: BaseDataset,
    model: ModelBase,
    table_data: List[Dict[str, Any]],
    all_metric_names: List[str],
) -> None:
    """Prints per-run summary and collects data for the final table."""
    print(f"\nResults for {dataset.name} with {model.name}:")
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
    print(f"Saved summary_results.csv to {output_dir / 'summary_results.csv'}")
    df.to_csv(output_dir / "results.csv", index=False)

    summary_path = output_dir / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write(summary_table)
    print(summary_table)


def get_predictions(
    model: ModelBase,
    loader: DataLoader,
    device: torch.device,
    cache_path: Path,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_cache and not force_recompute and cache_path.exists():
        print(f"Loading cached predictions from {cache_path}")
        cached_data = np.load(cache_path, allow_pickle=True)
        if "outputs" in cached_data and "labels" in cached_data and "probs" in cached_data:
            return cached_data["outputs"], cached_data["labels"], cached_data["probs"]
        elif "outputs" in cached_data and "labels" in cached_data:
            return cached_data["outputs"], cached_data["labels"]
        elif "logits" in cached_data and "true_labels" in cached_data:
            return cached_data["logits"], cached_data["true_labels"]
        else:
            raise KeyError(
                "Could not find 'outputs'/'labels' or 'logits'/'true_labels' in cached file."
            )

    print(f"Computing predictions and saving to {cache_path}")
    outputs, labels, probs = model.predict(loader, device)
    np.savez(cache_path, outputs=outputs, labels=labels, probs=probs)
    return outputs, labels, probs
