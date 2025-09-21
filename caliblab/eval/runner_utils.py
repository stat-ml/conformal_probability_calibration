from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
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

    headers = ["Dataset", "Model", "Calibrator"] + (list(all_metric_names))

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
    output_path = output_dir / "summary_results.txt"
    with open(output_path, "w") as f:
        f.write("Summary of Results\n")
        f.write("=" * 80 + "\n")
        f.write(table_string)
    print(f"\nResults table saved to {output_path}")


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
        cached_data = np.load(cache_path)
        if "outputs" in cached_data and "labels" in cached_data:
            return cached_data["outputs"], cached_data["labels"]
        elif "logits" in cached_data and "true_labels" in cached_data:
            return cached_data["logits"], cached_data["true_labels"]
        else:
            raise KeyError(
                "Could not find 'outputs'/'labels' or 'logits'/'true_labels' in cached file."
            )

    print(f"Computing predictions and saving to {cache_path}")
    outputs, labels = model.predict(loader, device)
    np.savez(cache_path, outputs=outputs, labels=labels)
    return outputs, labels
