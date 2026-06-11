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

        # Optionally log statistics of the conformal set sizes as additional metric-like values.
        # We both print them, store them in the row used for the summary table, and persist them
        # on the EvaluationReport so they end up in run_reports.pkl even after raw arrays are dropped.
        avg_set_size = None
        q25 = q50 = q75 = None

        if report.conformal_set_sizes is not None:
            set_sizes = report.conformal_set_sizes

            avg_set_size = float(np.mean(set_sizes))
            q25, q50, q75 = np.percentile(set_sizes, [25, 50, 75])

            # Persist on the report so that these statistics are available from run_reports.pkl.
            report.avg_set_size = avg_set_size
            report.set_size_q25 = float(q25)
            report.set_size_q50 = float(q50)
            report.set_size_q75 = float(q75)
        else:
            # When loading from cached run_reports.pkl, conformal_set_sizes may be None,
            # but the pre-computed statistics are still available.
            avg_set_size = getattr(report, "avg_set_size", None)
            q25 = getattr(report, "set_size_q25", None)
            q50 = getattr(report, "set_size_q50", None)
            q75 = getattr(report, "set_size_q75", None)

        if avg_set_size is not None:
            print(f"    avg_set_size: {avg_set_size:.4f}")
            row["avg_set_size"] = float(avg_set_size)

            if q25 is not None:
                print(f"    set_size_q25: {q25:.4f}")
                row["set_size_q25"] = float(q25)
            if q50 is not None:
                print(f"    set_size_q50: {q50:.4f}")
                row["set_size_q50"] = float(q50)
            if q75 is not None:
                print(f"    set_size_q75: {q75:.4f}")
                row["set_size_q75"] = float(q75)

            # Make sure these statistics are treated as metrics when aggregating the summary.
            for extra_metric in ["avg_set_size", "set_size_q25", "set_size_q50", "set_size_q75"]:
                if extra_metric not in all_metric_names:
                    all_metric_names.append(extra_metric)

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

    def format_mean_std(mean_value: float, std_value: float) -> str:
        if pd.isna(mean_value):
            return ""
        if pd.isna(std_value):
            std_value = 0.0
        return f"{mean_value:.4f} ± {std_value:.4f}"

    for column in [*TIMING_COLUMNS, *all_metric_names]:
        mean_col = (column, "mean")
        std_col = (column, "std")
        mean_values = agg_df[mean_col]
        std_values = agg_df[std_col].fillna(0)
        summary_data[column] = [
            format_mean_std(mean_value, std_value)
            for mean_value, std_value in zip(mean_values, std_values)
        ]
    
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
