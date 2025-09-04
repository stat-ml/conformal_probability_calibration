import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ..calibrators import (
    CALIBRATORS,
    CalibratorBase,
    ConformalCalibrator,
    get_calibrator,
)
from ..datasets import BaseDataset, dataset_getter
from ..eval.runner import EvaluationConfig
from ..metrics import MetricBase, get_metric
from ..models import ModelBase, get_model
from ..visualizations import ConfidenceVisualizer, CumulativeMassVisualizer

# A simple type alias for the JSON config
Config = Dict[str, Any]
EvaluationConfig = Tuple[BaseDataset, ModelBase]


def parse_config(
    config_path: Path,
) -> Tuple[
    List[EvaluationConfig],
    List[CalibratorBase],
    List[MetricBase],
    Dict[str, Any],
    List[Any],
]:
    with config_path.open("r") as f:
        try:
            config: Config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing {config_path}: {e}") from e

    # --- Parse runner settings ---
    runner_settings = config.get("runner_settings", {})
    output_dir = Path(runner_settings.get("output_dir", "experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_settings["output_dir"] = output_dir

    data_root_str = config.get("data_root", "data")
    data_root = Path(data_root_str)
    data_root.mkdir(parents=True, exist_ok=True)

    # --- Parse visualization settings ---
    visualizers = []
    vis_config = config.get("visualizations", {})
    if "confidence_curve" in vis_config:
        n_bins = vis_config["confidence_curve"].get("n_bins", 15)
        visualizers.append(ConfidenceVisualizer(n_bins=n_bins))
    if "cumulative_mass_curve" in vis_config:
        n_bins = vis_config["cumulative_mass_curve"].get("n_bins", 15)
        visualizers.append(CumulativeMassVisualizer(n_bins=n_bins))

    # --- Parse calibrator configurations ---
    calibrators: List[CalibratorBase] = []
    for calibrator_config in config.get("calibrators", []):
        if isinstance(calibrator_config, str):
            calibrators.append(get_calibrator(calibrator_config))
        elif isinstance(calibrator_config, dict):
            name = calibrator_config.get("name")
            if name == "conformal_calibrator":
                params = calibrator_config.get("params", {})
                calibrators.append(ConformalCalibrator(**params))
            elif name:
                calibrators.append(get_calibrator(name))

    # --- Parse metric configurations ---
    metrics: List[MetricBase] = []
    for metric_config in config.get("metrics", []):
        if isinstance(metric_config, str):
            metrics.append(get_metric(metric_config))
        elif isinstance(metric_config, dict):
            metrics.append(
                get_metric(metric_config["name"], **metric_config.get("params", {}))
            )

    # --- Parse evaluation configurations ---
    evaluation_configs: List[EvaluationConfig] = []
    for eval_config in config.get("evaluations", []):
        # Instantiate dataset
        dataset_config = eval_config["dataset"]
        dataset_name = dataset_config["name"]
        dataset_params = dataset_config.get("params", {})
        dataset = dataset_getter(
            dataset_name, data_dir=str(data_root / dataset_name), **dataset_params
        )

        # Instantiate model
        model_config = eval_config["model"]
        model_name = model_config["name"]
        model_source = model_config["source"]
        model_alias = model_config.get("alias")
        model_repo = model_config.get("repo")  # Can be None
        model_params = model_config.get("params", {})
        model = get_model(
            name=model_name,
            source=model_source,
            alias=model_alias,
            repo=model_repo,
            cache_dir=runner_settings.get("model_cache_dir"),
            **model_params,
        )

        evaluation_configs.append((dataset, model))

    return evaluation_configs, calibrators, metrics, runner_settings, visualizers
