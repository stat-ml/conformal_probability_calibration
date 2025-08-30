import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..calibrators import CalibratorBase, get_calibrator
from ..datasets import BaseDataset, dataset_getter
from ..eval.runner import EvaluationConfig
from ..metrics import MetricBase, get_metric
from ..models import ModelBase, get_model

Config = Dict[str, Any]
EvaluationConfig = Tuple[BaseDataset, ModelBase]


def parse_config(
    config_path: str,
) -> Tuple[List[EvaluationConfig], List[CalibratorBase], List[MetricBase], Dict[str, Any]]:
    """
    Parses a JSON configuration file to set up evaluation runs.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A tuple containing:
        - A list of evaluation configurations (dataset, model).
        - A list of calibrator instances.
        - A list of metric instances.
        - A dictionary of global runner settings.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config: Config = json.load(f)

    # --- Parse global settings ---
    data_root = Path(config.get("data_root", "data"))
    runner_settings = config.get("runner_settings", {})

    # --- Parse calibrator configurations ---
    calibrators: List[CalibratorBase] = []
    for calibrator_config in config.get("calibrators", []):
        if isinstance(calibrator_config, str):
            calibrators.append(get_calibrator(calibrator_config))
        elif isinstance(calibrator_config, dict):
            calibrators.append(
                get_calibrator(
                    calibrator_config["name"], **calibrator_config.get("params", {})
                )
            )

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

    return evaluation_configs, calibrators, metrics, runner_settings
