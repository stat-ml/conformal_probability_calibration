import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..calibrators import CalibratorBase, get_calibrator
from ..datasets import BaseDataset, dataset_getter
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

    # --- Parse calibrators ---
    calibrators = []
    for cal_config in config.get("calibrators", []):
        if isinstance(cal_config, str):
            name = cal_config
            params = {}
        elif isinstance(cal_config, dict):
            name = cal_config["name"]
            params = cal_config.get("params", {})
        else:
            raise TypeError(f"Invalid format for calibrator config: {cal_config}")

        if name.lower().strip() == "none":
            continue
        calibrators.append(get_calibrator(name, **params))

    # --- Parse metrics ---
    metrics = []
    for metric_config in config.get("metrics", []):
        if isinstance(metric_config, str):
            name = metric_config
            params = {}
        else:
            name = metric_config["name"]
            params = metric_config.get("params", {})
        metrics.append(get_metric(name, **params))

    # --- Parse evaluation configurations ---
    evaluation_configs: List[EvaluationConfig] = []
    for eval_config in config.get("evaluations", []):
        # Instantiate dataset
        dataset_name = eval_config["dataset"]
        dataset_params = eval_config.get("dataset_params", {})
        dataset = dataset_getter(
            dataset_name, data_dir=str(data_root / dataset_name), **dataset_params
        )

        # Instantiate model
        model_name = eval_config["model"]
        model_params = eval_config.get("model_params", {})
        model = get_model(model_name, **model_params)

        evaluation_configs.append((dataset, model))

    return evaluation_configs, calibrators, metrics, runner_settings
