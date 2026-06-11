import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ..calibrators import (
    CalibratorBase,
    get_calibrator,
)
from ..datasets import BaseDataset, dataset_getter
from ..eval.runner import EvaluationConfig
from ..metrics import MetricBase, get_metric
from ..models import ModelBase, get_model
from ..visualizations import (
    ConfidenceVisualizer,
    CumulativeMassVisualizer,
    OneMinusAlphaCoverageVisualizer,
    ConformalSetSizeVisualizer,
    AlphaSuffixCoverageVisualizer,
)

# A simple type alias for the JSON config
Config = Dict[str, Any]


def parse_config(
    config_path: Path,
) -> Tuple[
    List[Dict[str, Any]],
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
    runner_settings["data_root"] = data_root

    # --- Parse visualization settings ---
    visualizers = []
    vis_config = config.get("visualizations", {})
    if "confidence_curve" in vis_config:
        n_bins = vis_config["confidence_curve"].get("n_bins", 15)
        visualizers.append(ConfidenceVisualizer(n_bins=n_bins))
    if "cumulative_mass_curve" in vis_config:
        params = vis_config["cumulative_mass_curve"]
        # Forward only accepted params
        sig = inspect.signature(CumulativeMassVisualizer.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        filtered_params = {k: v for k, v in params.items() if k in accepted_params}
        visualizers.append(CumulativeMassVisualizer(**filtered_params))
    if "one_minus_alpha_coverage_curve" in vis_config:
        params = vis_config["one_minus_alpha_coverage_curve"]
        # Filter params to those accepted by the visualizer constructor to avoid unexpected kwargs
        sig = inspect.signature(OneMinusAlphaCoverageVisualizer.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        filtered_params = {k: v for k, v in params.items() if k in accepted_params}
        visualizers.append(OneMinusAlphaCoverageVisualizer(**filtered_params))
    if "alpha_suffix_coverage_curve" in vis_config:
        params = vis_config["alpha_suffix_coverage_curve"]
        sig = inspect.signature(AlphaSuffixCoverageVisualizer.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        filtered_params = {k: v for k, v in params.items() if k in accepted_params}
        visualizers.append(AlphaSuffixCoverageVisualizer(**filtered_params))
    if "conformal_set_size_distribution" in vis_config:
        visualizers.append(ConformalSetSizeVisualizer())

    # --- Parse calibrator configurations ---
    calibrators: List[CalibratorBase] = []
    for calibrator_config in config.get("calibrators", []):
        if isinstance(calibrator_config, str):
            calibrators.append(get_calibrator(calibrator_config))
        elif isinstance(calibrator_config, dict):
            name = calibrator_config.get("name")
            params = calibrator_config.get("params", {})
            calibrators.append(get_calibrator(name, **params))
        else:
            raise ValueError(f"Unknown calibrator config: {calibrator_config}")

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
    evaluation_configs: List[Dict[str, Any]] = []
    for eval_config in config.get("evaluations", []):
        dataset_config = eval_config["dataset"]
        model_config = eval_config["model"]
        evaluation_configs.append(
            {"dataset_config": dataset_config, "model_config": model_config}
        )

    return evaluation_configs, calibrators, metrics, runner_settings, visualizers
