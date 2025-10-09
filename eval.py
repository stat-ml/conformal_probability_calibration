import argparse
from pathlib import Path

from caliblab.eval.runner import run_evaluations
from caliblab.utils.config import parse_config


def main():
    """
    Main entry point for running evaluations from a configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Run model calibration evaluations from a config file."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the JSON configuration file.",
        default="config_cifar.json",
        required=False
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=1,
        help="Number of different train/test splits to average results over.",
    )
    parser.add_argument(
        "--cal-ratio",
        type=float,
        default=0.3,
        help="Ratio of data to use for calibration.",
    )
    parser.add_argument(
        "--subset-items",
        type=int,
        default=1_000_000,
        help="Number of items to use for calibration.",
    )
    parser.add_argument(
        "--do-not-stratify",
        action="store_true",
        help="Do not stratify the data for calibration.",
    )
    args = parser.parse_args()
    config_path = Path(args.config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    configs, calibrators, metrics, runner_settings, visualizers = parse_config(
        config_path
    )

    run_evaluations(
        configs=configs,
        calibrators=calibrators,
        metrics=metrics,
        visualizers=visualizers,
        num_splits=args.num_splits,
        cal_ratio=args.cal_ratio,
        subset_items=args.subset_items,
        do_not_stratify=args.do_not_stratify,
        **runner_settings,
    )


if __name__ == "__main__":
    main()
