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
        "config_file",
        type=str,
        help="Path to the JSON configuration file.",
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
        **runner_settings,
    )


if __name__ == "__main__":
    main()
