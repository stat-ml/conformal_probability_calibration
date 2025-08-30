import argparse

from caliblab.eval.runner import run_evaluations
from caliblab.utils.config import parse_config


def main():
    """
    Main entry point for running evaluations from a config file.
    """
    parser = argparse.ArgumentParser(
        description="Run model evaluations based on a JSON config file."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file.",
        default="config.json",
        nargs="?",
    )
    args = parser.parse_args()

    # Parse the config file to get evaluation setups
    configs, calibrators, metrics, runner_settings = parse_config(args.config_file)

    # Run all evaluations
    run_evaluations(
        configs=configs,
        calibrators=calibrators,
        metrics=metrics,
        **runner_settings,
    )


if __name__ == "__main__":
    main()
