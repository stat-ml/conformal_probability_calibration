## Architecture

The framework is designed to be modular and extensible. The core components are organized into the `caliblab` package:

-   `caliblab/datasets`: Contains data loader implementations for different datasets (e.g., CIFAR-10, CIFAR-100, ImageNet-mini). Each dataset inherits from a `BaseDataset` and is responsible for providing data loaders for calibration and testing splits.
-   `caliblab/models`: Provides wrappers for models from various sources like `torch.hub` and Hugging Face Transformers. Models inherit from a `ModelBase`, ensuring a consistent interface for the evaluation pipeline.
-   `caliblab/metrics`: Implements various evaluation metrics such as Accuracy, ECE, MCE, NLL, and Brier Score. Each metric inherits from `MetricBase`.
-   `caliblab/calibrators`: Includes implementations of post-hoc calibration methods like Temperature Scaling and Isotonic Regression, inheriting from `CalibratorBase`.
-   `caliblab/eval`: The core evaluation engine.
    -   `evaluator.py`: `ModelEvaluator` class orchestrates the process of fetching model predictions, applying calibrators, and computing metrics for a single model-dataset pair.
    -   `runner.py`: The high-level `run_evaluations` function reads parsed configurations and manages the overall evaluation loop across all specified experiments.
-   `caliblab/visualizations`: Contains logic for generating calibration plots, such as confidence calibration curves and cumulative mass curves.
-   `eval.py`: The main command-line entry point for running the evaluation pipeline.
-   `config.json`: A JSON file that defines the entire evaluation workflow, including datasets, models, metrics, and output settings.

## Installation

The project uses `uv` for dependency management.

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  Install the required packages:
    ```bash
    uv pip install -e .
    ```
    This installs the project in editable mode and fetches all dependencies listed in `pyproject.toml`.

## How to Run Evaluations

All evaluations are controlled via a JSON configuration file. To run the pipeline, execute `eval.py` and pass the path to your config file.

```bash
python eval.py config.json
```

## Configuration File (`config.json`)

The JSON configuration file is the primary interface for defining experiments. It has the following main sections:

### `runner_settings`
Global settings for the evaluation runner.

-   `output_dir` (string): Path to the directory where all results (logs, plots, summary tables) will be saved.
-   `use_cache` (boolean): If `true`, the runner will reuse cached model predictions (`predictions.npz`) if they exist.
-   `force_recompute` (boolean): If `true`, forces re-computation of model predictions, ignoring any existing cache.
-   `model_cache_dir` (string, optional): Path to a directory for caching downloaded model weights.

### `evaluations`
A list of experiment configurations. Each element is an object defining a dataset-model pair to be evaluated.

-   **`dataset`**:
    -   `name` (string): The name of the dataset (e.g., "cifar10", "cifar100"). Must match a registered dataset in the framework.
    -   `params` (object, optional): A dictionary of parameters to pass to the dataset's constructor (e.g., `{"data_dir": "path/to/data", "image_size": 224}`).
-   **`model`**:
    -   `source` (string): The source of the model. Supported values: `"torch_hub"`, `"vit"`.
    -   `repo` (string, required for `torch_hub`): The repository name for `torch.hub.load()`.
    -   `name` (string): The name of the model to load from the source.
    -   `alias` (string, optional): A shorter, user-friendly name for logging and reporting.
    -   `params` (object, optional): A dictionary of parameters to pass to the model loader (e.g., `{"num_labels": 100}`).

### `calibrators`
A list of strings specifying which calibration methods to apply. An `"uncalibrated"` baseline is always included.
Example: `["temperature_scaling", "isotonic_regression"]`

### `metrics`
A list of metrics to compute for each calibrator. Metrics can be specified as a simple string or as an object to include parameters.

-   `"accuracy"`
-   `{ "name": "ece", "params": { "n_bins": 20 } }`

### `visualizations`
Configuration for generating plots.

-   **`confidence_curve`**:
    -   `n_bins` (int): Number of bins for the confidence calibration curve.
-   **`cumulative_mass_curve`**:
    -   `n_bins` (int): Number of bins for the cumulative mass calibration curve.
