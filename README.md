## Conformal Probability Calibration — concise guide

This repository contains the code accompanying the paper "Adaptive Set-Mass Calibration with Conformal Prediction", submitted to AISTATS 2026.

Short README for paper reproducibility. Launch examples come first.

### Quickstart (uv recommended)

```bash
# 1) Python 3.13 and uv (https://astral.sh/uv). You can simply use `uv run`.

# CIFAR (default config)
uv run -- python eval.py --config_file configs/config_cifar.json

# CIFAR with extra flags
uv run -- python eval.py --config_file configs/config_cifar.json \
  --cal-ratio 0.3 --num-splits 10 --subset-items 50000

# ImageNet-mini
uv run -- python eval.py --config_file configs/config_imagenet.json

# iNaturalist (example without stratification)
uv run -- python eval.py --config_file configs/config_inaturalist.json \
  --cal-ratio 0.3 --num-splits 1 --subset-items 30000 --do-not-stratify
```

### Data placement
- ImageNet-mini: place the dataset under `./data/imagenet-mini` (with `train/` and `val/`).
- iNaturalist 2021 valid: extract under `./data/inaturalist/2021_valid`.
- CIFAR-10/100: downloaded automatically by the loaders; no manual placement needed.

### What the pipeline does
- Loads dataset and model, computes predictions (cached in `test_preds.npz`).
- Splits data into calibration/test by `--cal-ratio` (and across `--num-splits`).
- Applies selected calibrators and computes metrics.
- Optionally generates plots.

### Where to configure experiments
- Config files: `configs/config_cifar.json`, `configs/config_imagenet.json`, `configs/config_inaturalist.json`.
- Key sections: `evaluations` (dataset–model pairs), `calibrators`, `metrics`, `visualizations`, `runner_settings`.
- Data root is `data_root` (default `data/`). Expected layout: `data/<dataset_name>/...`.

### Outputs
- Root folder from `runner_settings.output_dir` (e.g., `experiments_cifar/`).
- Per pair: `<dataset>_<model>/[split_i]/` directories.
- Summary tables: `summary_results.txt` and/or `summary_results.csv` in the `output_dir` root.
- Prediction cache: `<dataset>_<model>/test_preds.npz`.
- Plots (if enabled): saved under respective `split_i` directories.

### Dependencies
- Python 3.13 (`.python-version`).
- Uses `uv` and `pyproject.toml` for dependencies. Running via `uv run` will resolve them automatically.

### Project layout (minimal)
- `caliblab/` — datasets, models, calibrators, metrics, visualizations, and the evaluation engine.
- `eval.py` — the single CLI entrypoint with flags:
  - `--config_file` (path to JSON), `--num-splits`, `--cal-ratio`, `--subset-items`, `--do-not-stratify`.

TL;DR: run one of the commands from “Quickstart”.
