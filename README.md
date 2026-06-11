# Conformal Probability Calibration

Code for the paper [Adaptive Set-Mass Calibration with Conformal Prediction](https://arxiv.org/abs/2505.15437).

The main entry point is `eval.py`. Experiments are configured with JSON files in `configs/`.

## Quickstart

Use Python 3.13 and `uv`.

```bash
# CIFAR-10 demo: all configured metrics, averaged over 3 splits.
uv run -- python eval.py \
  --config_file configs/config_cifar.json \
  --num-splits 3 \
  --cal-ratio 0.3 \
  --subset-items 10000

# Synthetic example. Prepare the model weights first with notebooks/synthetic_experiment.ipynb.
uv run -- python eval.py --config_file configs/config_synthetic.json

# ImageNet-mini.
uv run -- python eval.py --config_file configs/config_imagenet.json

# iNaturalist.
uv run -- python eval.py \
  --config_file configs/config_inaturalist.json \
  --num-splits 1 \
  --subset-items 30000 \
  --do-not-stratify
```

## Data

- CIFAR-10 is downloaded automatically by `torchvision`.
- ImageNet-mini should be placed under `data/imagenet-mini` with `train/` and `val/`.
- iNaturalist 2021 validation data should be extracted under `data/inaturalist/2021_valid`.

## Outputs

Each run writes to the configured `runner_settings.output_dir`, for example `experiments_cifar/`.

- Cached predictions: `<dataset>_<model>/test_preds.npz`.
- Per-split reports: `<dataset>_<model>/split_<i>/run_reports.pkl`.
- Tables: `results.csv`, `summary_results.csv`, and `summary_results.txt`.
- Plots are written inside each split directory when visualizations are enabled.

Generated data, experiment outputs, plots, model weights, and tables are ignored by Git.

## Project Layout

- `caliblab/`: datasets, models, calibrators, metrics, visualizations, and evaluation code.
- `configs/`: reproducible experiment configs.
- `eval.py`: CLI entry point.
