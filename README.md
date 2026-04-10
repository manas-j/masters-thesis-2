# TDA Forecasting Thesis Experiments

This repository provides a reproducible, paper-style experimental framework for multivariate time-series forecasting with Topological Data Analysis (TDA), including robust density-filtered persistent homology features.

## What is implemented

- Full experiment suite matching the thesis protocol in `intruct.md`:
  - Baseline comparison (`LSTM`, `XGBoost`, `TDA-LSTM`, robust density-filtered `TDA-LSTM`)
  - Impact of multidimensionality (target-only vs joint multivariate embedding)
  - Robustness ablation with synthetic noise stress testing
  - Heuristic parameter selection vs Bayesian optimization (Optuna)
  - SHAP-based interpretability with XGBoost surrogate
- Segment-wise TDA feature extraction using sliding windows:
  - Takens embedding
  - Persistence diagrams (`H0`, `H1`, `H2`) via `ripser`
  - Entropy, amplitude, number-of-points, and persistence-image statistics
- Density filtering via KDE before Vietoris-Rips construction


## Installation (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Run all experiments

```bash
tda-exp --config configs/experiment.yaml
```

Outputs are written to `results/`:

- `experiment_1_baseline.csv`
- `experiment_2_multidimensionality.csv`
- `experiment_3_robustness.csv`
- `experiment_4_optimization.csv`
- `experiment_5_shap_importance.csv`

## Generate plots and visualizations

Create point-cloud views, TDA feature plots, and experiment result figures:

```bash
uv run python scripts/generate_visualizations.py --config configs/experiment.yaml --visual-dir plots_and_visualizations
```

Or use the helper script:

```bash
bash scripts/run_visualizations.sh
```

The visual output directory will contain:

- `plots_and_visualizations/point_clouds/`
- `plots_and_visualizations/features/`
- `plots_and_visualizations/experiments/`