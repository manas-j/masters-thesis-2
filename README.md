# Experiments

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

## Installation

```bash
uv sync
```

## Run all experiments

```bash
uv run tda-exp --config configs/experiment.yaml
```

For a quicker local check (shorter history, fewer Optuna trials):

```bash
uv run tda-exp --config configs/experiment_smoke.yaml
```

## Plots (training curves, TDA diagnostics, experiment bars)

After experiments have produced `results/` (or `results_smoke/`), regenerate figures:

```bash
uv run python scripts/generate_visualizations.py --config configs/experiment.yaml
# or, for smoke outputs:
uv run python scripts/generate_visualizations.py --config configs/experiment_smoke.yaml --results-dir results_smoke
```

Outputs are written to `results/`:

- `experiment_1_baseline.csv`
- `experiment_2_multidimensionality.csv`
- `experiment_3_robustness.csv`
- `experiment_4_optimization.csv`
- `experiment_5_shap_importance.csv`
