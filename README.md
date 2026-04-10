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

## Methodological grounding

The implementation aligns with:

- Chaos (2025): end-to-end TDA pipeline for time-series/phase-space and machine-learning features from persistence descriptors.
- Entropy (2023): multivariate time-series TDA principles, metric/design sensitivity, and outlier stability concerns.
- Neural Computing and Applications (2025): financial forecasting with sliding-window TDA descriptors (entropy/amplitude/point count), and feature-augmented predictive models.

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

## Notes for thesis-quality reporting

- Fix random seeds and report hardware/software versions.
- Use rolling-origin (walk-forward) revalidation for final tables.
- Add statistical significance testing (Diebold-Mariano / Wilcoxon signed-rank) for model comparisons.
- Include hyperparameter ranges and compute budgets in the appendix.
