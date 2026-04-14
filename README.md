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

## Notes for thesis-quality reporting

- Fix random seeds and report hardware/software versions.
- Use walk-forward revalidation for final tables.
- Add statistical significance testing (Diebold-Mariano / Wilcoxon signed-rank) for model comparisons.
- Include hyperparameter ranges and compute budgets in the appendix.
