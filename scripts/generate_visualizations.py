#!/usr/bin/env python3
"""Regenerate thesis figures from experiment outputs and raw data (see Chapter 4)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from tda_forecasting.config import load_config
from tda_forecasting.plots import (
    export_feature_matrix_csv,
    plot_experiment_metrics,
    plot_feature_correlation_heatmap,
    plot_persistence_diagram_sample,
    plot_point_cloud_panel,
    plot_tda_feature_histograms,
    plot_training_curves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TDA / experiment plots for the thesis")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results CSV location")
    parser.add_argument("--plots-dir", type=str, default=None, help="Override plots output root")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir or cfg.paths.get("results_dir", "results"))
    plots_dir = Path(args.plots_dir or cfg.paths.get("plots_dir", "plots_and_visualizations"))

    curves = results_dir / "training_curves.json"
    feat_csv = plots_dir / "_cache" / "robust_tda_features.csv"

    plot_tasks: list[tuple[str, Callable[[], None]]] = [
        ("training curves", lambda: plot_training_curves(curves, plots_dir / "training")),
        ("experiment metrics", lambda: plot_experiment_metrics(results_dir, plots_dir)),
        ("export TDA feature CSV", lambda: export_feature_matrix_csv(cfg, feat_csv)),
        ("TDA histograms", lambda: plot_tda_feature_histograms(feat_csv, plots_dir)),
        ("feature correlation", lambda: plot_feature_correlation_heatmap(feat_csv, plots_dir)),
        ("point cloud panel", lambda: plot_point_cloud_panel(cfg, plots_dir)),
        ("persistence diagram sample", lambda: plot_persistence_diagram_sample(cfg, plots_dir)),
    ]

    for label, run in tqdm(plot_tasks, desc="Figures"):
        run()

    print(f"[OK] Figures written under {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
