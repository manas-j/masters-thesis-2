#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tda_forecasting.config import load_config
from tda_forecasting.visualization import generate_all_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and visualizations for TDA thesis experiments")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML config")
    parser.add_argument(
        "--visual-dir",
        type=str,
        default="plots_and_visualizations",
        help="Directory where figures will be written",
    )
    parser.add_argument(
        "--strict-experiments",
        action="store_true",
        help="Fail if experiment result CSV files are missing",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = generate_all_visuals(
        cfg=cfg,
        visual_dir=args.visual_dir,
        skip_missing_results=not args.strict_experiments,
    )
    print(f"[OK] Visualizations written to: {out['visual_dir']}")
    print(f"[INFO] Reading experiment CSVs from: {out['results_dir']}")
    if out["experiment_plots"] == "skipped_missing_results":
        print("[WARN] Experiment CSV files not found; generated point cloud and feature plots only.")


if __name__ == "__main__":
    main()
