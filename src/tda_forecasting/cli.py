from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiments import run_all_experiments
from .plots import write_quick_experiment_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TDA forecasting thesis experiments")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML config")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip regenerating training/experiment figures under plots_dir",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = run_all_experiments(cfg)

    for name, df in outputs.items():
        print(f"[OK] {name}: {len(df)} rows")
    curves = Path(cfg.paths["results_dir"]) / "training_curves.json"
    print(f"[OK] {curves}")

    if not args.no_plots:
        plots_root = Path(cfg.paths.get("plots_dir", "plots_and_visualizations"))
        write_quick_experiment_plots(cfg.paths["results_dir"], plots_root)
        print(f"[OK] plots: {plots_root.resolve()}")


if __name__ == "__main__":
    main()
