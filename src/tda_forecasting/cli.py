from __future__ import annotations

import argparse

from .config import load_config
from .experiments import run_all_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TDA forecasting thesis experiments")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = run_all_experiments(cfg)

    for name, df in outputs.items():
        print(f"[OK] {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
