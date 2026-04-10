#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/generate_visualizations.py --config configs/experiment.yaml --visual-dir plots_and_visualizations
