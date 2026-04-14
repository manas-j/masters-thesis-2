#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/generate_visualizations.py --config configs/experiment.yaml
