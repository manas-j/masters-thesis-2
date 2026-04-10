#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
tda-exp --config configs/experiment.yaml
