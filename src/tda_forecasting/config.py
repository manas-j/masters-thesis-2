from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    data: dict[str, Any]
    features: dict[str, Any]
    training: dict[str, Any]
    experiments: dict[str, Any]
    optimization: dict[str, Any]
    paths: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ExperimentConfig(
        data=cfg["data"],
        features=cfg["features"],
        training=cfg["training"],
        experiments=cfg["experiments"],
        optimization=cfg["optimization"],
        paths=cfg["paths"],
    )
