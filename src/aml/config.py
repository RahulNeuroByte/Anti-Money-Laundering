"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# AML_PROJECT_ROOT points every data/report/model path at a scratch folder (Azure ML jobs, tests).
PROJECT_ROOT = Path(os.environ.get("AML_PROJECT_ROOT") or Path(__file__).resolve().parents[2])


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML config from `path`, AML_CONFIG_PATH, or the default location."""
    cfg_path = Path(path or os.environ.get("AML_CONFIG_PATH", "configs/config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
