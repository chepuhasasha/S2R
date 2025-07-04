"""Configuration utilities for the renderer package."""
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load generation settings from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
