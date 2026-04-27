"""Shared utilities for experiment setup, logging, and configuration."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable smoke tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must contain a mapping.")
    return data


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_dict_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return

    output_path = ensure_parent_dir(path)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_dict_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append dictionaries to CSV, writing the header if the file is new."""
    if not rows:
        return

    output_path = ensure_parent_dir(path)
    fieldnames = list(rows[0].keys())
    should_write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def flatten_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """Flatten one level of nested metrics for CSV-friendly logging."""
    return {f"{prefix}_{key}": value for key, value in values.items()}
