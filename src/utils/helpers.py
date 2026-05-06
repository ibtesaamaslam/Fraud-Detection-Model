# helpers.py
# ----------
# Reusable utility functions shared across the project.
# Covers directory initialisation, reproducibility seeding,
# JSON / plain-text I/O, and lightweight value formatting.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import ALL_DIRS


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def ensure_project_directories() -> None:
    """Create all required project directories if they do not already exist.

    Iterates over every path listed in ALL_DIRS and calls
    mkdir(parents=True, exist_ok=True) so that downstream code can write
    files without performing its own existence checks.
    """
    # Iterate over every registered project directory and create it if missing
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed the Python and NumPy random-number generators for reproducibility.

    Call this once at the top of any entry-point script to ensure that
    stochastic operations yield identical results across runs.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    # Seed the standard-library RNG
    random.seed(seed)

    # Seed NumPy's global RNG to cover all np.random calls
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    """Serialise data to a UTF-8 encoded JSON file at path.

    Args:
        data: Any JSON-serialisable Python object.
        path: Destination file path (e.g. Path("outputs/results.json")).
    """
    # Ensure the destination directory exists before writing
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write with 2-space indentation for human-readable output
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: Path) -> Any:
    """Deserialise a UTF-8 encoded JSON file and return the Python object.

    Args:
        path: Path to an existing .json file.

    Returns:
        The deserialised Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    # Open in read mode and deserialise directly from the file handle
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Plain-text I/O
# ---------------------------------------------------------------------------

def save_text(text: str, path: Path) -> None:
    """Write text to a UTF-8 encoded plain-text or Markdown file at path.

    Args:
        text: String content to write.
        path: Destination file path (e.g. Path("reports/summary.md")).
    """
    # Ensure the destination directory exists before writing
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def human_percent(value: float) -> str:
    """Format a decimal fraction as a human-readable percentage string.

    Example:
        human_percent(0.8325) -> '83.25%'

    Args:
        value: A decimal fraction (e.g. 0.83 for 83%).

    Returns:
        A string formatted to two decimal places followed by %.
    """
    # Multiply by 100 and format to exactly two decimal places
    return f"{value * 100:.2f}%"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to float, returning default on failure.

    Useful when processing external data that may contain None,
    empty strings, or non-numeric types.

    Args:
        value:   The value to convert.
        default: Fallback value returned when conversion fails. Defaults to 0.0.

    Returns:
        float(value) on success, otherwise default.
    """
    try:
        return float(value)
    # Catch type mismatches (e.g. None, list) and invalid string literals
    except (TypeError, ValueError):
        return default