"""Where the data and outputs live.

Set ``DEADAHEAD_DATA`` to the folder that holds the data. If it is unset or empty, ``data/`` inside the repository is
used.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("DEADAHEAD_DATA") or REPO_ROOT / "data").expanduser().resolve()


def raw_dir() -> Path:
    """The extracted CellDeathDetect archives."""
    return data_root() / "raw" / "celldeathdetect"


def tables_dir() -> Path:
    """Movie, track, event, cell and forecast-row tables."""
    return data_root() / "tables"


def crops_dir() -> Path:
    return data_root() / "crops"


def runs_dir() -> Path:
    """Checkpoints, predictions and features written by the pipeline."""
    return data_root() / "runs"


def bundle_dir() -> Path:
    """The released predictions bundle used by ``deadahead reproduce``."""
    return data_root() / "bundle"
