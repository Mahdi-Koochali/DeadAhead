"""Read the YAML settings in ``configs/``."""

from __future__ import annotations

from pathlib import Path

import yaml

from deadahead.paths import REPO_ROOT


def read_config(name: str, path: Path | None = None) -> dict:
    """``configs/<name>.yaml`` as a dict, or the file at ``path``."""
    p = Path(path) if path is not None else REPO_ROOT / "configs" / f"{name}.yaml"
    return yaml.safe_load(p.read_text())


def windows_config(path: Path | None = None) -> dict:
    """Frame interval, landmark grid, history length, horizons and truncation."""
    return read_config("windows", path)


def head_config(path: Path | None = None) -> dict:
    """The survival head's time grid and the landmark panel it is fitted on."""
    return read_config("survival_head", path)
