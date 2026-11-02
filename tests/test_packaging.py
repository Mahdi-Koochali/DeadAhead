"""Packaging: the pinned PyTorch stack, the ruff settings, the licences and the test harness."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys

import pytest

from deadahead import __version__
from deadahead.paths import REPO_ROOT

PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text()


def test_video_extra_is_the_tested_set():
    extra = re.search(r"^video = \[(.*)\]$", PYPROJECT, re.M).group(1)
    pins = dict(p.split("==") for p in re.findall(r'"([^"]+)"', extra))
    tested = dict(line.split("==") for line in (REPO_ROOT / "environment" / "gpu-py310.txt").read_text().splitlines()
                  if "==" in line)
    assert set(pins) == {"torch", "transformers", "safetensors", "huggingface_hub"}
    assert pins == {k: tested[k] for k in pins}


def test_ruff_is_pinned_and_configured():
    assert re.search(r'^dev = \[.*"ruff==[\d.]+".*\]$', PYPROJECT, re.M)
    assert 'select = ["E9", "F"]' in PYPROJECT


def test_versions_agree():
    version = re.search(r'^version = "(.+)"$', PYPROJECT, re.M).group(1)
    citation = (REPO_ROOT / "CITATION.cff").read_text()
    assert version == __version__ and f"version: {version}\n" in citation


def test_notice_names_the_changes_to_both_datasets():
    notice = (REPO_ROOT / "NOTICE.md").read_text()
    assert "built from CellDeathDetect are shared under CC BY 4.0" in notice
    assert "built from DetecDiv are shared under CC BY 4.0" in notice


@pytest.mark.torch
def test_transformers_accepts_this_pytorch():
    pytest.importorskip("torch")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers is not installed")
    from transformers.utils import is_torch_available           # a failure here is a broken pair, not a missing extra
    assert is_torch_available()


@pytest.mark.torch
def test_a_thread_change_does_not_reach_the_next_test(tmp_path):
    pytest.importorskip("torch")
    shutil.copy(REPO_ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_order.py").write_text(
        "def test_a():\n"
        "    import torch\n"
        "    torch.set_num_threads(3)\n\n\n"
        "def test_b():\n"
        "    import torch\n"
        "    assert torch.get_num_threads() == 1\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
