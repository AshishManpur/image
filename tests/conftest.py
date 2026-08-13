"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.seed import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Every test runs from the same seed."""
    set_seed(1337)


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")
