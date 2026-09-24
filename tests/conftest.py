"""Shared pytest fixtures (owned by the integrator; request changes on the board)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from _helpers import make_synthetic_dataset

from tabpack_repro.data.pipeline import PreparedDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return Path(os.environ.get('TABPACK_DATA_DIR', PROJECT_DIR / 'data'))


@pytest.fixture
def churn_dir() -> Path:
    """Path of the real Churn dataset; skips the test when it is not downloaded."""
    path = _data_dir() / 'churn'
    if not (path / 'info.json').exists():
        pytest.skip(f'Churn is not available at {path} (run `tabpack-repro download`)')
    return path


@pytest.fixture
def synthetic_dataset() -> PreparedDataset:
    """Default small synthetic binclass dataset; for custom sizes use
    ``from _helpers import make_synthetic_dataset``."""
    return make_synthetic_dataset()


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return torch.device('cuda')
