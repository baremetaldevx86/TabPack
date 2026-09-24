"""Shared pytest fixtures (owned by the integrator; request changes on the board)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.types import TaskType

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


def make_synthetic_dataset(
    *,
    n_train: int = 512,
    n_val: int = 128,
    n_test: int = 128,
    n_num: int = 5,
    cat_cardinalities: tuple[int, ...] = (3, 2),
    seed: int = 0,
) -> PreparedDataset:
    """A small, learnable binclass PreparedDataset on CPU (no preprocessing needed)."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    x_num = rng.standard_normal((n, n_num)).astype(np.float32)
    x_cat = (
        np.stack([rng.integers(0, c, n) for c in cat_cardinalities], axis=1)
        if cat_cardinalities
        else None
    )
    logit = 1.5 * x_num[:, 0] - x_num[:, 1] + 0.5 * x_num[:, 2] * x_num[:, 3]
    if x_cat is not None:
        logit = logit + (x_cat[:, 0] == 1) * 1.0
    y = (logit + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    bounds = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n),
    }

    def split(a):
        return {k: torch.as_tensor(a[lo:hi]) for k, (lo, hi) in bounds.items()}

    return PreparedDataset(
        x_num=split(x_num),
        x_cat=None if x_cat is None else split(x_cat.astype(np.int64)),
        y=split(y),
        task=TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2),
        cat_cardinalities=list(cat_cardinalities),
    )


@pytest.fixture
def synthetic_dataset() -> PreparedDataset:
    return make_synthetic_dataset()


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return torch.device('cuda')
