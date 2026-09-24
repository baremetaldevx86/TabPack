"""Seeding (a29)."""

from __future__ import annotations

import operator
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python `random`, numpy's global RNG and torch (CPU + all CUDA devices).

    `seed` must be an integer in [0, 2**32) (numpy's legacy global RNG limit).
    CUDA seeding is lazy, so this is safe to call before (or without) CUDA init.
    """
    seed = operator.index(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f'seed must be an integer in [0, 2**32), got {seed!r}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
