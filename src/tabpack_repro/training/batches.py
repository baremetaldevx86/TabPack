"""Training batches (a18): every member gets its own random permutation."""

from __future__ import annotations

import torch
from torch import Tensor


def generate_member_batches(
    *, train_size: int, batch_size: int, pack_size: int, generator: torch.Generator
) -> list[Tensor]:
    """Batches for one epoch; each is int64 ``(pack_size, b)``.

    Official semantics: ``torch.rand((K, N), generator=g, device=g.device)
    .argsort(dim=1).split(batch_size, dim=1)`` (last batch may be smaller).
    """
    raise NotImplementedError


def epoch_size(train_size: int, batch_size: int) -> int:
    """Number of batches per epoch: ceil(train_size / batch_size)."""
    raise NotImplementedError
