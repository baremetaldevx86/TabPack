"""Training batches (a18): every member gets its own random permutation."""

from __future__ import annotations

import torch
from torch import Tensor


def _check_positive_int(name: str, value: object) -> int:
    # ``bool`` is a subclass of ``int`` but is never a meaningful size here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an int, got {type(value).__name__}')
    if value <= 0:
        raise ValueError(f'{name} must be positive, got {value}')
    return value


def generate_member_batches(
    *, train_size: int, batch_size: int, pack_size: int, generator: torch.Generator
) -> list[Tensor]:
    """Batches for one epoch; each is int64 ``(pack_size, b)``.

    Official semantics: ``torch.rand((K, N), generator=g, device=g.device)
    .argsort(dim=1).split(batch_size, dim=1)`` (last batch may be smaller).
    """
    _check_positive_int('train_size', train_size)
    _check_positive_int('batch_size', batch_size)
    _check_positive_int('pack_size', pack_size)
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            f'generator must be a torch.Generator, got {type(generator).__name__}'
        )

    # One uniform draw per (member, object); sorting each row yields an independent
    # permutation of range(train_size) per member. The RNG consumption (a single
    # ``torch.rand`` of shape (K, N) on the generator's device) matches the official
    # code, so a seeded generator produces exactly the official batch sequence.
    keys = torch.rand(
        (pack_size, train_size), generator=generator, device=generator.device
    )
    permutations = keys.argsort(dim=1)
    return list(permutations.split(batch_size, dim=1))


def epoch_size(train_size: int, batch_size: int) -> int:
    """Number of batches per epoch: ceil(train_size / batch_size)."""
    _check_positive_int('train_size', train_size)
    _check_positive_int('batch_size', batch_size)
    return -(-train_size // batch_size)
