"""AdamWPack (a15): AdamW with per-member lr / weight_decay.

Per-member hyperparameters are stored in each param group as float32 tensors of
shape (K,) on the parameters' device (a Python float is also accepted and means
"same for all members"). Every param in a group has the pack dim first, and the
(K,) tensors are broadcast over the trailing dims.

The update must match ``torch.optim.AdamW`` (foreach=False, amsgrad=False,
maximize=False) member-by-member:
    p <- p * (1 - lr * wd)
    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * g^2
    p <- p - (lr / (1 - beta1^t)) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
With shared_step=True, `t` is a single Python int per optimizer (all members step
together); otherwise a per-param step counter as in PyTorch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor

PerMember = float | Sequence[float] | Tensor


def adamw_update_(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    lr: float | Tensor,
    weight_decay: float | Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    step: int | Tensor,
) -> None:
    """One in-place AdamW update of a packed parameter ``p`` (pack dim first).

    lr / weight_decay: Python floats or (K,) tensors (broadcast over trailing dims).
    step: the 1-based step count used for bias correction; an int, or a (K,) tensor.
    Shared by AdamWPack and MuonAdamWPack (non-Muon groups).
    """
    raise NotImplementedError


class AdamWPack(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: PerMember,
        weight_decay: PerMember = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        pack_size: int,
        shared_step: bool = True,
    ) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        raise NotImplementedError
