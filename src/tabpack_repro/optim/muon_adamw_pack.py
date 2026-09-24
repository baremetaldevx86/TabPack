"""MuonAdamWPack (a16): Muon for hidden-layer weights, AdamW for the rest.

Param groups with ``group['muon'] is True`` get the Muon update (official
project/optim.py::MuonAdamWPack._step_muon), for each weight p of shape (K, a, b):
    p <- p * (1 - muon_lr * wd)                       (decoupled weight decay)
    buf <- lerp(buf, g, 1 - momentum)                 (momentum buffer)
    u <- lerp(g, buf, momentum) if nesterov else buf
    u <- zeropower_via_newtonschulz5(u, ns_steps)     (per member)
    u <- u * muon_scale[k]                            (group['muon_scale'], (K,))
    p <- p - muon_lr * u
``muon_scale`` defaults to sqrt(max(1, out_features / in_features)) when the group
has no 'muon_scale'. All other groups get exactly the AdamWPack update.
Per-member hyperparameters: lr, weight_decay, muon_lr ((K,) tensors or floats).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from tabpack_repro.optim.adamw_pack import PerMember


class MuonAdamWPack(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: PerMember,
        weight_decay: PerMember,
        muon_lr: PerMember,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
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
