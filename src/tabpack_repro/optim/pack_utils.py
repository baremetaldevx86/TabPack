"""Parameter groups and optimizer member removal (a17)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tabpack_repro.nn.model_pack import ModelPack


def make_param_groups(model: ModelPack, *, muon: bool) -> list[dict[str, Any]]:
    """Build optimizer param groups for a ModelPack.

    * muon=True: group ``{'params': [block.linear.weight], 'muon': True,
      'muon_scale': (K,) tensor}`` for EVERY backbone block (one group per block,
      muon_scale = sqrt(max(1, out/in)) as a float32 (K,) tensor), like the official
      code.
    * Biases (every parameter with ndim <= 2 in a pack, i.e. (K, d)) go to a group
      with ``'weight_decay': 0.0`` (a float overrides the per-member tensor).
    * All remaining weights (head weight; backbone weights when muon=False) go to
      the default group (per-member weight decay applies).
    * Empty groups are omitted. Every parameter appears exactly once.
    """
    raise NotImplementedError


def optimizer_select_(optimizer: torch.optim.Optimizer, keep_idx: Tensor) -> None:
    """Keep only members keep_idx in the optimizer, in place (after pack_select_).

    Slices along dim 0: every tensor in ``optimizer.state[p]`` whose dim 0 is the old
    pack size (exp_avg, exp_avg_sq, momentum buffers, per-param step tensors of shape
    (K,)); and every (K,) tensor-valued hyperparameter in each param group (lr,
    weight_decay, muon_lr, muon_scale). Scalars (floats, int steps) are untouched.
    """
    raise NotImplementedError
