"""Parameter groups and optimizer member removal (a17)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

from tabpack_repro.nn.model_pack import ModelPack


def _make_muon_scale(weight: Tensor) -> Tensor:
    """``sqrt(max(1, out / in))`` for a LinearPack weight ``(K, in, out)``, as (K,).

    Computed in float32 (divide, clamp, sqrt) like the official ``_make_muon_scale``,
    which works on the float32 ``out_features`` / ``in_features`` buffers.
    """
    if weight.ndim != 3:
        raise ValueError(
            f'A Muon weight must have shape (K, in, out), got {tuple(weight.shape)}'
        )
    pack_size, in_features, out_features = weight.shape
    kwargs: dict[str, Any] = {'dtype': torch.float32, 'device': weight.device}
    out_f = torch.full((pack_size,), float(out_features), **kwargs)
    in_f = torch.full((pack_size,), float(in_features), **kwargs)
    return (out_f / in_f).clamp_(min=1.0).sqrt_()


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
    muon_groups: list[dict[str, Any]] = []
    if muon:
        for block in model.backbone.blocks:
            weight = block.linear.weight
            muon_groups.append(
                {
                    'params': [weight],
                    'muon': True,
                    'muon_scale': _make_muon_scale(weight),
                }
            )
    muon_ids = {id(p) for group in muon_groups for p in group['params']}
    if len(muon_ids) != len(muon_groups):
        raise ValueError('Backbone blocks must not share their linear weights')

    # Group order follows the official make_parameter_groups: default, zero weight
    # decay, custom (Muon). Within a group, parameters follow model.parameters()
    # (deterministic, unlike the official frozenset), which also deduplicates
    # shared parameters.
    default_params: list[Tensor] = []
    zero_wd_params: list[Tensor] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
        (zero_wd_params if p.ndim <= 2 else default_params).append(p)

    groups: list[dict[str, Any]] = []
    if default_params:
        groups.append({'params': default_params})
    if zero_wd_params:
        groups.append({'params': zero_wd_params, 'weight_decay': 0.0})
    groups.extend(muon_groups)
    return groups


def _is_packed(value: Any) -> bool:
    return isinstance(value, Tensor) and value.ndim >= 1


def _iter_state_values(optimizer: torch.optim.Optimizer) -> Iterator[Any]:
    """All values of ``optimizer.state``: per-param entries and non-param entries."""
    for value in optimizer.state.values():
        if isinstance(value, dict):
            yield from value.values()
        else:
            yield value


def _iter_hyperparameters(optimizer: torch.optim.Optimizer) -> Iterator[Any]:
    for group in optimizer.param_groups:
        for key, value in group.items():
            if key != 'params':
                yield value
    yield from optimizer.defaults.values()


def _infer_old_pack_size(optimizer: torch.optim.Optimizer) -> int | None:
    """The pack size before removal, read from the optimizer's per-member tensors.

    Parameters already have the new pack size (pack_select_ runs first), so the old
    one is the dim 0 of the not-yet-sliced tensors: every tensor with ndim >= 1 in
    ``optimizer.state``, the param groups and the defaults. They must all agree.
    None means that there is nothing per-member to slice.
    """
    sizes = {
        value.shape[0]
        for value in (*_iter_state_values(optimizer), *_iter_hyperparameters(optimizer))
        if _is_packed(value)
    }
    if len(sizes) > 1:
        raise ValueError(
            'The per-member tensors of the optimizer disagree on the pack size:'
            f' {sorted(sizes)}. Every tensor with ndim >= 1 in the optimizer state and'
            ' param groups must have the pack dimension first.'
        )
    return sizes.pop() if sizes else None


@torch.no_grad()
def optimizer_select_(optimizer: torch.optim.Optimizer, keep_idx: Tensor) -> None:
    """Keep only members keep_idx in the optimizer, in place (after pack_select_).

    Slices along dim 0: every tensor in ``optimizer.state[p]`` whose dim 0 is the old
    pack size (exp_avg, exp_avg_sq, momentum buffers, per-param step tensors of shape
    (K,)); and every (K,) tensor-valued hyperparameter in each param group (lr,
    weight_decay, muon_lr, muon_scale). Scalars (floats, int steps) are untouched.
    """
    # Also handled (beyond the minimum above): tensors stored in optimizer.state
    # under a non-param key (e.g. an optimizer-level step) and tensors in
    # optimizer.defaults, so that a later add_param_group stays consistent.
    keep_idx = torch.as_tensor(keep_idx)
    if (
        keep_idx.ndim != 1
        or keep_idx.dtype == torch.bool
        or keep_idx.is_floating_point()
    ):
        raise ValueError(
            'keep_idx must be a 1-D tensor of integer indices,'
            f' got dtype={keep_idx.dtype}, shape={tuple(keep_idx.shape)}'
        )
    keep_idx = keep_idx.long()
    new_size = keep_idx.numel()

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.ndim == 0 or p.shape[0] != new_size:
                raise ValueError(
                    f'A parameter of shape {tuple(p.shape)} does not have pack size'
                    f' {new_size} = len(keep_idx). Call pack_select_(model, keep_idx)'
                    ' before optimizer_select_(optimizer, keep_idx).'
                )

    old_size = _infer_old_pack_size(optimizer)
    if old_size is None:
        return
    if new_size > 0 and not (
        0 <= int(keep_idx.min()) and int(keep_idx.max()) < old_size
    ):
        raise IndexError(
            f'keep_idx must be in [0, {old_size}), got {keep_idx.tolist()}'
        )

    devices_idx: dict[torch.device, Tensor] = {}

    def select(value: Any) -> Any:
        if not _is_packed(value) or value.shape[0] != old_size:
            return value
        idx = devices_idx.get(value.device)
        if idx is None:
            idx = devices_idx[value.device] = keep_idx.to(value.device)
        # Advanced indexing returns a new tensor with its own storage, so tensors
        # that were shared between groups do not stay aliased by accident.
        return value[idx]

    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if key != 'params':
                group[key] = select(value)
    for key, value in list(optimizer.defaults.items()):
        optimizer.defaults[key] = select(value)
    for key, value in list(optimizer.state.items()):
        if isinstance(value, dict):
            for state_key, state_value in list(value.items()):
                value[state_key] = select(state_value)
        else:
            optimizer.state[key] = select(value)
