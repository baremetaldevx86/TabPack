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

import math
import numbers
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from tabpack_repro.optim.adamw_pack import PerMember, adamw_update_
from tabpack_repro.optim.newton_schulz import zeropower_via_newtonschulz5

# Group keys whose value may differ per pack member: a Python float (shared by all
# members) or a float32 (K,) tensor on the parameters' device.
_PER_MEMBER_KEYS = ('lr', 'weight_decay', 'muon_lr', 'muon_scale')
# Group keys that may be None: muon_lr=None means "use lr" (official semantics) and
# muon_scale=None (or missing) means "use the default spectral scale".
_NULLABLE_KEYS = ('muon_lr', 'muon_scale')
_MUON_STATE_KEY = 'muon_momentum_buffer'


def _as_per_member(
    value: Any, *, name: str, pack_size: int, device: torch.device | None
) -> float | Tensor | None:
    """Normalize a hyperparameter to a Python float or a fresh float32 (K,) tensor.

    Rejects negative (and NaN) values. None is passed through (callers decide
    whether it is allowed). A 0-dim tensor is treated as a shared float.
    """
    if value is None:
        return None
    if isinstance(value, bool | str | bytes):
        raise TypeError(f'{name} must be a float, a sequence or a tensor: {value!r}')
    if isinstance(value, numbers.Real):
        value = float(value)
        if not value >= 0.0:
            raise ValueError(f'{name} must be non-negative, got {value}')
        return value
    if isinstance(value, Tensor):
        # clone() so that every param group owns separate storage.
        tensor = value.detach().to(device=device, dtype=torch.float32).clone()
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError(
                f'{name} must be a float, a sequence or a tensor, got {value!r}'
            ) from err
    if tensor.ndim == 0:
        return _as_per_member(
            tensor.item(), name=name, pack_size=pack_size, device=device
        )
    if tensor.shape != (pack_size,):
        raise ValueError(
            f'{name} must have shape ({pack_size},), got {tuple(tensor.shape)}'
        )
    if not bool((tensor >= 0.0).all()):
        raise ValueError(f'{name} must be non-negative, got {tensor.tolist()}')
    return tensor


def _check_unit_interval(value: Any, name: str) -> None:
    """Momentum-like coefficients: a float in [0, 1)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'{name} must be a float, got {value!r}')
    if not 0.0 <= value < 1.0:
        raise ValueError(f'{name} must be in [0, 1), got {value}')


def _bcast(value: float | Tensor, p: Tensor) -> float | Tensor:
    """View a (K,) tensor as (K, 1, ..., 1) so it broadcasts over p (pack dim first)."""
    if isinstance(value, Tensor):
        return value.to(dtype=p.dtype).view(-1, *((1,) * (p.ndim - 1)))
    return value


def _default_muon_scale(p: Tensor) -> float:
    """sqrt(max(1, out_features / in_features)) for our (K, in, out) weight layout.

    The official LinearPack stores (K, out, in) and computes
    ``max(1, size(-2) / size(-1)) ** 0.5``; with our transposed layout, out_features
    is ``size(-1)`` and in_features is ``size(-2)``.
    """
    in_features, out_features = p.shape[-2], p.shape[-1]
    return math.sqrt(max(1.0, out_features / in_features))


class MuonAdamWPack(torch.optim.Optimizer):
    """A pack of K Muon+AdamW optimizers with per-member hyperparameters.

    Parameters are packs (member dim first). Group keys:

    * ``muon`` (bool, default False): Muon update for this group (params must be
      3-D (K, in, out) LinearPack weights), otherwise the AdamWPack update.
    * ``lr``, ``weight_decay``, ``muon_lr``: Python float or (K,) (float / sequence /
      tensor; stored as float32 (K,) tensors on the params' device, one copy per
      group). ``muon_lr=None`` makes Muon groups use ``lr`` (official semantics).
    * ``muon_scale`` (Muon groups only): float or (K,); missing/None means
      sqrt(max(1, out_features / in_features)).
    * ``beta1``, ``beta2``, ``eps`` (AdamW), ``muon_momentum``, ``muon_nesterov``,
      ``muon_ns_steps`` (Muon): shared by all members.

    State (per param, nothing at the optimizer level): Muon groups keep
    ``muon_momentum_buffer`` (like p); other groups keep ``step`` (a Python int with
    shared_step=True, else a (K,) int64 tensor), ``exp_avg`` and ``exp_avg_sq``.
    Unlike the official code, the Nesterov step does not overwrite ``p.grad``.
    """

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
        if isinstance(pack_size, bool) or not isinstance(pack_size, int):
            raise TypeError(f'pack_size must be an int, got {pack_size!r}')
        if pack_size <= 0:
            raise ValueError(f'pack_size must be positive, got {pack_size}')
        for name, value in [
            ('muon_momentum', muon_momentum),
            ('beta1', beta1),
            ('beta2', beta2),
        ]:
            _check_unit_interval(value, name)
        if not (isinstance(eps, int | float) and eps > 0.0):
            raise ValueError(f'eps must be a positive float, got {eps!r}')
        if (
            isinstance(muon_ns_steps, bool)
            or not isinstance(muon_ns_steps, int)
            or muon_ns_steps < 0
        ):
            raise ValueError(
                f'muon_ns_steps must be a non-negative int, got {muon_ns_steps!r}'
            )
        if lr is None:
            raise ValueError('lr must not be None')
        if weight_decay is None:
            raise ValueError('weight_decay must not be None')

        self._pack_size = pack_size
        self._shared_step = bool(shared_step)

        # Per-member defaults live on the CPU as float32 (K,) tensors (so that
        # optimizer_select_ can slice them); add_param_group copies them per group
        # onto the parameters' device.
        per_member_defaults = {
            name: _as_per_member(value, name=name, pack_size=pack_size, device=None)
            for name, value in [
                ('lr', lr),
                ('weight_decay', weight_decay),
                ('muon_lr', muon_lr),
            ]
        }
        defaults: dict[str, Any] = {
            **per_member_defaults,
            'muon': False,
            'muon_momentum': float(muon_momentum),
            'muon_nesterov': bool(muon_nesterov),
            'muon_ns_steps': muon_ns_steps,
            'beta1': float(beta1),
            'beta2': float(beta2),
            'eps': float(eps),
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if p.shape[0] != pack_size:
                    raise ValueError(
                        f'Every parameter must have pack_size={pack_size} as its '
                        f'first dimension, got shape {tuple(p.shape)}'
                    )

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        params: list[Tensor] = group['params']

        if not isinstance(group['muon'], bool):
            raise TypeError(f"group['muon'] must be a bool, got {group['muon']!r}")
        for p in params:
            if p.ndim < 1:
                raise ValueError('Parameters must have a leading pack dimension')
            if group['muon'] and p.ndim != 3:
                raise ValueError(
                    'Muon groups only accept 3-D (K, in, out) weights, got shape '
                    f'{tuple(p.shape)}'
                )
        # The pack size is derived from the parameters (members may have been
        # removed since construction); an empty group falls back to pack_size.
        pack_size = params[0].shape[0] if params else self._pack_size
        if any(p.shape[0] != pack_size for p in params):
            raise ValueError(
                'All parameters of a group must share the pack size, got shapes '
                f'{[tuple(p.shape) for p in params]}'
            )
        device = params[0].device if params else None

        for key in _PER_MEMBER_KEYS:
            if key not in group:
                continue
            value = _as_per_member(
                group[key], name=key, pack_size=pack_size, device=device
            )
            if value is None and key not in _NULLABLE_KEYS:
                raise ValueError(f'{key} must not be None')
            group[key] = value

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['muon']:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict[str, Any]) -> None:
        lr = group['muon_lr']
        if lr is None:
            lr = group['lr']
        weight_decay = group['weight_decay']
        momentum = group['muon_momentum']
        nesterov = group['muon_nesterov']
        ns_steps = group['muon_ns_steps']
        muon_scale = group.get('muon_scale')

        # Official _make_weight_decay_multiplier: skipped when lr or wd is a float 0.
        skip_weight_decay = (isinstance(lr, float) and lr == 0.0) or (
            isinstance(weight_decay, float) and weight_decay == 0.0
        )
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('MuonAdamWPack does not support sparse gradients')
            if p.ndim != 3:
                raise ValueError(f'Muon expects (K, in, out), got {tuple(p.shape)}')

            state = self.state[p]
            if len(state) == 0:
                state[_MUON_STATE_KEY] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            buf: Tensor = state[_MUON_STATE_KEY]

            if not skip_weight_decay:
                p.mul_(1 - _bcast(lr, p) * _bcast(weight_decay, p))

            buf.lerp_(grad, _bcast(1 - momentum, p))
            # Out of place: the official code does grad.lerp_ and so clobbers p.grad.
            update = grad.lerp(buf, _bcast(momentum, p)) if nesterov else buf
            # Newton-Schulz is layout-agnostic: NS(G^T) = NS(G)^T, since each quintic
            # iteration a X + (b X X^T + c (X X^T)^2) X equals the transpose of the
            # one on X^T, and the Frobenius normalization is transpose-invariant.
            # Batched over the pack dim, so each member is orthogonalized alone.
            update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(p.dtype)

            scale = _default_muon_scale(p) if muon_scale is None else muon_scale
            p.sub_(update.mul_(_bcast(lr, p) * _bcast(scale, p)))

    def _step_adamw(self, group: dict[str, Any]) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('MuonAdamWPack does not support sparse gradients')

            state = self.state[p]
            if len(state) == 0:
                state['step'] = (
                    0
                    if self._shared_step
                    else torch.zeros(p.shape[0], dtype=torch.int64, device=p.device)
                )
                state['exp_avg'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state['exp_avg_sq'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            state['step'] += 1

            adamw_update_(
                p,
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                beta1=group['beta1'],
                beta2=group['beta2'],
                eps=group['eps'],
                step=state['step'],
            )
