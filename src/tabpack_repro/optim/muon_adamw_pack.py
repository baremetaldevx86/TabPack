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

Layout notes (our LinearPack weight is (K, in, out), the official one (K, out, in)):
* Newton-Schulz commutes with transposition, NS(G^T) = NS(G)^T: every quintic
  iteration a X + (b A + c A^2) X with A = X X^T is the transpose of the one run on
  X^T (because (X X^T)^k X = X (X^T X)^k), and the Frobenius normalization is
  transpose-invariant. Moreover NS itself transposes tall matrices to the wide
  orientation, so for non-square weights both layouts orthogonalize the very same
  matrix (bit-identical); square weights differ only by bf16 rounding.
* The spectral scale is not symmetric: Keller Jordan's / the official
  ``max(1, size(-2) / size(-1)) ** 0.5`` is sqrt(max(1, out/in)) for (out, in);
  for our (K, in, out) it must read ``size(-1) / size(-2)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from tabpack_repro.optim.adamw_pack import PerMember, _PackOptimizer, adamw_update_
from tabpack_repro.optim.newton_schulz import zeropower_via_newtonschulz5

_MUON_STATE_KEY = 'muon_momentum_buffer'


def _check_unit_interval(value: Any, name: str) -> None:
    """Momentum-like coefficients: a float in [0, 1)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'{name} must be a float, got {value!r}')
    if not 0.0 <= value < 1.0:
        raise ValueError(f'{name} must be in [0, 1), got {value}')


def _bcast(value: float | Tensor, p: Tensor) -> float | Tensor:
    """View a (K,) tensor as (K, 1, ..., 1) in p.dtype; floats are returned as is."""
    if isinstance(value, Tensor):
        return value.to(dtype=p.dtype).view(-1, *((1,) * (p.ndim - 1)))
    return value


def _default_muon_scale(p: Tensor) -> float:
    """sqrt(max(1, out_features / in_features)) for a (K, in, out) weight."""
    in_features, out_features = p.shape[-2], p.shape[-1]
    return math.sqrt(max(1.0, out_features / in_features))


class MuonAdamWPack(_PackOptimizer):
    """A pack of K Muon+AdamW optimizers with per-member hyperparameters.

    Parameters are packs (member dim first). Group keys:

    * ``muon`` (bool, default False): Muon update for this group (params must be
      3-D (K, in, out) LinearPack weights), otherwise the AdamWPack update.
    * ``lr``, ``weight_decay``, ``muon_lr``, ``muon_scale``: a Python float (same
      for all members) or per-member values (sequence / tensor), stored as float32
      (K,) tensors on the params' device with separate storage per group.
      ``muon_lr=None`` makes Muon groups use ``lr`` (official semantics);
      ``muon_scale`` missing/None means sqrt(max(1, out_features / in_features)).
    * ``beta1``, ``beta2``, ``eps`` (AdamW), ``muon_momentum``, ``muon_nesterov``,
      ``muon_ns_steps`` (Muon): shared by all members.

    State: Muon params keep ``muon_momentum_buffer`` (like p) and no step. Other
    params use AdamWPack's layout: ``exp_avg``/``exp_avg_sq``, plus a per-param
    (K,) int64 ``step`` when shared_step=False; with shared_step=True a single
    optimizer-level step in ``state['__shared__']``, advanced once per ``step()``
    call in which at least one non-Muon param has a gradient.
    Unlike the official code, the Nesterov step does not overwrite ``p.grad``.
    """

    _per_member_keys = ('lr', 'weight_decay', 'muon_lr', 'muon_scale')
    # muon_lr=None means "use lr" (official semantics); muon_scale=None (or missing)
    # means the default spectral scale.
    _nullable_keys = ('muon_lr', 'muon_scale')

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
        for name, value in [
            ('muon_momentum', muon_momentum),
            ('beta1', beta1),
            ('beta2', beta2),
        ]:
            _check_unit_interval(value, name)
        if isinstance(eps, bool) or not (isinstance(eps, int | float) and eps > 0.0):
            raise ValueError(f'eps must be a positive float, got {eps!r}')
        if (
            isinstance(muon_ns_steps, bool)
            or not isinstance(muon_ns_steps, int)
            or muon_ns_steps < 0
        ):
            raise ValueError(
                f'muon_ns_steps must be a non-negative int, got {muon_ns_steps!r}'
            )
        super().__init__(
            params,
            {
                'lr': lr,
                'weight_decay': weight_decay,
                'muon_lr': muon_lr,
                'muon': False,
                'muon_momentum': float(muon_momentum),
                'muon_nesterov': bool(muon_nesterov),
                'muon_ns_steps': muon_ns_steps,
                'beta1': float(beta1),
                'beta2': float(beta2),
                'eps': float(eps),
            },
            pack_size=pack_size,
            shared_step=bool(shared_step),
        )

    def _normalize_group(self, group: dict[str, Any]) -> None:
        # Called by _PackOptimizer.add_param_group, which drops the group on error.
        if not isinstance(group['muon'], bool):
            raise TypeError(f"group['muon'] must be a bool, got {group['muon']!r}")
        if group['muon']:
            for p in group['params']:
                if p.ndim != 3:
                    raise ValueError(
                        'Muon groups only accept 3-D (K, in, out) weights, got shape '
                        f'{tuple(p.shape)}'
                    )
        super()._normalize_group(group)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        shared_step = (
            self._advance_shared_step()
            if self._shared_step and self._has_any_adamw_grad()
            else None
        )
        for group in self.param_groups:
            if group['muon']:
                self._step_muon(group)
            else:
                self._step_adamw(group, shared_step)
        return loss

    def _has_any_adamw_grad(self) -> bool:
        return any(
            p.grad is not None
            for group in self.param_groups
            if not group['muon']
            for p in group['params']
        )

    def _step_muon(self, group: dict[str, Any]) -> None:
        momentum = group['muon_momentum']
        nesterov = group['muon_nesterov']
        ns_steps = group['muon_ns_steps']

        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError('MuonAdamWPack does not support sparse gradients')
            if p.ndim != 3:
                raise ValueError(f'Muon expects (K, in, out), got {tuple(p.shape)}')

            lr_key = 'lr' if group['muon_lr'] is None else 'muon_lr'
            lr = self._group_value(group, lr_key, p.device)
            weight_decay = self._group_value(group, 'weight_decay', p.device)
            scale = (
                _default_muon_scale(p)
                if group.get('muon_scale') is None
                else self._group_value(group, 'muon_scale', p.device)
            )

            state = self.state[p]
            if len(state) == 0:
                state[_MUON_STATE_KEY] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            buf: Tensor = state[_MUON_STATE_KEY]

            # Official _make_weight_decay_multiplier: skipped when lr or wd is 0.0.
            if not (
                (isinstance(lr, float) and lr == 0.0)
                or (isinstance(weight_decay, float) and weight_decay == 0.0)
            ):
                p.mul_(1 - _bcast(lr, p) * _bcast(weight_decay, p))

            buf.lerp_(grad, _bcast(1 - momentum, p))
            # Out of place: the official code does grad.lerp_ and clobbers p.grad.
            update = grad.lerp(buf, _bcast(momentum, p)) if nesterov else buf
            # Batched over the pack dim: each member is orthogonalized on its own
            # (see the module docstring for why the layout does not matter here).
            update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(p.dtype)
            p.sub_(update.mul_(_bcast(lr, p) * _bcast(scale, p)))

    def _step_adamw(self, group: dict[str, Any], shared_step: int | None) -> None:
        # State handling identical to AdamWPack.step; the math is adamw_update_.
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError('MuonAdamWPack does not support sparse gradients')

            state = self.state[p]
            if not state:
                state['exp_avg'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state['exp_avg_sq'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                self._init_param_state(p, state)
            step = (
                shared_step
                if shared_step is not None
                else self._next_param_step(p, state)
            )
            adamw_update_(
                p,
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                lr=self._group_value(group, 'lr', p.device),
                weight_decay=self._group_value(group, 'weight_decay', p.device),
                beta1=group['beta1'],
                beta2=group['beta2'],
                eps=group['eps'],
                step=step,
            )
