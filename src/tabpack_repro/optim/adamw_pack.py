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

Implementation notes
--------------------
* Hyperparameter normalization (constructor defaults and every param group, also
  groups added later with ``add_param_group``): a Python number stays a float
  (same value for all members; e.g. a ``'weight_decay': 0.0`` bias group); a
  sequence / array / 1-D tensor becomes a NEW float32 (K,) tensor (own storage per
  group) on the device of the group's params. If the params are moved afterwards,
  ``step()`` moves the (K,) tensors lazily and stores them back in the group.
* shared_step=True: the step count lives at optimizer level in
  ``self.state['__shared__'] = {'step': t}`` (a string key, which
  ``torch.optim.Optimizer.state_dict``/``load_state_dict`` carry over as is). It is
  incremented once per ``step()`` call that updates at least one parameter, and it
  is untouched by member removal (it is a plain int). Per-param states then hold
  only ``exp_avg`` / ``exp_avg_sq``.
* shared_step=False: every param state has ``'step'``, an int64 (K,) tensor (sliced
  together with the moments on member removal), incremented only when the param
  has a gradient -- exactly PyTorch's per-param semantics.
* A param whose ``.grad`` is None is skipped (no update, no weight decay, no state).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor

PerMember = float | Sequence[float] | Tensor

# Key of the optimizer-level (not per-param) state; its value is {'step': int}.
_SHARED_STATE_KEY = '__shared__'


def _to_per_member(value: Any, pack_size: int, name: str) -> float | Tensor:
    """Normalize a per-member hyperparameter.

    Returns a float (same value for all members) or a new float32 (pack_size,)
    tensor with its own storage, on the device of the input (CPU for sequences).
    All values must be finite and non-negative.
    """
    if value is None or isinstance(value, bool | str | bytes):
        raise TypeError(
            f'{name} must be a float or a sequence of floats, got {type(value)}'
        )
    if isinstance(value, int | float):
        value = float(value)
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f'{name} must be finite and non-negative, got {value}')
        return value
    if isinstance(value, Tensor):
        value = value.detach()
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.ndim == 0:
        return _to_per_member(t.item(), pack_size, name)
    if t.shape != (pack_size,):
        raise ValueError(
            f'{name} must be a float or have shape ({pack_size},) (one value per'
            f' pack member), got shape {tuple(t.shape)}'
        )
    if not bool((torch.isfinite(t) & (t >= 0.0)).all()):
        raise ValueError(f'{name} must be finite and non-negative, got {t.tolist()}')
    return t.clone()


def _check_beta(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f'{name} must be in [0, 1), got {value}')


def _scalar_dtype(p: Tensor) -> torch.dtype:
    # (K,)-sized hyperparameter math is done in float64 like PyTorch's Python-float
    # math (MPS has no float64).
    return torch.float32 if p.device.type == 'mps' else torch.float64


def _member_values(value: float | Tensor, p: Tensor, name: str) -> float | Tensor:
    """A float, or a (K,) tensor checked against p and converted for scalar math."""
    if not isinstance(value, Tensor):
        return float(value)
    if value.shape != (p.shape[0],):
        raise ValueError(
            f'{name}: expected a float or a ({p.shape[0]},) tensor for a parameter'
            f' of shape {tuple(p.shape)}, got shape {tuple(value.shape)}'
        )
    return value.to(device=p.device, dtype=_scalar_dtype(p))


def _column(value: float | Tensor, p: Tensor) -> float | Tensor:
    """(K,) -> (K, 1, ..., 1) in p.dtype, broadcastable over p; floats unchanged."""
    if isinstance(value, Tensor):
        return value.to(p.dtype).view(-1, *(1,) * (p.ndim - 1))
    return value


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

    ``exp_avg`` and ``exp_avg_sq`` are updated in place; ``step`` is NOT incremented
    here (the caller owns the counter). With float lr / weight_decay and an int step
    the arithmetic is literally that of ``torch.optim.AdamW(foreach=False)``.
    """
    # Validate everything before touching p or the moments.
    lr = _member_values(lr, p, 'lr')
    weight_decay = _member_values(weight_decay, p, 'weight_decay')
    if isinstance(step, Tensor) and step.ndim == 0:
        step = int(step.item())
    if isinstance(step, Tensor):
        step = _member_values(step, p, 'step')
    elif step < 1:
        raise ValueError(f'step must be >= 1 (1-based), got {step}')

    # Decoupled weight decay (skipped only when it is statically a no-op).
    if not (
        (isinstance(lr, float) and lr == 0.0)
        or (isinstance(weight_decay, float) and weight_decay == 0.0)
    ):
        p.mul_(_column(1 - lr * weight_decay, p))

    # Biased first and second moment estimates (same kernels as PyTorch).
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Bias correction (Python floats for an int step, like PyTorch).
    bias_correction1 = 1 - beta1**step
    if isinstance(step, Tensor):
        bias_correction2_sqrt = (1 - beta2**step).sqrt()
    else:
        bias_correction2_sqrt = (1 - beta2**step) ** 0.5
    step_size = lr / bias_correction1

    denom = (exp_avg_sq.sqrt() / _column(bias_correction2_sqrt, p)).add_(eps)
    if isinstance(step_size, float):
        p.addcdiv_(exp_avg, denom, value=-step_size)
    else:
        p.addcdiv_(exp_avg * _column(-step_size, p), denom)


class _PackOptimizer(torch.optim.Optimizer):
    """Machinery shared by optimizers whose hyperparameters differ per pack member.

    * Hyperparameters named in ``_per_member_keys`` are normalized in every param
      group by `_to_per_member` (float, or a float32 (K,) tensor on the params'
      device, with separate storage per group). Keys also listed in
      ``_nullable_keys`` may be None (kept as None, e.g. "fall back to lr").
    * ``shared_step=True``: one optimizer-level int step (`_advance_shared_step`).
    * ``pack_size`` and ``shared_step`` survive pickling / ``copy.deepcopy``.
    """

    _per_member_keys: tuple[str, ...] = ()
    _nullable_keys: tuple[str, ...] = ()

    def __init__(
        self,
        params: Iterable[Any],
        defaults: dict[str, Any],
        *,
        pack_size: int,
        shared_step: bool,
    ) -> None:
        if pack_size < 1:
            raise ValueError(f'pack_size must be positive, got {pack_size}')
        # Set before super().__init__, which calls add_param_group.
        self._pack_size = pack_size
        self._shared_step = shared_step
        defaults = dict(defaults)
        for key in self._per_member_keys:
            if key in defaults:
                defaults[key] = self._normalize(defaults[key], pack_size, key)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if p.shape[0] != pack_size:
                    raise ValueError(
                        f'Every parameter must have the pack dim first with size'
                        f' pack_size={pack_size}, got shape {tuple(p.shape)}'
                    )

    def _normalize(self, value: Any, pack_size: int, key: str) -> float | Tensor | None:
        if value is None and key in self._nullable_keys:
            return None
        return _to_per_member(value, pack_size, key)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        try:
            self._normalize_group(self.param_groups[-1])
        except Exception:
            self.param_groups.pop()  # leave the optimizer as it was
            raise

    def _normalize_group(self, group: dict[str, Any]) -> None:
        params = group['params']
        if any(p.ndim == 0 for p in params):
            raise ValueError('Packed parameters must have the pack dim first')
        sizes = {p.shape[0] for p in params}
        if len(sizes) > 1:
            raise ValueError(f'Params of one group have different pack sizes: {sizes}')
        pack_size = sizes.pop() if sizes else self._pack_size
        device = params[0].device if params else None
        values = {
            key: self._normalize(group[key], pack_size, key)
            for key in self._per_member_keys
            if key in group
        }
        for key, value in values.items():
            if isinstance(value, Tensor) and device is not None:
                value = value.to(device)
            group[key] = value

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state['_pack_size'] = self._pack_size
        state['_shared_step'] = self._shared_step
        return state

    @staticmethod
    def _group_value(group: dict[str, Any], key: str, device: torch.device) -> Any:
        """group[key], moving a tensor to `device` (and storing it back) if needed."""
        value = group[key]
        if isinstance(value, Tensor) and value.device != device:
            value = group[key] = value.to(device)
        return value

    def _has_any_grad(self) -> bool:
        return any(
            p.grad is not None for group in self.param_groups for p in group['params']
        )

    def _advance_shared_step(self) -> int:
        """Increment and return the optimizer-level step (shared_step=True)."""
        shared = self.state.get(_SHARED_STATE_KEY)
        step = (shared['step'] if shared else 0) + 1
        # A new dict every time: never mutate a dict that a previously returned
        # state_dict() (or a loaded one) may still reference.
        self.state[_SHARED_STATE_KEY] = {'step': step}
        return step

    def _init_param_state(self, p: Tensor, state: dict[str, Any]) -> None:
        """Lazily create the (K,) int64 per-param step (shared_step=False only)."""
        if not self._shared_step:
            state['step'] = torch.zeros(p.shape[0], dtype=torch.int64, device=p.device)

    @staticmethod
    def _next_param_step(p: Tensor, state: dict[str, Any]) -> Tensor:
        """Increment and return the per-param (K,) int64 step (shared_step=False)."""
        step = state.get('step')
        if step is None:
            raise RuntimeError(
                'The optimizer state has no per-param step; was it saved with'
                ' shared_step=True and loaded into an optimizer with'
                ' shared_step=False?'
            )
        if step.device != p.device:
            step = step.to(p.device)
        step += 1
        state['step'] = step
        return step


class AdamWPack(_PackOptimizer):
    _per_member_keys = ('lr', 'weight_decay')

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
        _check_beta(beta1, 'beta1')
        _check_beta(beta2, 'beta2')
        if not eps >= 0.0:
            raise ValueError(f'eps must be non-negative, got {eps}')
        super().__init__(
            params,
            {
                'lr': lr,
                'weight_decay': weight_decay,
                'beta1': beta1,
                'beta2': beta2,
                'eps': eps,
            },
            pack_size=pack_size,
            shared_step=shared_step,
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        shared_step = (
            self._advance_shared_step()
            if self._shared_step and self._has_any_grad()
            else None
        )
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError('AdamWPack does not support sparse gradients')

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
        return loss
