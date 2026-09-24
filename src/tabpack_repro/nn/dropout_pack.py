"""Per-member dropout and activations (a10)."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn


class DropoutPack(nn.Module):
    """Dropout with an individual rate per pack member.

    * Buffer ``p``: float32 ``(K,)``; accepts a float (broadcast) or a sequence of K.
    * Training mode: element-wise Bernoulli(1 - p_k) mask scaled by 1 / (1 - p_k);
      members with p_k == 0 must be exactly the identity (no RNG influence on them
      is required, but outputs must equal inputs bit-for-bit).
    * Eval mode: identity.
    * ``forward(x, member_idx=None)``: x ``(K', B, d)``, uses p[member_idx].
    """

    p: Tensor

    def __init__(self, p: float | Sequence[float], *, pack_size: int) -> None:
        super().__init__()
        pack_size = operator.index(pack_size)
        if pack_size < 1:
            raise ValueError(f'pack_size must be positive, got {pack_size}')

        p_tensor = torch.as_tensor(p, dtype=torch.float32).detach().clone()
        if p_tensor.ndim == 0:
            p_tensor = p_tensor.expand(pack_size).clone()
        elif p_tensor.ndim != 1 or len(p_tensor) != pack_size:
            raise ValueError(
                f'p must be a float or a sequence of pack_size={pack_size} floats,'
                f' got shape {tuple(p_tensor.shape)}'
            )
        # Validated after the float32 cast: e.g. 1 - 1e-9 rounds to 1.0 in float32,
        # which would make the 1 / (1 - p) scale infinite. NaN fails both checks.
        if not bool(((p_tensor >= 0.0) & (p_tensor < 1.0)).all()):
            raise ValueError(
                f'every p must satisfy 0 <= p < 1, got {p_tensor.tolist()}'
            )
        self.register_buffer('p', p_tensor)

    @property
    def pack_size(self) -> int:
        # Derived from the buffer (never cached): members can be removed in place.
        return self.p.shape[0]

    def extra_repr(self) -> str:
        return f'pack_size={self.pack_size}, p={self.p.tolist()}'

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        p = self.p if member_idx is None else self.p[member_idx]
        if x.ndim == 0 or x.shape[0] != p.shape[0]:
            raise ValueError(
                f'x must have the pack dimension first with size {p.shape[0]},'
                f' got shape {tuple(x.shape)}'
            )
        if not self.training:
            return x

        # The mask and the scale are computed in (at least) float32 and the result is
        # cast back, so that low-precision inputs (e.g. bf16 under autocast) keep
        # their dtype without rounding the 1 / (1 - p) scale to bf16. For members
        # with p == 0 the mask is Bernoulli(1) == 1 and the scale is exactly 1, so
        # x * 1 is the identity bit-for-bit (no data-dependent branch, no host sync).
        compute_dtype = torch.promote_types(x.dtype, torch.float32)
        keep = (1.0 - p.to(compute_dtype)).view(-1, *((1,) * (x.ndim - 1)))
        scale = torch.bernoulli(keep.expand(x.shape)).div_(keep)
        return (x * scale).to(x.dtype)


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
}


def make_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name ('ReLU', 'GELU', 'SiLU'); else raise."""
    factory = _ACTIVATIONS.get(name) if isinstance(name, str) else None
    if factory is None:
        raise ValueError(
            f'Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}'
        )
    return factory()
