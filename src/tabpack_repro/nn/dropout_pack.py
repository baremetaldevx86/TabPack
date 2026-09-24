"""Per-member dropout and activations (a10)."""

from __future__ import annotations

from collections.abc import Sequence

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

    def __init__(self, p: float | Sequence[float], *, pack_size: int) -> None:
        super().__init__()
        raise NotImplementedError

    @property
    def pack_size(self) -> int:
        raise NotImplementedError

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        raise NotImplementedError


def make_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name ('ReLU', 'GELU', 'SiLU'); else raise."""
    raise NotImplementedError
