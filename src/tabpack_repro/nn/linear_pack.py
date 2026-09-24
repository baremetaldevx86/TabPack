"""LinearPack (a09): K independent linear layers applied with one batched matmul."""

from __future__ import annotations

from torch import Tensor, nn


class LinearPack(nn.Module):
    """K independent ``nn.Linear(in_features, out_features)`` layers.

    * ``weight``: Parameter ``(K, in_features, out_features)`` (note: in x out, so the
      forward pass is ``x @ weight``; this is the transpose of nn.Linear's layout).
    * ``bias``: Parameter ``(K, out_features)`` or None.
    * Initialization: for each member independently, identical in distribution to
      ``nn.Linear.reset_parameters`` (kaiming_uniform_(a=sqrt(5)) on the (out, in)
      matrix, bias ~ U(-1/sqrt(in), 1/sqrt(in))).
    * ``forward(x, member_idx=None)``: x ``(K', B, in)`` -> ``(K', B, out)`` where
      K' = K if member_idx is None else len(member_idx), using weight[member_idx].
    * ``pack_size`` is a property derived from ``weight.shape[0]`` (never cached),
      because members can be removed in place by pack_ops.pack_select_.
    """

    def __init__(
        self, in_features: int, out_features: int, *, pack_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        raise NotImplementedError

    @property
    def pack_size(self) -> int:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        raise NotImplementedError
