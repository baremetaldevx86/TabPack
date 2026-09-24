"""LinearPack (a09): K independent linear layers applied with one batched matmul."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tabpack_repro.types import BATCH_DIM, PACK_DIM


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

    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self, in_features: int, out_features: int, *, pack_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        for name, value in (
            ('in_features', in_features),
            ('out_features', out_features),
            ('pack_size', pack_size),
        ):
            if value <= 0:
                raise ValueError(f'{name} must be positive, got {value}')
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(pack_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(pack_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return self.weight.shape[PACK_DIM]

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # nn.Linear: kaiming_uniform_(a=sqrt(5)) on (out, in) is U(-b, b) with
        # b = gain * sqrt(3 / fan_in) = sqrt(1/3) * sqrt(3 / in) = 1 / sqrt(in), and the
        # bias uses the same bound. All members are drawn in one call, which is the
        # same as K independent draws from that distribution.
        bound = self.in_features**-0.5
        # Draw in nn.Linear's per-member (out, in) order and store the transpose, so
        # that under the same RNG state the values match an (K, out, in) layout draw
        # (for K=1: nn.Linear itself; for any K: the official LinearPack).
        weight_out_in = torch.empty(
            self.pack_size,
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            device=self.weight.device,
        ).uniform_(-bound, bound)
        self.weight.copy_(weight_out_in.transpose(-2, -1))
        if self.bias is not None:
            self.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        weight = self.weight if member_idx is None else self.weight[member_idx]
        if x.ndim != 3 or x.shape[PACK_DIM] != weight.shape[PACK_DIM]:
            raise ValueError(
                f'Expected x of shape ({weight.shape[PACK_DIM]}, B, {self.in_features})'
                f', got {tuple(x.shape)}'
            )
        if x.shape[-1] != weight.shape[1]:
            raise ValueError(
                f'Expected x.shape[-1] == {weight.shape[1]}, got {x.shape[-1]}'
            )
        if self.bias is None:
            return torch.bmm(x, weight)
        bias = self.bias if member_idx is None else self.bias[member_idx]
        return torch.baddbmm(bias.unsqueeze(BATCH_DIM), x, weight)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'pack_size={self.pack_size}, bias={self.bias is not None}'
        )
