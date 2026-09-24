"""MLPBackbonePack (a11): K MLPs with (possibly) different depths and dropout rates."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from tabpack_repro.nn.dropout_pack import DropoutPack
from tabpack_repro.nn.linear_pack import LinearPack


class MLPBlockPack(nn.Module):
    """Linear -> activation -> dropout.

    Attributes: ``linear``, ``activation``, ``dropout``.
    """

    linear: LinearPack
    dropout: DropoutPack

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float | Sequence[float],
        activation: str,
        pack_size: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class MLPBackbonePack(nn.Module):
    """``max(n_blocks)`` blocks; member k only applies its first ``n_blocks[k]`` blocks.

    * Block 0 maps d_in -> d_block, the others d_block -> d_block.
    * ``blocks``: nn.ModuleList[MLPBlockPack] (the optimizer code iterates
      ``backbone.blocks[i].linear.weight`` to build Muon parameter groups).
    * Buffer ``n_blocks``: int64 ``(K,)``.
    * Block i is applied only to members with n_blocks > i (use index_select /
      index_copy along PACK_DIM); the other members pass through unchanged. Members
      that skip a block must receive exactly zero gradient from it.
    * Must not cache anything derived from ``n_blocks`` across forward calls
      (members can be removed in place).
    * forward(x ``(K, B, d_in)``) -> ``(K, B, d_block)``.
    """

    blocks: nn.ModuleList

    def __init__(
        self,
        *,
        d_in: int,
        d_block: int,
        n_blocks: int | Sequence[int],
        dropout: float | Sequence[float],
        activation: str = 'ReLU',
        pack_size: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    @property
    def pack_size(self) -> int:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
