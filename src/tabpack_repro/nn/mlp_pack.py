"""MLPBackbonePack (a11): K MLPs with (possibly) different depths and dropout rates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tabpack_repro.nn.dropout_pack import DropoutPack, make_activation
from tabpack_repro.nn.linear_pack import LinearPack
from tabpack_repro.types import PACK_DIM


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
        self.linear = LinearPack(in_features, out_features, pack_size=pack_size)
        # Activations are element-wise and stateless, so one plain module serves all
        # members (and any subset of them).
        self.activation = make_activation(activation)
        self.dropout = DropoutPack(dropout, pack_size=pack_size)

    @property
    def pack_size(self) -> int:
        return self.linear.pack_size

    def forward(self, x: Tensor, member_idx: Tensor | None = None) -> Tensor:
        """x ``(K', B, in)`` -> ``(K', B, out)``; K' = len(member_idx) if given."""
        x = self.linear(x, member_idx)
        x = self.activation(x)
        return self.dropout(x, member_idx)


def _normalize_n_blocks(n_blocks: int | Sequence[int], pack_size: int) -> list[int]:
    if pack_size < 1:
        raise ValueError(f'pack_size must be positive, got {pack_size}')
    if isinstance(n_blocks, int):
        values = [n_blocks] * pack_size
    else:
        values = [int(n) for n in n_blocks]
        if len(values) != pack_size:
            raise ValueError(
                f'n_blocks has {len(values)} values, expected pack_size={pack_size}'
            )
    if any(n < 1 for n in values):
        raise ValueError(f'every n_blocks value must be >= 1, got {values}')
    return values


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
    n_blocks: Tensor

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
        n_blocks_list = _normalize_n_blocks(n_blocks, pack_size)
        self.blocks = nn.ModuleList(
            [
                MLPBlockPack(
                    d_in if i == 0 else d_block,
                    d_block,
                    dropout=dropout,
                    activation=activation,
                    pack_size=pack_size,
                )
                for i in range(max(n_blocks_list))
            ]
        )
        self.register_buffer('n_blocks', torch.tensor(n_blocks_list, dtype=torch.int64))

    @property
    def pack_size(self) -> int:
        # Derived from the buffer on every access: members may be removed in place.
        return self.n_blocks.shape[PACK_DIM]

    def _active_members(self) -> list[Tensor | None]:
        """Per block: indices of the members that apply it (None = all members).

        Blocks that no member applies are omitted (they can only be trailing blocks).
        Recomputed on every forward from the current ``n_blocks`` buffer; the only
        host-device synchronization is reading the per-block member counts.
        """
        n_blocks = self.n_blocks
        pack_size = n_blocks.shape[PACK_DIM]
        block_ids = torch.arange(len(self.blocks), device=n_blocks.device)
        # active[i, k] == True <=> member k applies block i.
        active = block_ids[:, None] < n_blocks[None, :]
        counts = active.sum(dim=1).tolist()
        result: list[Tensor | None] = []
        for i, count in enumerate(counts):
            if count == 0:
                # n_blocks > i is monotone in i, so all later blocks are unused too.
                break
            if count == pack_size:
                result.append(None)
            else:
                # nonzero_static avoids one more sync per block (indices are sorted).
                result.append(torch.nonzero_static(active[i], size=count)[:, 0])
        return result

    def forward(self, x: Tensor) -> Tensor:
        for block, member_idx in zip(self.blocks, self._active_members(), strict=False):
            if member_idx is None:
                x = block(x)
            else:
                # Members outside member_idx keep their activations unchanged and get
                # no gradient through this block (their rows are simply copied).
                y = block(x.index_select(PACK_DIM, member_idx), member_idx)
                x = x.index_copy(PACK_DIM, member_idx, y)
        return x
