"""ModelPack (a12): input encoding + MLPBackbonePack + LinearPack head."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from tabpack_repro.nn.linear_pack import LinearPack
from tabpack_repro.nn.mlp_pack import MLPBackbonePack


class OneHotEncoding(nn.Module):
    """One-hot encode int64 ordinal codes ``(..., n_cat)`` -> ``(..., sum(cards))``.

    A code >= cardinality (unknown category) encodes to all zeros for that column.
    Output dtype is float32. No parameters or buffers (keeps the pack invariant).
    """

    def __init__(self, cardinalities: Sequence[int]) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ModelPack(nn.Module):
    """K MLPs for tabular data.

    forward(x_num, x_cat):
    * each input is either shared ``(B, f)`` (evaluation: every member sees the same
      rows; expand to ``(K, B, f)`` without copying) or per-member ``(K, B, f)``
      (training: every member has its own batch);
    * x = concat([x_num, one_hot(x_cat)], -1) -> backbone -> head;
    * returns logits ``(K, B)`` if n_classes in (None, 2) else ``(K, B, n_classes)``.

    Attributes: ``backbone`` (MLPBackbonePack), ``head`` (LinearPack).
    """

    backbone: MLPBackbonePack
    head: LinearPack

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        n_classes: int | None,
        pack_size: int,
        d_block: int,
        n_blocks: int | Sequence[int],
        dropout: float | Sequence[float],
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()
        raise NotImplementedError

    @property
    def pack_size(self) -> int:
        raise NotImplementedError

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        raise NotImplementedError
