"""ModelPack (a12): input encoding + MLPBackbonePack + LinearPack head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tabpack_repro.nn.linear_pack import LinearPack
from tabpack_repro.nn.mlp_pack import MLPBackbonePack
from tabpack_repro.types import PACK_DIM


def _cat_last(tensors: list[Tensor]) -> Tensor:
    """torch.cat along the last dim that does not copy a single tensor."""
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)


class OneHotEncoding(nn.Module):
    """One-hot encode int64 ordinal codes ``(..., n_cat)`` -> ``(..., sum(cards))``.

    A code >= cardinality (unknown category) encodes to all zeros for that column.
    Output dtype is float32. No parameters or buffers (keeps the pack invariant).
    """

    def __init__(self, cardinalities: Sequence[int]) -> None:
        super().__init__()
        cardinalities = [int(c) for c in cardinalities]
        if any(c < 1 for c in cardinalities):
            raise ValueError(f'cardinalities must be positive, got {cardinalities}')
        # A plain Python list (not a buffer): the module stays free of tensors, so it
        # never violates the pack invariant and never needs slicing.
        self._cardinalities = cardinalities

    @property
    def cardinalities(self) -> list[int]:
        return list(self._cardinalities)

    @property
    def d_out(self) -> int:
        return sum(self._cardinalities)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != len(self._cardinalities):
            raise ValueError(
                f'Expected (..., {len(self._cardinalities)}) category codes,'
                f' got shape {tuple(x.shape)}'
            )
        if x.dtype.is_floating_point or x.dtype == torch.bool:
            raise TypeError(f'Expected integer category codes, got {x.dtype}')
        if not self._cardinalities:
            return x.new_zeros((*x.shape[:-1], 0), dtype=torch.float32)
        x = x.long()
        return torch.cat(
            [
                # Every unknown code (>= cardinality) is mapped to the extra class
                # `cardinality`, whose column is then dropped: known codes give the
                # standard one-hot row, unknown codes give all zeros. Clamping makes
                # this work for any code >= cardinality, not only == cardinality.
                nn.functional.one_hot(x[..., i].clamp(max=c), c + 1)[..., :-1]
                for i, c in enumerate(self._cardinalities)
            ],
            dim=-1,
        ).to(torch.float32)


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
        cat_cardinalities = [int(c) for c in cat_cardinalities]
        if n_num_features < 0:
            raise ValueError(f'n_num_features must be >= 0, got {n_num_features}')
        if n_num_features == 0 and not cat_cardinalities:
            raise ValueError(
                'The model needs at least one numerical or categorical feature'
            )
        if n_classes is not None and n_classes < 2:
            raise ValueError(f'n_classes must be None or >= 2, got {n_classes}')
        if pack_size < 1:
            raise ValueError(f'pack_size must be >= 1, got {pack_size}')

        self._n_num_features = n_num_features
        self._n_cat_features = len(cat_cardinalities)
        # Binclass and regression use one output unit, squeezed away in forward.
        self._squeeze_output = n_classes is None or n_classes == 2
        # Holds no parameters/buffers (see OneHotEncoding); None without cat features.
        self.cat_encoding = (
            OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        d_in = n_num_features + sum(cat_cardinalities)
        self.backbone = MLPBackbonePack(
            d_in=d_in,
            d_block=d_block,
            n_blocks=n_blocks,
            dropout=dropout,
            activation=activation,
            pack_size=pack_size,
        )
        d_out = 1 if self._squeeze_output else n_classes
        self.head = LinearPack(d_block, d_out, pack_size=pack_size)

    @property
    def pack_size(self) -> int:
        # Derived from the parameters on every call (members can be removed in place).
        return self.backbone.pack_size

    def _to_pack(self, x: Tensor, pack_size: int, name: str) -> Tensor:
        """``(B, f)`` -> expanded view ``(K, B, f)``; ``(K, B, f)`` passes through."""
        if x.ndim == 2:
            return x.unsqueeze(PACK_DIM).expand(pack_size, -1, -1)
        if x.ndim == 3 and x.shape[PACK_DIM] == pack_size:
            return x
        raise ValueError(
            f'{name} must have shape (B, f) or (K={pack_size}, B, f),'
            f' got {tuple(x.shape)}'
        )

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        if x_num is None and x_cat is None:
            raise ValueError('At least one of x_num and x_cat must be provided')
        if (x_num is None) != (self._n_num_features == 0):
            raise ValueError(
                f'x_num is {"None" if x_num is None else "given"}, but the model has'
                f' n_num_features={self._n_num_features}'
            )
        if (x_cat is None) != (self._n_cat_features == 0):
            raise ValueError(
                f'x_cat is {"None" if x_cat is None else "given"}, but the model has'
                f' {self._n_cat_features} categorical features'
            )

        inputs: list[tuple[str, Tensor]] = []
        if x_num is not None:
            if x_num.shape[-1] != self._n_num_features:
                raise ValueError(
                    f'x_num must have {self._n_num_features} features,'
                    f' got shape {tuple(x_num.shape)}'
                )
            inputs.append(('x_num', x_num))
        if x_cat is not None:
            assert self.cat_encoding is not None
            inputs.append(('x_cat', self.cat_encoding(x_cat)))

        k = self.pack_size
        if all(t.ndim == 2 for _, t in inputs):
            # Shared rows (evaluation): concatenate once on (B, d_in), then expand to
            # (K, B, d_in) as a stride-0 view, so no K-fold copy of the input is made.
            x = self._to_pack(_cat_last([t for _, t in inputs]), k, 'input')
        else:
            # A per-member input (training). A shared one is expanded as a view first;
            # the concatenation then materializes the (K, B, d_in) input.
            x = _cat_last([self._to_pack(t, k, name) for name, t in inputs])

        x = self.backbone(x)
        x = self.head(x)
        return x.squeeze(-1) if self._squeeze_output else x
