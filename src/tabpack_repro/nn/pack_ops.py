"""In-place member removal/selection and per-member checkpointing (a13).

All functions rely on the pack invariant (see tabpack_repro.types): every parameter
and buffer of a pack module has the pack dimension first.
"""

from __future__ import annotations

from torch import Tensor, nn


def get_pack_size(module: nn.Module) -> int:
    """Pack size of a pack module; assert all params/buffers agree on dim 0."""
    raise NotImplementedError


def make_keep_idx(pack_size: int, remove_idx: Tensor) -> Tensor:
    """Sorted int64 indices of members NOT in remove_idx (on remove_idx.device)."""
    raise NotImplementedError


def pack_select_(module: nn.Module, keep_idx: Tensor) -> None:
    """Keep only members `keep_idx` (in that order), in place.

    Parameter identity MUST be preserved (assign ``param.data = param.data[keep_idx]``
    and set ``param.grad = None``) so that optimizer param groups stay valid; the
    optimizer state is sliced separately by optim.pack_utils.optimizer_select_.
    Buffers are replaced by their slices.
    """
    raise NotImplementedError


def pack_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Detached clones of all params and buffers (a per-member-sliceable checkpoint)."""
    raise NotImplementedError


def pack_load_members_(
    module: nn.Module, state: dict[str, Tensor], member_idx: Tensor
) -> None:
    """For members member_idx, copy their slices from `state` into `module` in place.

    `state` has the same keys as pack_state_dict(module) and the same pack size.
    """
    raise NotImplementedError
