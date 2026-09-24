"""In-place member removal/selection and per-member checkpointing (a13).

All functions rely on the pack invariant (see tabpack_repro.types): every parameter
and buffer of a pack module has the pack dimension first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from tabpack_repro.types import PACK_DIM


class _PackInvariantError(ValueError, AssertionError):
    """A module violates the pack invariant (every param/buffer has dim 0 == K).

    The contract says get_pack_size "asserts" the invariant. An explicit exception
    (unlike a bare ``assert``) survives ``python -O``; it subclasses both ValueError
    and AssertionError so callers can catch it either way.
    """


def _named_tensors(module: nn.Module) -> Iterator[tuple[str, Tensor]]:
    """All parameters, then all buffers, with their qualified names (deduplicated)."""
    yield from module.named_parameters()
    yield from module.named_buffers()


def _as_index(idx: Tensor, what: str) -> Tensor:
    """Validate an index tensor and return it as a 1-D int64 CPU tensor."""
    idx = torch.as_tensor(idx).detach()
    if idx.ndim == 0:
        idx = idx.reshape(1)
    if idx.ndim != 1:
        raise ValueError(f'{what} must be 1-D, got shape {tuple(idx.shape)}')
    # torch.tensor([]) is float32; an empty index of any dtype is accepted.
    if idx.numel() > 0 and (
        idx.dtype == torch.bool or idx.is_floating_point() or idx.is_complex()
    ):
        raise TypeError(f'{what} must have an integer dtype, got {idx.dtype}')
    return idx.to(device='cpu', dtype=torch.int64)


def _check_range(idx: Tensor, pack_size: int, what: str) -> None:
    if idx.numel() > 0:
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or hi >= pack_size:
            raise IndexError(
                f'{what} must be in [0, {pack_size}), got values in [{lo}, {hi}]'
            )


def _index_on_device(idx: Tensor) -> Callable[[torch.device], Tensor]:
    """Return ``device -> idx on that device``; each device is copied to once."""
    cache: dict[torch.device, Tensor] = {idx.device: idx}

    def get(device: torch.device) -> Tensor:
        out = cache.get(device)
        if out is None:
            out = cache[device] = idx.to(device)
        return out

    return get


def get_pack_size(module: nn.Module) -> int:
    """Pack size of a pack module; assert all params/buffers agree on dim 0."""
    pack_size: int | None = None
    first_name = ''
    for name, x in _named_tensors(module):
        if x.ndim == 0:
            raise _PackInvariantError(
                f'{name!r} is a scalar tensor; every param/buffer of a pack module '
                'must have the pack dimension first'
            )
        if pack_size is None:
            pack_size, first_name = x.shape[PACK_DIM], name
        elif x.shape[PACK_DIM] != pack_size:
            raise _PackInvariantError(
                f'Inconsistent pack sizes: {first_name!r} has {pack_size} members, '
                f'{name!r} has shape {tuple(x.shape)}'
            )
    if pack_size is None:
        raise _PackInvariantError(
            f'{type(module).__name__} has no parameters or buffers, '
            'so its pack size is undefined'
        )
    return pack_size


def make_keep_idx(pack_size: int, remove_idx: Tensor) -> Tensor:
    """Sorted int64 indices of members NOT in remove_idx (on remove_idx.device)."""
    device = torch.as_tensor(remove_idx).device
    remove = _as_index(remove_idx, 'remove_idx')
    _check_range(remove, pack_size, 'remove_idx')
    keep = torch.ones(pack_size, dtype=torch.bool)
    keep[remove] = False
    return keep.nonzero().squeeze(1).to(device)


def pack_select_(module: nn.Module, keep_idx: Tensor) -> None:
    """Keep only members `keep_idx` (in that order), in place.

    Parameter identity MUST be preserved (assign ``param.data = param.data[keep_idx]``
    and set ``param.grad = None``) so that optimizer param groups stay valid; the
    optimizer state is sliced separately by optim.pack_utils.optimizer_select_.
    Buffers are replaced by their slices.

    Call it only when no autograd graph that used the parameters is alive (e.g.
    after backward/step, or after evaluation under no_grad): such a graph caches
    the parameters' old shapes, and a later backward through it would fail.
    """
    # Validate everything before mutating, so that a bad call leaves `module` intact.
    pack_size = get_pack_size(module)
    keep = _as_index(keep_idx, 'keep_idx')
    _check_range(keep, pack_size, 'keep_idx')
    if torch.unique(keep).numel() != keep.numel():
        raise ValueError(f'keep_idx must not contain duplicates, got {keep.tolist()}')
    index_on = _index_on_device(keep)

    # Walk the module tree ourselves instead of using named_parameters/named_buffers:
    # a tensor shared by several (sub)modules must be sliced exactly once, and a
    # shared buffer must be re-registered in every module that holds it.
    seen_params: set[int] = set()  # ids; the module keeps every parameter alive
    # id(old buffer) -> (old buffer, its slice); the old buffer is kept alive so
    # that its id cannot be reused by a new tensor while we iterate.
    sliced_buffers: dict[int, tuple[Tensor, Tensor]] = {}
    with torch.no_grad():
        for submodule in module.modules():
            for param in submodule.parameters(recurse=False):
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))
                # index_select makes a new contiguous tensor: no aliasing with the
                # old storage. The Parameter object (and so its identity) survives.
                param.data = param.data.index_select(PACK_DIM, index_on(param.device))
                param.grad = None
            for name, buffer in submodule.named_buffers(
                recurse=False, remove_duplicate=False
            ):
                entry = sliced_buffers.get(id(buffer))
                if entry is None:
                    new = buffer.index_select(PACK_DIM, index_on(buffer.device))
                    entry = sliced_buffers[id(buffer)] = (buffer, new)
                # Assigning to an existing buffer name goes through
                # nn.Module.__setattr__, which updates `_buffers[name]` and leaves
                # the persistent/non-persistent flag untouched.
                setattr(submodule, name, entry[1])


def pack_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Detached clones of all params and buffers (a per-member-sliceable checkpoint)."""
    get_pack_size(module)  # Checks the invariant: every entry is sliceable along K.
    return {name: x.detach().clone() for name, x in _named_tensors(module)}


def pack_load_members_(
    module: nn.Module, state: dict[str, Tensor], member_idx: Tensor
) -> None:
    """For members member_idx, copy their slices from `state` into `module` in place.

    `state` has the same keys as pack_state_dict(module) and the same pack size.
    """
    pack_size = get_pack_size(module)
    tensors = dict(_named_tensors(module))
    missing = [k for k in tensors if k not in state]
    unexpected = [k for k in state if k not in tensors]
    if missing or unexpected:
        raise KeyError(
            f'State keys do not match the module: missing={missing}, '
            f'unexpected={unexpected}'
        )
    for name, x in tensors.items():
        src = state[name]
        if src.shape != x.shape or src.dtype != x.dtype:
            raise ValueError(
                f'{name!r}: state has {tuple(src.shape)} {src.dtype}, '
                f'module has {tuple(x.shape)} {x.dtype}'
            )
    idx = _as_index(member_idx, 'member_idx')
    _check_range(idx, pack_size, 'member_idx')
    if idx.numel() == 0:
        return
    # Duplicates would make index_copy_ nondeterministic on CUDA; order is irrelevant.
    index_on = _index_on_device(torch.unique(idx))

    with torch.no_grad():
        for name, x in tensors.items():
            src = state[name]
            rows = src.index_select(PACK_DIM, index_on(src.device)).to(x.device)
            # In place on the Parameter/buffer itself: identity and storage are kept,
            # and autograd's version counter sees the change.
            x.index_copy_(PACK_DIM, index_on(x.device), rows)
