"""Device and mixed-precision helpers (a29)."""

from __future__ import annotations

import contextlib
from types import TracebackType
from typing import Self

import torch

_AMP_DTYPES: dict[str, torch.dtype] = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}


def resolve_device(spec: str = 'auto') -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise torch.device(spec)."""
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


class _ReusableAutocast(contextlib.AbstractContextManager):
    """A context manager that can be entered any number of times (also nested).

    torch.autocast instances are single-use, so every `with` creates a fresh one;
    a stack keeps nested `with` blocks paired with their own instance.
    """

    def __init__(self, device_type: str, dtype: torch.dtype) -> None:
        self.device_type = device_type
        self.dtype = dtype
        self._active: list[torch.autocast] = []

    def __enter__(self) -> Self:
        ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        ctx.__enter__()
        self._active.append(ctx)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return self._active.pop().__exit__(exc_type, exc, tb)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(device_type={self.device_type!r}, '
            f'dtype={self.dtype})'
        )


def make_autocast(
    amp_dtype: str | None, device: torch.device
) -> contextlib.AbstractContextManager | None:
    """torch.autocast(device_type='cuda', dtype=...) for 'bfloat16'/'float16' on CUDA;
    None on CPU or when amp_dtype is None. Must be reusable (a fresh context manager
    per `with`): return an object whose __enter__ creates a new torch.autocast.

    Any other amp_dtype raises ValueError (also on CPU, so that config errors are not
    hidden by the device). float16 needs a GradScaler in the trainer; bfloat16 does not.
    """
    if amp_dtype is None:
        return None
    if amp_dtype not in _AMP_DTYPES:
        raise ValueError(
            f'amp_dtype must be one of {sorted(_AMP_DTYPES)} or None, got {amp_dtype!r}'
        )
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    return _ReusableAutocast('cuda', _AMP_DTYPES[amp_dtype])


def describe_device(device: torch.device) -> dict[str, str | None]:
    """{'device': str(device), 'gpu': name or None, 'torch': torch.__version__,
    'cuda': torch.version.cuda}."""
    device = torch.device(device)
    gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    return {
        'device': str(device),
        'gpu': gpu,
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
    }
