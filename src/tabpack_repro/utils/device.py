"""Device and mixed-precision helpers (a29)."""

from __future__ import annotations

import contextlib

import torch


def resolve_device(spec: str = 'auto') -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise torch.device(spec)."""
    raise NotImplementedError


def make_autocast(
    amp_dtype: str | None, device: torch.device
) -> contextlib.AbstractContextManager | None:
    """torch.autocast(device_type='cuda', dtype=...) for 'bfloat16'/'float16' on CUDA;
    None on CPU or when amp_dtype is None. Must be reusable (a fresh context manager
    per `with`): return an object whose __enter__ creates a new torch.autocast."""
    raise NotImplementedError


def describe_device(device: torch.device) -> dict[str, str | None]:
    """{'device': str(device), 'gpu': name or None, 'torch': torch.__version__,
    'cuda': torch.version.cuda}."""
    raise NotImplementedError
