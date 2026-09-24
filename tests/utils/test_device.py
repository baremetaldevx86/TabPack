from __future__ import annotations

import contextlib

import pytest
import torch

from tabpack_repro.utils.device import describe_device, make_autocast, resolve_device

CPU = torch.device('cpu')
CUDA = torch.device('cuda')


# resolve_device


def test_resolve_device_explicit() -> None:
    assert resolve_device('cpu') == CPU
    assert resolve_device('cuda') == CUDA
    assert resolve_device('cuda:1') == torch.device('cuda', 1)


@pytest.mark.parametrize('available', [False, True])
def test_resolve_device_auto(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    expected = CUDA if available else CPU
    assert resolve_device('auto') == expected
    assert resolve_device() == expected


def test_resolve_device_auto_real() -> None:
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert resolve_device('auto').type == expected


def test_resolve_device_invalid() -> None:
    with pytest.raises(RuntimeError):
        resolve_device('not-a-device')


# make_autocast


@pytest.mark.parametrize('amp_dtype', [None, 'bfloat16', 'float16'])
def test_make_autocast_none_on_cpu(amp_dtype: str | None) -> None:
    assert make_autocast(amp_dtype, CPU) is None


def test_make_autocast_none_without_dtype_on_cuda() -> None:
    assert make_autocast(None, CUDA) is None


@pytest.mark.parametrize('amp_dtype', ['bf16', 'fp16', 'float32', 'BFloat16', ''])
@pytest.mark.parametrize('device', [CPU, CUDA])
def test_make_autocast_invalid_name(amp_dtype: str, device: torch.device) -> None:
    with pytest.raises(ValueError, match='amp_dtype'):
        make_autocast(amp_dtype, device)


@pytest.mark.parametrize(
    ('amp_dtype', 'dtype'), [('bfloat16', torch.bfloat16), ('float16', torch.float16)]
)
def test_make_autocast_builds_cuda_context(amp_dtype: str, dtype: torch.dtype) -> None:
    # Construction does not touch CUDA, so this runs on CPU-only machines too.
    ctx = make_autocast(amp_dtype, CUDA)
    assert isinstance(ctx, contextlib.AbstractContextManager)
    assert ctx.device_type == 'cuda'
    assert ctx.dtype == dtype


@pytest.mark.gpu
@pytest.mark.parametrize(
    ('amp_dtype', 'dtype'), [('bfloat16', torch.bfloat16), ('float16', torch.float16)]
)
def test_make_autocast_is_reusable_on_cuda(
    cuda_device: torch.device, amp_dtype: str, dtype: torch.dtype
) -> None:
    ctx = make_autocast(amp_dtype, cuda_device)
    assert ctx is not None
    x = torch.randn(4, 8, device=cuda_device)
    w = torch.randn(8, 3, device=cuda_device)
    for _ in range(3):  # the trainer enters the same object many times
        assert not torch.is_autocast_enabled('cuda')
        with ctx:
            assert torch.is_autocast_enabled('cuda')
            assert (x @ w).dtype == dtype
        assert not torch.is_autocast_enabled('cuda')
        assert (x @ w).dtype == torch.float32


@pytest.mark.gpu
def test_make_autocast_nested_and_exception(cuda_device: torch.device) -> None:
    ctx = make_autocast('bfloat16', cuda_device)
    assert ctx is not None
    x = torch.randn(4, 4, device=cuda_device)
    with ctx:
        with ctx:
            assert (x @ x).dtype == torch.bfloat16
        assert (x @ x).dtype == torch.bfloat16
    assert (x @ x).dtype == torch.float32
    with pytest.raises(KeyError), ctx:
        raise KeyError('boom')
    assert not torch.is_autocast_enabled('cuda')
    with ctx:
        assert (x @ x).dtype == torch.bfloat16


# describe_device


def test_describe_device_cpu() -> None:
    info = describe_device(CPU)
    assert info == {
        'device': 'cpu',
        'gpu': None,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }
    assert type(info['torch']) is str


@pytest.mark.gpu
def test_describe_device_cuda(cuda_device: torch.device) -> None:
    info = describe_device(cuda_device)
    assert info['device'] == str(cuda_device)
    assert isinstance(info['gpu'], str) and info['gpu']
    assert info['cuda'] == torch.version.cuda
