"""Tests for tabpack_repro.nn.pack_ops (a13).

The core tests use small toy pack modules (params and buffers with a leading pack
dimension K), so that they do not depend on the other nn modules.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.nn.pack_ops import (
    get_pack_size,
    make_keep_idx,
    pack_load_members_,
    pack_select_,
    pack_state_dict,
)

K = 5
D_IN = 3
D_OUT = 4


class ToyLayer(nn.Module):
    """K independent affine maps followed by a per-member scale."""

    def __init__(self, pack_size: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(pack_size, d_in, d_out))
        self.bias = nn.Parameter(torch.randn(pack_size, d_out))
        self.register_buffer('scale', torch.rand(pack_size) + 0.5)
        # A non-persistent int64 buffer and an unset (None) buffer.
        self.register_buffer('ids', torch.arange(pack_size), persistent=False)
        self.register_buffer('unset', None)

    def forward(self, x: Tensor) -> Tensor:
        # x: (K, B, d_in) -> (K, B, d_out)
        return (x @ self.weight + self.bias[:, None]) * self.scale[:, None, None]


class ToyPack(nn.Module):
    """A nested toy pack: a ModuleList of layers plus a head."""

    def __init__(self, pack_size: int = K) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ToyLayer(pack_size, D_IN, D_OUT), ToyLayer(pack_size, D_OUT, D_OUT)]
        )
        self.head = ToyLayer(pack_size, D_OUT, 1)
        self.activation = nn.ReLU()  # A submodule without params or buffers.

    def forward(self, x: Tensor) -> Tensor:
        # x: shared (B, d_in) -> (K, B)
        x = x.expand(get_pack_size(self), *x.shape)
        for block in self.blocks:
            x = self.activation(block(x))
        return self.head(x).squeeze(-1)


def _make_toy(seed: int = 0, pack_size: int = K) -> ToyPack:
    torch.manual_seed(seed)
    return ToyPack(pack_size)


def _x(seed: int = 1) -> Tensor:
    return torch.randn(7, D_IN, generator=torch.Generator().manual_seed(seed))


def _clone_tensors(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: x.detach().clone()
        for name, x in [*module.named_parameters(), *module.named_buffers()]
    }


# >>> get_pack_size


def test_get_pack_size_nested_module() -> None:
    assert get_pack_size(_make_toy()) == K
    assert get_pack_size(_make_toy(pack_size=1)) == 1
    assert get_pack_size(nn.Linear(3, 3)) == 3  # Any module obeying the invariant.


def test_get_pack_size_only_buffers() -> None:
    module = nn.Module()
    module.register_buffer('p', torch.zeros(4))
    assert get_pack_size(module) == 4


@pytest.mark.parametrize('exc', [ValueError, AssertionError])
def test_get_pack_size_mismatch_raises(exc: type[Exception]) -> None:
    module = _make_toy()
    module.head.register_buffer('extra', torch.zeros(K + 1, 2))
    with pytest.raises(exc, match='Inconsistent pack sizes'):
        get_pack_size(module)


def test_get_pack_size_rejects_scalars_and_empty_modules() -> None:
    module = _make_toy()
    module.register_buffer('scalar', torch.tensor(1.0))
    with pytest.raises(ValueError, match='scalar'):
        get_pack_size(module)
    with pytest.raises(ValueError, match='no parameters or buffers'):
        get_pack_size(nn.ReLU())


# >>> make_keep_idx


@pytest.mark.parametrize(
    ('remove', 'expected'),
    [
        ([], [0, 1, 2, 3, 4]),
        ([2], [0, 1, 3, 4]),
        ([4, 0], [1, 2, 3]),
        ([3, 3, 1], [0, 2, 4]),
        ([0, 1, 2, 3, 4], []),
    ],
)
def test_make_keep_idx(remove: list[int], expected: list[int]) -> None:
    keep = make_keep_idx(K, torch.tensor(remove, dtype=torch.int64))
    assert keep.dtype == torch.int64
    assert keep.device == torch.device('cpu')
    assert keep.tolist() == expected


def test_make_keep_idx_accepts_other_int_dtypes_and_float_empty() -> None:
    assert make_keep_idx(3, torch.tensor([1], dtype=torch.int32)).tolist() == [0, 2]
    # torch.tensor([]) is float32: an empty removal is still valid.
    keep = make_keep_idx(3, torch.tensor([]))
    assert keep.dtype == torch.int64
    assert keep.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ('remove', 'exc'),
    [
        (torch.tensor([5]), IndexError),
        (torch.tensor([-1]), IndexError),
        (torch.tensor([1.0]), TypeError),
        (torch.tensor([True, False, False, False, False]), TypeError),
        (torch.tensor([[0, 1]]), ValueError),
    ],
)
def test_make_keep_idx_invalid(remove: Tensor, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        make_keep_idx(K, remove)


# >>> pack_select_


@pytest.mark.parametrize(
    'keep', [[0, 1, 2, 3, 4], [1, 3], [3, 0, 2], [4], [4, 3, 2, 1, 0]]
)
def test_pack_select_outputs_of_kept_members_unchanged(keep: list[int]) -> None:
    module = _make_toy().eval()
    x = _x()
    with torch.no_grad():
        expected = module(x)[keep]
    pack_select_(module, torch.tensor(keep))
    assert get_pack_size(module) == len(keep)
    with torch.no_grad():
        actual = module(x)
    assert actual.shape == (len(keep), x.shape[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pack_select_slices_every_tensor_in_order() -> None:
    module = _make_toy()
    before = _clone_tensors(module)
    keep = torch.tensor([3, 0, 2])
    pack_select_(module, keep)
    after = _clone_tensors(module)
    assert after.keys() == before.keys()
    for name, x in before.items():
        assert after[name].dtype == x.dtype, name
        assert after[name].is_contiguous(), name
        torch.testing.assert_close(after[name], x[keep], rtol=0, atol=0, msg=name)


def test_pack_select_preserves_parameter_identity_and_optimizer_groups() -> None:
    module = _make_toy()
    params = list(module.parameters())
    ids = [id(p) for p in params]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module(_x()).sum().backward()
    assert all(p.grad is not None for p in params)

    pack_select_(module, torch.tensor([1, 4]))

    assert [id(p) for p in module.parameters()] == ids
    assert all(isinstance(p, nn.Parameter) and p.requires_grad for p in params)
    assert all(p.grad is None for p in params)
    group_params = optimizer.param_groups[0]['params']
    assert all(a is b for a, b in zip(group_params, params, strict=True))

    # The optimizer still trains the (smaller) parameters.
    before = [p.detach().clone() for p in params]
    module(_x()).sum().backward()
    assert all(p.grad.shape == p.shape for p in params)
    optimizer.step()
    assert all(not torch.equal(p, b) for p, b in zip(params, before, strict=True))


def test_pack_select_does_not_alias_old_storage() -> None:
    module = _make_toy()
    old_weight = module.head.weight.data
    old_scale = module.head.scale
    pack_select_(module, torch.tensor([0, 1, 2, 3, 4]))
    kept_weight = module.head.weight.detach().clone()
    kept_scale = module.head.scale.clone()
    old_weight.add_(1.0)
    old_scale.add_(1.0)
    torch.testing.assert_close(module.head.weight.detach(), kept_weight)
    torch.testing.assert_close(module.head.scale, kept_scale)


def test_pack_select_buffers_keep_persistence_flag() -> None:
    module = _make_toy()
    state_keys = set(module.state_dict())
    buffer_names = {name for name, _ in module.named_buffers()}
    pack_select_(module, torch.tensor([2, 0]))
    assert set(module.state_dict()) == state_keys
    assert {name for name, _ in module.named_buffers()} == buffer_names
    assert 'head.ids' not in state_keys  # Non-persistent, before and after.
    assert 'head.scale' in state_keys
    assert module.head.ids.tolist() == [2, 0]
    assert module.head.unset is None
    # The state dict round trip still works.
    module.load_state_dict(module.state_dict())


def test_pack_select_shared_tensors_sliced_once() -> None:
    module = _make_toy()
    # A parameter and a buffer that are shared by two submodules.
    module.blocks[1].bias = module.blocks[0].bias
    shared_buffer = torch.arange(K) * 10
    module.blocks[0].register_buffer('shared', shared_buffer)
    module.head.register_buffer('shared', shared_buffer)

    pack_select_(module, torch.tensor([4, 1]))

    assert module.blocks[1].bias is module.blocks[0].bias
    assert module.blocks[0].bias.shape == (2, D_OUT)
    assert module.blocks[0].shared is module.head.shared
    assert module.head.shared.tolist() == [40, 10]
    assert get_pack_size(module) == 2


def test_pack_select_remove_all_but_one_then_all() -> None:
    module = _make_toy()
    x = _x()
    with torch.no_grad():
        expected = module(x)[[2]]
    pack_select_(module, make_keep_idx(K, torch.tensor([0, 1, 3, 4])))
    assert get_pack_size(module) == 1
    with torch.no_grad():
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    pack_select_(module, make_keep_idx(1, torch.tensor([0])))
    assert get_pack_size(module) == 0
    assert all(x.shape[0] == 0 for x in module.parameters())


def test_pack_select_with_empty_removal_is_identity() -> None:
    module = _make_toy()
    before = _clone_tensors(module)
    pack_select_(module, make_keep_idx(K, torch.tensor([], dtype=torch.int64)))
    after = _clone_tensors(module)
    for name, x in before.items():
        torch.testing.assert_close(after[name], x, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('keep', 'exc'),
    [
        (torch.tensor([0, 5]), IndexError),
        (torch.tensor([-1]), IndexError),
        (torch.tensor([1, 1]), ValueError),
        (torch.tensor([[0, 1]]), ValueError),
        (torch.tensor([0.0]), TypeError),
    ],
)
def test_pack_select_invalid_keep_idx_leaves_module_intact(
    keep: Tensor, exc: type[Exception]
) -> None:
    module = _make_toy()
    before = _clone_tensors(module)
    with pytest.raises(exc):
        pack_select_(module, keep)
    after = _clone_tensors(module)
    for name, x in before.items():
        assert torch.equal(after[name], x), name


# >>> pack_state_dict / pack_load_members_


def test_pack_state_dict_is_a_detached_clone() -> None:
    module = _make_toy()
    state = pack_state_dict(module)
    names = [name for name, _ in [*module.named_parameters(), *module.named_buffers()]]
    assert list(state) == names
    assert 'head.ids' in state  # Non-persistent buffers are included too.
    for name, x in [*module.named_parameters(), *module.named_buffers()]:
        assert not state[name].requires_grad
        assert not isinstance(state[name], nn.Parameter)
        assert state[name].data_ptr() != x.data_ptr()
        assert torch.equal(state[name], x)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(1.0)
    for name, p in module.named_parameters():
        assert not torch.equal(state[name], p)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for x in [*module.parameters(), *module.buffers()]:
            x.add_(torch.ones_like(x))


@pytest.mark.parametrize('member_idx', [[1, 3], [3, 1], [0], [4, 4, 2]])
def test_pack_load_members_restores_only_selected(member_idx: list[int]) -> None:
    module = _make_toy()
    state = pack_state_dict(module)
    _perturb(module)
    perturbed = _clone_tensors(module)
    params = dict(module.named_parameters())
    ptrs = {
        name: x.data_ptr()
        for name, x in [*module.named_parameters(), *module.named_buffers()]
    }

    pack_load_members_(module, state, torch.tensor(member_idx))

    others = [k for k in range(K) if k not in member_idx]
    for name, x in [*module.named_parameters(), *module.named_buffers()]:
        assert x.data_ptr() == ptrs[name], f'{name} was not updated in place'
        assert torch.equal(x[member_idx], state[name][member_idx]), name
        assert torch.equal(x[others], perturbed[name][others]), name
    assert all(p is params[name] for name, p in module.named_parameters())


def test_pack_load_members_round_trip_of_the_whole_pack() -> None:
    module = _make_toy().eval()
    x = _x()
    with torch.no_grad():
        expected = module(x)
    state = pack_state_dict(module)
    _perturb(module)
    pack_load_members_(module, state, torch.arange(K))
    with torch.no_grad():
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    for name, x in pack_state_dict(module).items():
        assert torch.equal(x, state[name]), name


def test_pack_load_members_empty_is_noop() -> None:
    module = _make_toy()
    state = pack_state_dict(module)
    _perturb(module)
    before = _clone_tensors(module)
    pack_load_members_(module, state, torch.tensor([], dtype=torch.int64))
    for name, x in _clone_tensors(module).items():
        assert torch.equal(x, before[name]), name


def test_pack_load_members_after_pack_select_with_sliced_state() -> None:
    # Best checkpoints are recorded for the whole pack and then sliced alongside
    # the model when members are removed.
    module = _make_toy().eval()
    x = _x()
    state = pack_state_dict(module)
    with torch.no_grad():
        expected = module(x)
    _perturb(module)
    keep = torch.tensor([3, 1])
    pack_select_(module, keep)
    sliced = {name: t[keep] for name, t in state.items()}
    pack_load_members_(module, sliced, torch.tensor([1]))
    with torch.no_grad():
        actual = module(x)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert not torch.equal(actual[0], expected[3])


def test_pack_load_members_works_with_autograd_and_bumps_version() -> None:
    module = _make_toy()
    state = pack_state_dict(module)
    weight = module.head.weight
    assert weight.requires_grad
    version = weight._version
    pack_load_members_(module, state, torch.tensor([0]))
    assert weight._version > version
    assert weight.requires_grad and weight.is_leaf


def test_pack_load_members_validates_state() -> None:
    module = _make_toy()
    state = pack_state_dict(module)
    idx = torch.tensor([0])

    missing = dict(state)
    del missing['head.scale']
    with pytest.raises(KeyError, match=r'head\.scale'):
        pack_load_members_(module, missing, idx)

    unexpected = {**state, 'foo': torch.zeros(K)}
    with pytest.raises(KeyError, match='foo'):
        pack_load_members_(module, unexpected, idx)

    wrong_shape = {**state, 'head.bias': torch.zeros(K + 1, 1)}
    with pytest.raises(ValueError, match=r'head\.bias'):
        pack_load_members_(module, wrong_shape, idx)

    wrong_dtype = {**state, 'head.ids': state['head.ids'].float()}
    with pytest.raises(ValueError, match=r'head\.ids'):
        pack_load_members_(module, wrong_dtype, idx)

    with pytest.raises(IndexError):
        pack_load_members_(module, state, torch.tensor([K]))


# >>> CUDA (skipped unless run with TABPACK_GPU=1 on a machine with a GPU)


@pytest.mark.gpu
@pytest.mark.parametrize('idx_on_device', [False, True])
def test_pack_ops_on_cuda(cuda_device: torch.device, idx_on_device: bool) -> None:
    module = _make_toy().to(cuda_device).eval()
    x = _x().to(cuda_device)
    idx_device = cuda_device if idx_on_device else torch.device('cpu')

    remove = torch.tensor([0, 2], device=idx_device)
    keep = make_keep_idx(K, remove)
    assert keep.device.type == idx_device.type
    assert keep.tolist() == [1, 3, 4]

    params = list(module.parameters())
    state = pack_state_dict(module)
    assert all(t.device.type == 'cuda' for t in state.values())
    with torch.no_grad():
        expected = module(x)[keep.cpu()]
    pack_select_(module, keep)
    assert all(a is b for a, b in zip(module.parameters(), params, strict=True))
    assert all(t.device.type == 'cuda' for t in module.buffers())
    with torch.no_grad():
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)

    # A CPU checkpoint can be loaded into a CUDA module (and vice versa for idx).
    sliced = {name: t[keep].cpu() for name, t in state.items()}
    _perturb(module)
    pack_load_members_(module, sliced, torch.tensor([0, 2], device=idx_device))
    with torch.no_grad():
        actual = module(x)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
