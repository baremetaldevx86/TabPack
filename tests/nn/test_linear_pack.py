"""Tests for LinearPack (a09)."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from tabpack_repro.nn.linear_pack import LinearPack

K, B, D_IN, D_OUT = 4, 7, 5, 3


def _independent_linears(pack: LinearPack) -> list[nn.Linear]:
    """K nn.Linear layers holding copies of the pack members' weights."""
    layers = []
    for k in range(pack.pack_size):
        layer = nn.Linear(
            pack.in_features, pack.out_features, bias=pack.bias is not None
        )
        with torch.no_grad():
            layer.weight.copy_(pack.weight[k].T)
            if pack.bias is not None:
                layer.bias.copy_(pack.bias[k])
        layers.append(layer)
    return layers


@pytest.mark.parametrize('bias', [True, False])
def test_parameter_shapes(bias: bool) -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=K, bias=bias)
    assert pack.weight.shape == (K, D_IN, D_OUT)
    assert pack.pack_size == K
    if bias:
        assert pack.bias is not None
        assert pack.bias.shape == (K, D_OUT)
    else:
        assert pack.bias is None
    # The pack invariant: every parameter has the pack dimension first.
    for p in pack.parameters():
        assert p.shape[0] == K
    assert list(pack.state_dict()) == (['weight', 'bias'] if bias else ['weight'])


@pytest.mark.parametrize('bias', [True, False])
def test_forward_matches_independent_linears(bias: bool) -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K, bias=bias)
    layers = _independent_linears(pack)
    x = torch.randn(K, B, D_IN)

    y = pack(x)
    expected = torch.stack([layer(x[k]) for k, layer in enumerate(layers)])
    assert y.shape == (K, B, D_OUT)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)


def test_gradients_match_independent_linears() -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    layers = _independent_linears(pack)
    x = torch.randn(K, B, D_IN)
    target = torch.randn(K, B, D_OUT)

    ((pack(x) - target) ** 2).sum().backward()
    for k, layer in enumerate(layers):
        ((layer(x[k]) - target[k]) ** 2).sum().backward()
        assert pack.weight.grad is not None and pack.bias is not None
        assert pack.bias.grad is not None and layer.bias.grad is not None
        torch.testing.assert_close(
            pack.weight.grad[k], layer.weight.grad.T, atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            pack.bias.grad[k], layer.bias.grad, atol=1e-5, rtol=0
        )


def test_input_gradient_flows() -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    x = torch.randn(K, B, D_IN, requires_grad=True)
    pack(x).sum().backward()
    assert x.grad is not None
    # d(sum(x @ W + b)) / dx = W.sum(-1), identical for every row of a member.
    expected = pack.weight.detach().sum(-1).unsqueeze(1).expand(K, B, D_IN)
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('idx', [[2, 0], [3], [1, 1, 3], [0, 1, 2, 3]])
def test_member_idx_equals_slicing(bias: bool, idx: list[int]) -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K, bias=bias)
    member_idx = torch.tensor(idx)
    x_sub = torch.randn(len(idx), B, D_IN)

    y_sub = pack(x_sub, member_idx)

    # 1) The same as a pack built from the sliced parameters.
    sliced = LinearPack(D_IN, D_OUT, pack_size=len(idx), bias=bias)
    with torch.no_grad():
        sliced.weight.copy_(pack.weight[member_idx])
        if bias:
            assert sliced.bias is not None and pack.bias is not None
            sliced.bias.copy_(pack.bias[member_idx])
    assert y_sub.shape == (len(idx), B, D_OUT)
    torch.testing.assert_close(y_sub, sliced(x_sub), atol=0, rtol=0)

    # 2) The same as slicing the output of the full pack (for distinct indices).
    if len(set(idx)) == len(idx):
        x_full = torch.randn(K, B, D_IN)
        x_full[member_idx] = x_sub
        torch.testing.assert_close(y_sub, pack(x_full)[member_idx], atol=1e-6, rtol=0)


def test_gradients_flow_only_into_selected_members() -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    member_idx = torch.tensor([3, 1])
    pack(torch.randn(2, B, D_IN), member_idx).square().sum().backward()

    assert pack.weight.grad is not None and pack.bias is not None
    assert pack.bias.grad is not None
    selected = torch.zeros(K, dtype=torch.bool)
    selected[member_idx] = True
    for grad in (pack.weight.grad, pack.bias.grad):
        assert grad.shape[0] == K
        assert torch.all(grad[~selected] == 0)
        assert all(grad[k].abs().sum() > 0 for k in member_idx.tolist())


def test_init_statistics() -> None:
    torch.manual_seed(0)
    k, d_in, d_out = 8, 64, 48
    pack = LinearPack(d_in, d_out, pack_size=k)
    assert pack.bias is not None
    bound = 1 / d_in**0.5
    uniform_std = bound / 3**0.5
    weight = pack.weight.detach()
    bias = pack.bias.detach()

    assert weight.abs().max() <= bound
    assert bias.abs().max() <= bound
    for member in weight:
        # 3072 samples: the standard error of the mean is ~0.0013.
        assert member.mean().abs() < 0.01
        assert member.std().item() == pytest.approx(uniform_std, rel=0.05)
        # The samples fill the whole interval.
        assert member.abs().max() > 0.95 * bound
    # 48 samples per member, so only a loose check of the bias statistics.
    assert bias.mean().abs() < 0.05
    assert bias.std().item() == pytest.approx(uniform_std, rel=0.2)
    for a, b in itertools.combinations(range(k), 2):
        assert not torch.allclose(weight[a], weight[b])
        assert not torch.allclose(bias[a], bias[b])


def test_init_matches_nn_linear_distribution() -> None:
    # nn.Linear draws U(-1/sqrt(in), 1/sqrt(in)) for its (out, in) weight, then the
    # bias; for K=1 the pack consumes the RNG in the same order and layout.
    for d_in, d_out in [(1, 1), (13, 7), (64, 3)]:
        torch.manual_seed(123)
        reference = nn.Linear(d_in, d_out)
        torch.manual_seed(123)
        pack = LinearPack(d_in, d_out, pack_size=1)
        assert pack.bias is not None
        torch.testing.assert_close(pack.weight[0].T, reference.weight, atol=0, rtol=0)
        torch.testing.assert_close(pack.bias[0], reference.bias, atol=0, rtol=0)


def test_reset_parameters_redraws_in_place() -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    weight, bias = pack.weight, pack.bias
    assert bias is not None
    before = weight.detach().clone()
    pack.reset_parameters()
    assert pack.weight is weight and pack.bias is bias
    assert not torch.equal(before, pack.weight)
    assert pack.weight.abs().max() <= D_IN**-0.5


def test_bias_false() -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K, bias=False)
    assert pack.bias is None
    assert [name for name, _ in pack.named_parameters()] == ['weight']
    x = torch.randn(K, B, D_IN)
    torch.testing.assert_close(pack(x), x @ pack.weight, atol=1e-6, rtol=0)
    member_idx = torch.tensor([2])
    torch.testing.assert_close(
        pack(x[:1], member_idx), x[:1] @ pack.weight[2], atol=1e-6, rtol=0
    )
    assert 'bias=False' in repr(pack)


@pytest.mark.parametrize('bias', [True, False])
def test_pack_size_one(bias: bool) -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=1, bias=bias)
    assert pack.pack_size == 1
    assert pack.weight.shape == (1, D_IN, D_OUT)
    x = torch.randn(1, B, D_IN)
    assert pack(x).shape == (1, B, D_OUT)
    torch.testing.assert_close(pack(x, torch.tensor([0])), pack(x))
    (layer,) = _independent_linears(pack)
    torch.testing.assert_close(pack(x)[0], layer(x[0]), atol=1e-6, rtol=0)


def test_pack_size_is_derived_from_weight() -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    assert pack.bias is not None
    keep = torch.tensor([0, 2])
    expected = pack(torch.zeros(K, B, D_IN))[keep]
    # What pack_ops.pack_select_ does: slice the parameter data in place.
    pack.weight.data = pack.weight.data[keep]
    pack.bias.data = pack.bias.data[keep]
    assert pack.pack_size == 2
    assert 'pack_size=2' in repr(pack)
    torch.testing.assert_close(pack(torch.zeros(2, B, D_IN)), expected)


def test_expanded_input() -> None:
    # A (B, in) batch broadcast to all members without a copy (stride 0 on dim 0).
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    x2d = torch.randn(B, D_IN)
    x = x2d.expand(K, -1, -1)
    torch.testing.assert_close(pack(x), pack(x.contiguous()), atol=1e-6, rtol=0)


def test_autocast_bf16() -> None:
    torch.manual_seed(0)
    pack = LinearPack(16, 8, pack_size=K)
    x = torch.randn(K, B, 16)
    member_idx = torch.tensor([1, 3])
    expected = pack(x)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = pack(x)
        y_sub = pack(x[member_idx], member_idx)
    assert y.dtype == torch.bfloat16
    assert y_sub.dtype == torch.bfloat16
    torch.testing.assert_close(y.float(), expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        y_sub.float(), expected[member_idx], atol=5e-2, rtol=5e-2
    )
    # The parameters stay float32 and receive float32 gradients.
    y.float().sum().backward()
    assert pack.weight.dtype == torch.float32
    assert pack.weight.grad is not None
    assert pack.weight.grad.dtype == torch.float32


def test_float64() -> None:
    torch.manual_seed(0)
    pack = LinearPack(D_IN, D_OUT, pack_size=K).double()
    x = torch.randn(K, B, D_IN, dtype=torch.float64)
    y = pack(x)
    assert y.dtype == torch.float64
    assert pack.bias is not None
    torch.testing.assert_close(y, x @ pack.weight + pack.bias[:, None])


def test_extra_repr() -> None:
    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    assert pack.extra_repr() == (
        f'in_features={D_IN}, out_features={D_OUT}, pack_size={K}, bias=True'
    )
    assert repr(pack).startswith('LinearPack(')


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match='pack_size'):
        LinearPack(D_IN, D_OUT, pack_size=0)
    with pytest.raises(ValueError, match='in_features'):
        LinearPack(0, D_OUT, pack_size=K)
    with pytest.raises(ValueError, match='out_features'):
        LinearPack(D_IN, 0, pack_size=K)

    pack = LinearPack(D_IN, D_OUT, pack_size=K)
    with pytest.raises(ValueError):
        pack(torch.randn(B, D_IN))  # missing pack dimension
    with pytest.raises(ValueError):
        pack(torch.randn(K - 1, B, D_IN))  # wrong pack size
    with pytest.raises(ValueError):
        pack(torch.randn(K, B, D_IN), torch.tensor([0, 1]))  # K' != len(member_idx)
    with pytest.raises(ValueError):
        pack(torch.randn(K, B, D_IN + 1))  # wrong in_features
