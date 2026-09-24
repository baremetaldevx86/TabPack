"""Tests for DropoutPack and make_activation (a10)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tabpack_repro.nn.dropout_pack import DropoutPack, make_activation

# Per-member sample size of the statistical tests: B * D = 131072 elements, so the
# standard error of a keep rate is <= 0.0014 and that of the mean <= 0.0034 (p=0.6).
B, D = 256, 512


def _bits(x: torch.Tensor) -> torch.Tensor:
    """Raw bit pattern (distinguishes -0.0 from 0.0, unlike torch.equal)."""
    int_dtype = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }[x.dtype]
    return x.view(int_dtype)


def _special_input(k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = torch.randn(k, 64, 32).to(dtype)
    x[:, 0, :4] = torch.tensor([-0.0, float('inf'), float('-inf'), 1e30])
    return x


# ---------------------------------------------------------------------- construction


@pytest.mark.parametrize(
    ('p', 'expected'),
    [
        (0.25, [0.25] * 3),
        (0, [0.0] * 3),
        ([0.0, 0.1, 0.5], [0.0, 0.1, 0.5]),
        ((0.2, 0.0, 0.3), [0.2, 0.0, 0.3]),
        (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), [0.1, 0.2, 0.3]),
    ],
)
def test_buffer_p_float32_shape_k(p, expected) -> None:
    module = DropoutPack(p, pack_size=3)
    assert module.p.dtype == torch.float32
    assert module.p.shape == (3,)
    assert module.pack_size == 3
    torch.testing.assert_close(module.p, torch.tensor(expected, dtype=torch.float32))


def test_buffer_is_owned_copy() -> None:
    source = torch.tensor([0.1, 0.2])
    module = DropoutPack(source, pack_size=2)
    source.fill_(0.9)
    torch.testing.assert_close(module.p, torch.tensor([0.1, 0.2]))


def test_buffer_in_state_dict_and_round_trip() -> None:
    module = DropoutPack([0.0, 0.3, 0.6], pack_size=3)
    state = module.state_dict()
    assert list(state) == ['p']
    assert state['p'].shape == (3,)
    assert state['p'].dtype == torch.float32
    assert [name for name, _ in module.named_buffers()] == ['p']
    assert list(module.parameters()) == []

    other = DropoutPack(0.0, pack_size=3)
    other.load_state_dict(state)
    torch.testing.assert_close(other.p, module.p)


def test_pack_size_follows_buffer_after_in_place_selection() -> None:
    module = DropoutPack([0.1, 0.2, 0.3, 0.4], pack_size=4)
    module.p = module.p[torch.tensor([1, 3])]  # what pack_ops.pack_select_ does
    assert module.pack_size == 2
    assert 'p' in dict(module.named_buffers())
    torch.testing.assert_close(module.p, torch.tensor([0.2, 0.4]))
    x = torch.ones(2, 8, 4)
    assert module.eval()(x) is x
    with pytest.raises(ValueError, match='pack dimension'):
        module(torch.ones(4, 8, 4))


@pytest.mark.parametrize(
    'p',
    [
        -0.1,
        1.0,
        1.5,
        float('nan'),
        1.0 - 1e-9,  # rounds to 1.0 in float32
        [0.1, 1.0],
        [0.1, -0.01],
        [0.1, float('nan')],
    ],
)
def test_invalid_p_raises(p) -> None:
    with pytest.raises(ValueError, match='0 <= p < 1'):
        DropoutPack(p, pack_size=2)


@pytest.mark.parametrize('p', [[0.1], [0.1, 0.2, 0.3], [[0.1, 0.2]], []])
def test_wrong_p_length_raises(p) -> None:
    with pytest.raises(ValueError, match='pack_size=2'):
        DropoutPack(p, pack_size=2)


@pytest.mark.parametrize('pack_size', [0, -1])
def test_invalid_pack_size_raises(pack_size) -> None:
    with pytest.raises(ValueError, match='pack_size'):
        DropoutPack(0.1, pack_size=pack_size)


# ---------------------------------------------------------------------- identities


def test_eval_mode_is_identity() -> None:
    module = DropoutPack([0.0, 0.5, 0.9], pack_size=3).eval()
    x = _special_input(3)
    out = module(x)
    assert out is x
    x2 = x[:2]
    assert module(x2, torch.tensor([2, 1])) is x2


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_train_mode_p_zero_is_exact_identity(dtype) -> None:
    torch.manual_seed(0)
    module = DropoutPack(0.0, pack_size=3).train()
    x = _special_input(3, dtype)
    out = module(x)
    assert out.dtype == dtype
    assert torch.equal(_bits(out), _bits(x))


def test_train_mode_p_zero_members_exact_among_nonzero_members() -> None:
    torch.manual_seed(0)
    p = [0.0, 0.5, 0.0, 0.9]
    module = DropoutPack(p, pack_size=4).train()
    x = _special_input(4)
    out = module(x)
    for k in (0, 2):
        assert torch.equal(_bits(out[k]), _bits(x[k]))
    for k in (1, 3):
        assert not torch.equal(out[k], x[k])


def test_p_zero_member_gradient_is_exact_identity() -> None:
    torch.manual_seed(0)
    module = DropoutPack([0.0, 0.5], pack_size=2).train()
    x = torch.randn(2, 64, 16, requires_grad=True)
    out = module(x)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert torch.equal(x.grad[0], grad_out[0])
    # For the p=0.5 member, the gradient is the same mask-and-scale as the output.
    kept = out[1] != 0
    torch.testing.assert_close(x.grad[1][kept], 2.0 * grad_out[1][kept])
    assert torch.all(x.grad[1][~kept] == 0)


# ---------------------------------------------------------------------- statistics


def _check_statistics(out: torch.Tensor, x: torch.Tensor, p: list[float]) -> None:
    for k, p_k in enumerate(p):
        kept = out[k] != 0
        keep_rate = kept.float().mean().item()
        mean = out[k].float().mean().item()
        assert keep_rate == pytest.approx(1.0 - p_k, abs=0.01), (k, p_k)
        assert mean == pytest.approx(x[k].float().mean().item(), abs=0.03), (k, p_k)
        # Kept elements are scaled by exactly 1 / (1 - p_k).
        expected = x[k][kept].double() / (1.0 - p_k)
        torch.testing.assert_close(out[k][kept].double(), expected, rtol=1e-6, atol=0.0)


def test_keep_rate_and_mean_preservation_per_member() -> None:
    torch.manual_seed(0)
    p = [0.0, 0.1, 0.3, 0.6]
    module = DropoutPack(p, pack_size=len(p)).train()
    x = torch.ones(len(p), B, D)
    out = module(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    _check_statistics(out, x, p)


def test_masks_are_independent_across_members_and_calls() -> None:
    torch.manual_seed(0)
    module = DropoutPack(0.5, pack_size=2).train()
    x = torch.ones(2, B, D)
    a, b = module(x), module(x)
    # Two independent Bernoulli(0.5) masks agree on ~50% of the elements.
    assert (a[0] == a[1]).float().mean().item() == pytest.approx(0.5, abs=0.01)
    assert (a[0] == b[0]).float().mean().item() == pytest.approx(0.5, abs=0.01)


def test_same_seed_same_output() -> None:
    module = DropoutPack([0.1, 0.4], pack_size=2).train()
    x = torch.randn(2, 32, 8)
    torch.manual_seed(123)
    a = module(x)
    torch.manual_seed(123)
    b = module(x)
    assert torch.equal(a, b)


def test_arbitrary_trailing_dims() -> None:
    torch.manual_seed(0)
    module = DropoutPack([0.0, 0.5], pack_size=2).train()
    x = torch.ones(2, 16, 8, 4)
    out = module(x)
    assert out.shape == x.shape
    assert torch.equal(out[0], x[0])
    assert set(out[1].unique().tolist()) <= {0.0, 2.0}
    x1 = torch.ones(2)
    assert torch.equal(module(x1)[0], x1[0])


# ---------------------------------------------------------------------- member_idx


def test_member_idx_selects_rates() -> None:
    torch.manual_seed(0)
    p = [0.0, 0.5, 0.0, 0.1]
    module = DropoutPack(p, pack_size=4).train()

    # Only p=0 members selected: exact identity although the pack has p > 0.
    x = _special_input(2)
    out = module(x, torch.tensor([2, 0]))
    assert torch.equal(_bits(out), _bits(x))

    # Reordered with duplicates: statistics follow p[member_idx].
    idx = torch.tensor([3, 1, 1, 0])
    x = torch.ones(len(idx), B, D)
    out = module(x, idx)
    _check_statistics(out, x, [p[i] for i in idx.tolist()])


def test_member_idx_equals_full_pack_with_same_seed() -> None:
    p = [0.2, 0.0, 0.7]
    module = DropoutPack(p, pack_size=3).train()
    x = torch.randn(3, 16, 8)
    torch.manual_seed(7)
    full = module(x)
    torch.manual_seed(7)
    indexed = module(x, torch.arange(3))
    assert torch.equal(full, indexed)
    # A standalone pack with the selected rates behaves identically.
    idx = torch.tensor([2, 0])
    sub = DropoutPack([p[i] for i in idx.tolist()], pack_size=2).train()
    torch.manual_seed(7)
    a = module(x[idx], idx)
    torch.manual_seed(7)
    b = sub(x[idx])
    assert torch.equal(a, b)


def test_shape_mismatch_raises() -> None:
    module = DropoutPack([0.1, 0.2, 0.3], pack_size=3).train()
    with pytest.raises(ValueError, match='pack dimension'):
        module(torch.ones(2, 4, 4))
    with pytest.raises(ValueError, match='pack dimension'):
        module(torch.ones(3, 4, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='pack dimension'):
        module(torch.tensor(1.0))
    with pytest.raises(ValueError, match='pack dimension'):
        module.eval()(torch.ones(2, 4, 4))


# ---------------------------------------------------------------------- dtypes


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize('training', [True, False])
def test_dtype_preserved(dtype, training) -> None:
    torch.manual_seed(0)
    module = DropoutPack([0.0, 0.3], pack_size=2).train(training)
    x = torch.randn(2, 16, 8, dtype=dtype)
    out = module(x)
    assert out.dtype == dtype
    assert module.p.dtype == torch.float32


def test_bf16_statistics() -> None:
    torch.manual_seed(0)
    p = [0.0, 0.1, 0.3, 0.6]
    module = DropoutPack(p, pack_size=len(p)).train()
    x = torch.ones(len(p), B, D, dtype=torch.bfloat16)
    out = module(x)
    assert out.dtype == torch.bfloat16
    for k, p_k in enumerate(p):
        kept = out[k] != 0
        assert kept.float().mean().item() == pytest.approx(1.0 - p_k, abs=0.01)
        # The scale is applied in float32 and rounded once to bf16.
        expected = torch.tensor(1.0 / (1.0 - p_k)).to(torch.bfloat16)
        assert torch.all(out[k][kept] == expected)


def test_bf16_under_autocast() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(8, 16)
    module = DropoutPack([0.0, 0.5], pack_size=2).train()
    x = torch.randn(2, 32, 8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        h = linear(x)
        assert h.dtype == torch.bfloat16
        out = module(h)
        assert out.dtype == torch.bfloat16
        assert torch.equal(_bits(out[0]), _bits(h[0]))
        assert module.eval()(h) is h


def test_module_to_float64_keeps_working() -> None:
    torch.manual_seed(0)
    module = DropoutPack([0.0, 0.5], pack_size=2).to(torch.float64).train()
    assert module.p.dtype == torch.float64
    x = torch.ones(2, B, D, dtype=torch.float64)
    out = module(x)
    assert out.dtype == torch.float64
    assert torch.equal(out[0], x[0])
    assert set(out[1].unique().tolist()) <= {0.0, 2.0}


# ---------------------------------------------------------------------- activations


@pytest.mark.parametrize(
    ('name', 'cls'), [('ReLU', nn.ReLU), ('GELU', nn.GELU), ('SiLU', nn.SiLU)]
)
def test_make_activation(name, cls) -> None:
    a = make_activation(name)
    b = make_activation(name)
    assert type(a) is cls
    assert a is not b
    x = torch.linspace(-3, 3, 13)
    torch.testing.assert_close(a(x), cls()(x))


@pytest.mark.parametrize('name', ['relu', 'Tanh', 'LeakyReLU', '', 'Linear', None, 1])
def test_make_activation_unknown_raises(name) -> None:
    with pytest.raises(ValueError, match='Unknown activation'):
        make_activation(name)


def test_statistical_tolerances_are_loose_enough() -> None:
    # Guard for the constants above: the tolerances are >= 7 standard errors.
    n = B * D
    assert 0.01 >= 7 * math.sqrt(0.25 / n)
    assert 0.03 >= 7 * math.sqrt(0.6 / 0.4 / n)
