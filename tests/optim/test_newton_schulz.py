"""Tests for the batched quintic Newton-Schulz orthogonalization (a14)."""

from __future__ import annotations

import pytest
import torch

from tabpack_repro.optim.newton_schulz import (
    NS_COEFFICIENTS,
    zeropower_via_newtonschulz5,
)

# bf16 has an 8-bit significand (relative spacing 2**-7), so two bf16 computations of
# the same O(1) quantity through different kernels may differ by a few ulps.
BF16_TOL = {'atol': 2e-2, 'rtol': 2e-2}

# Rectangular or small square Gaussian matrices: their smallest singular value is well
# away from zero, so five quintic steps push every singular value into ~[0.5, 1.5].
# (Large square Gaussian matrices have near-zero singular values that five steps
# cannot lift, which is inherent to Muon's iteration and not tested here.)
SHAPES = [
    (4, 16, 32),
    (4, 32, 16),
    (4, 8, 8),
    (8, 3, 5),
    (2, 128, 512),
    (4, 1, 16),
    (4, 16, 1),
]


def _randn(
    *shape: int, seed: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


def _svdvals(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x.float().cpu())


def test_coefficients() -> None:
    assert NS_COEFFICIENTS == (3.4445, -4.7750, 2.0315)


@pytest.mark.parametrize(
    'shape', [(5, 16, 32), (5, 32, 16), (3, 12, 12), (2, 3, 24, 10)]
)
def test_batched_equals_per_matrix(shape: tuple[int, ...]) -> None:
    G = _randn(*shape, seed=1)
    # Very different scales per member: a batch-wide norm would squash the small ones.
    scales = torch.logspace(-3, 3, shape[0]).view(-1, *[1] * (len(shape) - 1))
    G = G * scales
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == G.shape
    for idx in torch.cartesian_prod(*[torch.arange(n) for n in shape[:-2]]).view(
        -1, len(shape) - 2
    ):
        i = tuple(idx.tolist())
        torch.testing.assert_close(
            out[i], zeropower_via_newtonschulz5(G[i]), **BF16_TOL
        )


def test_per_matrix_normalization_is_scale_invariant() -> None:
    # Scaling a member by a power of two is exact in every op, so with per-matrix
    # normalization each member's output is bit-identical to the unscaled one.
    G = _randn(4, 16, 32, seed=2)
    scales = torch.tensor([2.0**-6, 1.0, 2.0**6, 2.0**15]).view(-1, 1, 1)
    assert torch.equal(
        zeropower_via_newtonschulz5(G * scales), zeropower_via_newtonschulz5(G)
    )


@pytest.mark.parametrize('shape', SHAPES)
def test_singular_values_roughly_one(shape: tuple[int, ...]) -> None:
    G = _randn(*shape, seed=3)
    s = _svdvals(zeropower_via_newtonschulz5(G))
    # The quintic iteration does not converge to exactly 1 (S' ~ Uniform(0.5, 1.5)).
    assert s.min() >= 0.5
    assert s.max() <= 1.5


@pytest.mark.parametrize('shape', SHAPES)
def test_approximates_polar_factor(shape: tuple[int, ...]) -> None:
    # The output keeps G's singular vectors: it is close in direction to U V^T.
    G = _randn(*shape, seed=4)
    out = zeropower_via_newtonschulz5(G).float()
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    polar = U @ Vh
    cos = (out * polar).sum((-2, -1)) / (
        torch.linalg.matrix_norm(out) * torch.linalg.matrix_norm(polar)
    )
    assert cos.min() > 0.95


@pytest.mark.parametrize('shape', [(3, 8, 20), (3, 20, 8), (2, 1, 9), (2, 9, 1)])
def test_wide_and_tall(shape: tuple[int, ...]) -> None:
    G = _randn(*shape, seed=5)
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == G.shape
    # A tall input is processed as its (wide) transpose, so the two agree exactly.
    assert torch.equal(zeropower_via_newtonschulz5(G.mT), out.mT)
    # The orthonormal side: rows (wide) or columns (tall) are nearly orthonormal.
    X = out.float()
    gram = X @ X.mT if shape[-2] <= shape[-1] else X.mT @ X
    eig = torch.linalg.eigvalsh(gram)
    assert eig.min() >= 0.25
    assert eig.max() <= 2.25


def test_pack_size_one_matches_2d() -> None:
    G = _randn(1, 12, 7, seed=6)
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == (1, 12, 7)
    torch.testing.assert_close(out[0], zeropower_via_newtonschulz5(G[0]), **BF16_TOL)


def test_zero_matrix_gives_zero_without_nan() -> None:
    G = _randn(3, 6, 10, seed=7)
    G[1] = 0.0
    out = zeropower_via_newtonschulz5(G)
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    # The zero member does not affect the others.
    torch.testing.assert_close(out[0], zeropower_via_newtonschulz5(G[0]), **BF16_TOL)
    torch.testing.assert_close(out[2], zeropower_via_newtonschulz5(G[2]), **BF16_TOL)
    assert torch.equal(
        zeropower_via_newtonschulz5(torch.zeros(4, 5)),
        torch.zeros(4, 5, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    'dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_output_is_bf16_and_input_untouched(dtype: torch.dtype) -> None:
    G = _randn(2, 8, 16, seed=8, dtype=dtype)
    G_before = G.clone()
    out = zeropower_via_newtonschulz5(G)
    assert out.dtype == torch.bfloat16
    assert out.shape == G.shape
    assert torch.equal(G, G_before)


def test_zero_steps_returns_frobenius_normalized_input() -> None:
    G = _randn(3, 8, 16, seed=9) * torch.tensor([0.01, 1.0, 100.0]).view(-1, 1, 1)
    out = zeropower_via_newtonschulz5(G, steps=0).float()
    torch.testing.assert_close(
        torch.linalg.matrix_norm(out), torch.ones(3), atol=1e-2, rtol=0
    )
    torch.testing.assert_close(
        out, G / torch.linalg.matrix_norm(G, keepdim=True), **BF16_TOL
    )


def test_more_steps_keep_singular_values_bounded() -> None:
    s = _svdvals(zeropower_via_newtonschulz5(_randn(2, 16, 32, seed=10), steps=10))
    assert s.min() >= 0.5
    assert s.max() <= 1.5


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(torch.randn(5))
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(torch.randn(3, 4), steps=-1)


@pytest.mark.gpu
def test_cuda(cuda_device: torch.device) -> None:
    G = _randn(4, 32, 64, seed=11) * torch.logspace(-2, 2, 4).view(-1, 1, 1)
    out = zeropower_via_newtonschulz5(G.to(cuda_device))
    assert out.device.type == 'cuda'
    assert out.dtype == torch.bfloat16
    assert out.shape == G.shape
    torch.testing.assert_close(out.cpu(), zeropower_via_newtonschulz5(G), **BF16_TOL)
    for i in range(G.shape[0]):
        torch.testing.assert_close(
            out[i], zeropower_via_newtonschulz5(G[i].to(cuda_device)), **BF16_TOL
        )
    s = _svdvals(out)
    assert s.min() >= 0.5
    assert s.max() <= 1.5
    tall = zeropower_via_newtonschulz5(G.mT.contiguous().to(cuda_device))
    assert tall.shape == G.mT.shape
    assert torch.isfinite(
        zeropower_via_newtonschulz5(torch.zeros(2, 3, 4, device=cuda_device))
    ).all()
