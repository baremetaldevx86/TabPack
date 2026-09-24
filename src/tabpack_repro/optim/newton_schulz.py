"""Batched Newton-Schulz orthogonalization used by Muon (a14)."""

from __future__ import annotations

import torch
from torch import Tensor

# Quintic iteration coefficients from Keller Jordan's Muon.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximately orthogonalize each matrix in ``G`` of shape ``(..., m, n)``.

    Semantics of the reference Muon implementation, batched over leading dims:
    compute in bfloat16; transpose so that rows <= cols; normalize each matrix by
    its Frobenius norm (+eps); run `steps` quintic iterations; transpose back; return
    in bfloat16 (the caller casts). Each matrix is processed independently, so the
    result for a batch equals the per-matrix results.
    """
    if G.ndim < 2:
        raise ValueError(
            f'G must have at least 2 dims (..., m, n), got shape {tuple(G.shape)}'
        )
    if steps < 0:
        raise ValueError(f'steps must be non-negative, got {steps}')
    a, b, c = NS_COEFFICIENTS

    # Iterate on the "wide" orientation so that the Gram matrix X X^T is the smaller
    # (min(m, n) x min(m, n)) one.
    tall = G.size(-2) > G.size(-1)
    X = G.to(torch.bfloat16)
    if tall:
        X = X.mT

    # Frobenius norm of every matrix separately (reduce over the last two dims only),
    # which bounds its spectral norm by 1. An all-zero matrix stays zero thanks to eps.
    X = X / (torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True) + eps)

    # Quintic step X <- a X + (b A + c A^2) X with A = X X^T; the scalar is applied
    # to A before the product, i.e. (c A) A, to reproduce the reference's bf16 rounding.
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + (c * A) @ A
        X = a * X + B @ X

    if tall:
        X = X.mT
    return X
