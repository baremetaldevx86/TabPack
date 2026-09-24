"""Batched Newton-Schulz orthogonalization used by Muon (a14)."""

from __future__ import annotations

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
    raise NotImplementedError
