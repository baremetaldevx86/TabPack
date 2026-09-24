"""Prediction aggregation (a27)."""

from __future__ import annotations

import numpy as np
from torch import Tensor


def average_predictions(
    predictions: Tensor | np.ndarray, weights: Tensor | np.ndarray | None = None
) -> Tensor | np.ndarray:
    """Weighted mean over dim 0 of (M, N[, C]) -> (N[, C]); same array type as input.

    weights: (M,) non-negative, not all zero; normalized to sum to 1. None = uniform.
    """
    raise NotImplementedError
