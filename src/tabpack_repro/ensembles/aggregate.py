"""Prediction aggregation (a27)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def average_predictions(
    predictions: Tensor | np.ndarray, weights: Tensor | np.ndarray | None = None
) -> Tensor | np.ndarray:
    """Weighted mean over dim 0 of (M, N[, C]) -> (N[, C]); same array type as input.

    weights: (M,) non-negative, not all zero; normalized to sum to 1. None = uniform.

    Predictions must already be in aggregation-friendly units (probabilities for
    classification, raw values for regression; never logits). The result is float32
    (torch: on the input device). float64 inputs are reduced in float64 and then cast;
    lower-precision floats, integers and bools are cast to float32 first. Weights may
    be a numpy array or a tensor independently of ``predictions``.

    Raises TypeError for unsupported array types or complex dtypes, and ValueError for
    a bad ``predictions`` shape (ndim not 2/3, M == 0) or bad weights (shape != (M,),
    NaN/inf, negative entries, zero sum).
    """
    if isinstance(predictions, Tensor):
        return _average_torch(predictions, weights)
    if isinstance(predictions, np.ndarray):
        return _average_numpy(predictions, weights)
    raise TypeError(
        'predictions must be a torch.Tensor or np.ndarray, '
        f'got {type(predictions).__name__}'
    )


def _check_predictions_shape(shape: tuple[int, ...]) -> None:
    if len(shape) not in (2, 3):
        raise ValueError(
            f'predictions must have shape (M, N) or (M, N, C), got {tuple(shape)}'
        )
    if shape[0] == 0:
        raise ValueError(
            'predictions must contain at least one member (M >= 1), got M=0'
        )


def _check_weights_shape(shape: tuple[int, ...], n_members: int) -> None:
    if tuple(shape) != (n_members,):
        raise ValueError(
            f'weights must have shape (M,) = ({n_members},) to match predictions, '
            f'got {tuple(shape)}'
        )


def _raise_bad_weights(finite: bool, nonnegative: bool, total: float) -> None:
    if not finite:
        raise ValueError('weights must be finite (no NaN or inf)')
    if not nonnegative:
        raise ValueError('weights must be non-negative')
    if not total > 0:
        raise ValueError(f'weights must have a positive sum, got {total}')


def _average_numpy(
    predictions: np.ndarray, weights: Tensor | np.ndarray | None
) -> np.ndarray:
    _check_predictions_shape(predictions.shape)
    if np.iscomplexobj(predictions) or not (
        np.issubdtype(predictions.dtype, np.number) or predictions.dtype == np.bool_
    ):
        raise TypeError(f'predictions must have a real dtype, got {predictions.dtype}')
    compute_dtype = np.float64 if predictions.dtype == np.float64 else np.float32
    x = predictions.astype(compute_dtype, copy=False)

    if weights is None:
        out = x.mean(0)
    else:
        if isinstance(weights, Tensor):
            weights = weights.detach().cpu().numpy()
        w = np.asarray(weights)
        _check_weights_shape(w.shape, x.shape[0])
        if np.iscomplexobj(w):
            raise TypeError(f'weights must have a real dtype, got {w.dtype}')
        w = w.astype(compute_dtype, copy=False)
        total = w.sum()
        finite = bool(np.isfinite(w).all())
        nonnegative = bool((w >= 0).all())
        if not (finite and nonnegative and total > 0):
            _raise_bad_weights(finite, nonnegative, float(total))
        w = w / total
        out = (x * w.reshape((-1,) + (1,) * (x.ndim - 1))).sum(0)
    return np.asarray(out, dtype=np.float32)


def _average_torch(predictions: Tensor, weights: Tensor | np.ndarray | None) -> Tensor:
    _check_predictions_shape(tuple(predictions.shape))
    if predictions.is_complex():
        raise TypeError(f'predictions must have a real dtype, got {predictions.dtype}')
    compute_dtype = (
        torch.float64 if predictions.dtype == torch.float64 else torch.float32
    )
    x = predictions.to(compute_dtype)

    if weights is None:
        out = x.mean(0)
    else:
        w = torch.as_tensor(weights)
        _check_weights_shape(tuple(w.shape), x.shape[0])
        if w.is_complex():
            raise TypeError(f'weights must have a real dtype, got {w.dtype}')
        w = w.to(device=x.device, dtype=compute_dtype)
        total = w.sum()
        # One host sync for all three checks; details are computed only on failure.
        finite_t = torch.isfinite(w).all()
        nonnegative_t = (w >= 0).all()
        if not bool(finite_t & nonnegative_t & (total > 0)):
            _raise_bad_weights(bool(finite_t), bool(nonnegative_t), float(total))
        w = w / total
        out = (x * w.reshape((-1,) + (1,) * (x.ndim - 1))).sum(0)
    return out.to(torch.float32)
