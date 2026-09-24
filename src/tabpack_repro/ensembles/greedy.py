"""Greedy ensemble selection (a25): Caruana et al. (2004), official variant."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tabpack_repro.metrics import ScoreFn


def _validate_predictions(predictions: Tensor) -> None:
    if not isinstance(predictions, Tensor):
        raise TypeError(f'predictions must be a torch.Tensor, got {type(predictions)}')
    if not predictions.is_floating_point():
        raise TypeError(
            f'predictions must be a floating point tensor, got {predictions.dtype}'
        )
    if predictions.ndim < 1 or predictions.numel() == 0:
        raise ValueError(
            'predictions must be a non-empty tensor of shape (M, N[, C]), '
            f'got {tuple(predictions.shape)}'
        )
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError('predictions must be finite (found NaN or inf)')


def _score(score_fn: ScoreFn, predictions: Tensor) -> list[float]:
    """Scores of M predictions as host floats (one device sync per call).

    float32/float64 scores convert to Python floats exactly, so comparisons on the
    host are identical to comparisons of the score tensors.
    """
    scores = torch.as_tensor(score_fn(predictions))
    m = predictions.shape[0]
    if scores.shape != (m,):
        raise ValueError(
            f'score_fn must return scores of shape ({m},), got {tuple(scores.shape)}'
        )
    values = scores.tolist()
    if any(math.isnan(v) for v in values):
        # The official code would silently pick the NaN entry or fail an assertion.
        raise ValueError('score_fn returned NaN scores')
    return values


def _first_argmax(values: list[float], idx: list[int] | range) -> int:
    """The element of ``idx`` with the largest ``values[i]`` (the first one on ties)."""
    return max(idx, key=values.__getitem__)


def greedy_ensemble(
    predictions: Tensor,
    *,
    score_fn: ScoreFn,
    max_ensemble_size: int | None = None,
) -> Tensor:
    """Select members by greedy forward selection WITHOUT replacement.

    Official semantics (project/ensemble_utils_torch.py::greedy_ensemble, default
    options): start with argmax of individual scores (first max); repeatedly score
    every candidate ensemble ``mean * s/(s+1) + cand/(s+1)``; stop when the best
    candidate score is not strictly better than the current ensemble score, or the
    size limit is reached. Ties between best candidates are broken by the best
    individual score (first max). Returns sorted int64 indices into dim 0.
    """
    _validate_predictions(predictions)
    if max_ensemble_size is not None and (
        isinstance(max_ensemble_size, bool)
        or not isinstance(max_ensemble_size, int)
        or max_ensemble_size < 1
    ):
        raise ValueError(
            'max_ensemble_size must be None or a positive int, '
            f'got {max_ensemble_size!r}'
        )
    device = predictions.device
    n_predictions = predictions.shape[0]
    size_limit = (
        n_predictions
        if max_ensemble_size is None
        else min(n_predictions, max_ensemble_size)
    )

    individual_scores = _score(score_fn, predictions)
    first = _first_argmax(individual_scores, range(n_predictions))
    selected = [first]
    # Unselected indices in ascending order (the official candidate order).
    remaining = [i for i in range(n_predictions) if i != first]
    # The mean of a single prediction is the prediction itself (bit-exact).
    ensemble_prediction = predictions[first]
    ensemble_score = _score(score_fn, ensemble_prediction[None])[0]

    while len(selected) < size_limit:
        size = len(selected)
        candidate_idx = torch.tensor(remaining, dtype=torch.int64, device=device)
        # Incremental mean update, written exactly as in the official code so that the
        # candidate predictions (and hence the scores) are bit-identical.
        candidate_predictions = ensemble_prediction[None] * (
            size / (size + 1)
        ) + predictions[candidate_idx] * (1 / (size + 1))
        candidate_scores = _score(score_fn, candidate_predictions)

        best_score = max(candidate_scores)
        if best_score <= ensemble_score:
            # No candidate strictly improves the ensemble.
            break
        ties = [j for j, s in enumerate(candidate_scores) if s == best_score]
        best_local = (
            ties[0]
            if len(ties) == 1
            else max(ties, key=lambda j: individual_scores[remaining[j]])
        )

        selected.append(remaining.pop(best_local))
        ensemble_prediction = candidate_predictions[best_local]
        ensemble_score = candidate_scores[best_local]

    return torch.tensor(sorted(selected), dtype=torch.int64, device=device)
