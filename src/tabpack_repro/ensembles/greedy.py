"""Greedy ensemble selection (a25): Caruana et al. (2004), official variant."""

from __future__ import annotations

from torch import Tensor

from tabpack_repro.metrics import ScoreFn


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
    raise NotImplementedError
