"""Metrics (a08): numpy metrics for reports and batched torch scores for packs.

Binclass accuracy must use ``torch.round``/``np.round`` semantics on probabilities
(round half to even, so p=0.5 -> class 0), exactly like the official
``metrics_torch.calculate_metrics_pack``. This matters for greedy-selection parity.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo

# score_fn(predictions of shape (M, N[, C])) -> scores of shape (M,), higher is better.
ScoreFn = Callable[[Tensor], Tensor]

HIGHER_IS_BETTER = {'accuracy': True, 'roc-auc': True, 'log-loss': False, 'rmse': False}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task: TaskInfo
) -> dict[str, float]:
    """Metrics of ONE prediction vector (not a pack).

    binclass (y_pred = P(y=1), shape (N,)): {'accuracy', 'roc-auc', 'log-loss', 'score'}
    multiclass (y_pred (N, C) probs): {'accuracy', 'log-loss', 'score'}
    regression: {'rmse', 'score'} with score = -rmse.
    'score' is the task's main metric, sign-adjusted so that higher is better.
    """
    raise NotImplementedError


def score_pack(y_true: Tensor, y_pred: Tensor, task: TaskInfo) -> Tensor:
    """Main score for each of M predictions: y_pred (M, N[, C]) -> (M,) float32.

    Higher is better (negate error metrics). Must work on CPU and CUDA, no host sync
    beyond what is unavoidable.
    """
    raise NotImplementedError


def make_score_fn(y_true: Tensor, task: TaskInfo) -> ScoreFn:
    """Bind labels: returns ``lambda y_pred: score_pack(y_true, y_pred, task)``."""
    raise NotImplementedError
