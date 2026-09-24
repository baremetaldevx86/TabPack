"""Metrics (a08): numpy metrics for reports and batched torch scores for packs.

Binclass accuracy must use ``torch.round``/``np.round`` semantics on probabilities
(round half to even, so p=0.5 -> class 0), exactly like the official
``metrics_torch.calculate_metrics_pack``. This matters for greedy-selection parity.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import sklearn.metrics
import torch
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.types import TaskType

# score_fn(predictions of shape (M, N[, C])) -> scores of shape (M,), higher is better.
ScoreFn = Callable[[Tensor], Tensor]

HIGHER_IS_BETTER = {'accuracy': True, 'roc-auc': True, 'log-loss': False, 'rmse': False}

# Probabilities are clipped to [_EPS, 1 - _EPS] before taking logs. The value is the
# float32 machine epsilon, so that 1 - _EPS is still < 1 in float32 and the numpy and
# torch log-losses agree.
_EPS = float(np.finfo(np.float32).eps)

# Main metrics that each task type supports.
_SUPPORTED_SCORES = {
    TaskType.BINCLASS: ('accuracy', 'roc-auc', 'log-loss'),
    TaskType.MULTICLASS: ('accuracy', 'log-loss'),
    TaskType.REGRESSION: ('rmse',),
}


def _check_score(task: TaskInfo) -> TaskType:
    task_type = TaskType(task.type_)
    if task.score not in _SUPPORTED_SCORES[task_type]:
        raise ValueError(
            f'score={task.score!r} is not supported for {task_type.value} tasks; '
            f'expected one of {_SUPPORTED_SCORES[task_type]}'
        )
    return task_type


def _signed(name: str, value):
    return value if HIGHER_IS_BETTER[name] else -value


# ---------------------------------------------------------------------------
# numpy: one prediction vector
# ---------------------------------------------------------------------------


def _roc_auc_numpy(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        # ROC-AUC is undefined for a single class; sklearn raises here.
        return float('nan')
    return float(sklearn.metrics.roc_auc_score(y_true, probs))


def _log_loss_numpy(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    probs = np.clip(probs.astype(np.float64), _EPS, 1.0 - _EPS)
    with warnings.catch_warnings():
        # float32 softmax rows only sum to one up to float32 precision.
        warnings.filterwarnings('ignore', message='.*do not sum to one.*')
        return float(
            sklearn.metrics.log_loss(y_true, probs, labels=np.arange(n_classes))
        )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task: TaskInfo
) -> dict[str, float]:
    """Metrics of ONE prediction vector (not a pack).

    binclass (y_pred = P(y=1), shape (N,)): {'accuracy', 'roc-auc', 'log-loss', 'score'}
    multiclass (y_pred (N, C) probs): {'accuracy', 'log-loss', 'score'}
    regression: {'rmse', 'score'} with score = -rmse.
    'score' is the task's main metric, sign-adjusted so that higher is better.
    """
    task_type = _check_score(task)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1:
        raise ValueError(f'y_true must be 1D, got shape {y_true.shape}')
    expected_ndim = 2 if task_type == TaskType.MULTICLASS else 1
    if y_pred.ndim != expected_ndim or len(y_pred) != len(y_true):
        raise ValueError(
            f'{task_type.value}: y_pred must have shape '
            f'{"(N, C)" if expected_ndim == 2 else "(N,)"} with N={len(y_true)}, '
            f'got {y_pred.shape}'
        )

    result: dict[str, float]
    if task_type == TaskType.REGRESSION:
        diff = y_pred.astype(np.float64) - y_true.astype(np.float64)
        result = {'rmse': float(np.sqrt(np.mean(np.square(diff))))}
    else:
        y_true = y_true.astype(np.int64)
        if task_type == TaskType.BINCLASS:
            # np.round rounds half to even: p=0.5 -> class 0 (official semantics).
            labels = np.round(y_pred).astype(np.int64)
            result = {
                'accuracy': float(np.mean(labels == y_true)),
                'roc-auc': _roc_auc_numpy(y_true, y_pred),
                'log-loss': _log_loss_numpy(y_true, y_pred, 2),
            }
        else:
            labels = y_pred.argmax(axis=1)
            result = {
                'accuracy': float(np.mean(labels == y_true)),
                'log-loss': _log_loss_numpy(y_true, y_pred, y_pred.shape[1]),
            }
    result['score'] = _signed(task.score, result[task.score])
    return result


# ---------------------------------------------------------------------------
# torch: M predictions at once
# ---------------------------------------------------------------------------


def _roc_auc_pack(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Tie-aware ROC-AUC of each row of y_pred (M, N): Mann-Whitney U / (P * Q).

    Equals sklearn.metrics.roc_auc_score (ties get average ranks). The rank sums are
    exact int64 arithmetic; NaN when y_true has a single class.
    """
    n = y_true.shape[0]
    positive = y_true == 1
    sorted_pred = y_pred.sort(dim=1).values
    below = torch.searchsorted(sorted_pred, y_pred, side='left')
    below_or_equal = torch.searchsorted(sorted_pred, y_pred, side='right')
    # Twice the average 1-based rank: ties occupy ranks below+1 .. below_or_equal.
    twice_rank_sum = ((below + below_or_equal + 1) * positive).sum(dim=1)
    n_pos = positive.sum()
    n_neg = n - n_pos
    u_times_2 = twice_rank_sum - n_pos * (n_pos + 1)
    denominator = (2 * n_pos * n_neg).to(torch.float64)
    auc = u_times_2.to(torch.float64) / denominator
    return torch.where(denominator > 0, auc, torch.nan)


def score_pack(y_true: Tensor, y_pred: Tensor, task: TaskInfo) -> Tensor:
    """Main score for each of M predictions: y_pred (M, N[, C]) -> (M,) float32.

    Higher is better (negate error metrics). Must work on CPU and CUDA, no host sync
    beyond what is unavoidable.
    """
    task_type = _check_score(task)
    expected_ndim = 3 if task_type == TaskType.MULTICLASS else 2
    if y_true.ndim != 1:
        raise ValueError(f'y_true must be 1D, got shape {tuple(y_true.shape)}')
    if y_pred.ndim != expected_ndim or y_pred.shape[1] != y_true.shape[0]:
        raise ValueError(
            f'{task_type.value}: y_pred must have shape '
            f'{"(M, N, C)" if expected_ndim == 3 else "(M, N)"} with '
            f'N={y_true.shape[0]}, got {tuple(y_pred.shape)}'
        )
    # float16/bfloat16 (autocast) are upcast; float64 input keeps its precision.
    dtype = torch.float64 if y_pred.dtype == torch.float64 else torch.float32
    y_pred = y_pred.to(dtype).contiguous()
    y_true = y_true.to(y_pred.device)
    n = y_true.shape[0]

    if task_type == TaskType.REGRESSION:
        mse = (y_pred - y_true.to(dtype)).square().mean(dim=1)
        value = mse.sqrt()
    elif task.score == 'accuracy':
        if task_type == TaskType.BINCLASS:
            # torch.round rounds half to even: p=0.5 -> class 0 (official semantics).
            correct = y_pred.round() == y_true.to(dtype)
        else:
            correct = y_pred.argmax(dim=-1) == y_true.long()
        # The count is exact in float32 (N < 2**24), so this equals the official
        # ``.float().mean(1)``.
        value = correct.sum(dim=1).to(torch.float32) / n
    elif task.score == 'roc-auc':
        value = _roc_auc_pack(y_true.long(), y_pred)
    else:  # log-loss
        probs = y_pred.clamp(_EPS, 1.0 - _EPS)
        if task_type == TaskType.BINCLASS:
            positive = (y_true == 1).expand_as(probs)
            log_likelihood = torch.where(positive, probs.log(), (-probs).log1p())
        else:
            index = y_true.long()[None, :, None].expand(probs.shape[0], n, 1)
            log_likelihood = probs.gather(-1, index).squeeze(-1).log()
        value = -log_likelihood.mean(dim=1)
    return _signed(task.score, value).to(torch.float32)


def make_score_fn(y_true: Tensor, task: TaskInfo) -> ScoreFn:
    """Bind labels: returns ``lambda y_pred: score_pack(y_true, y_pred, task)``."""
    _check_score(task)
    if y_true.ndim != 1:
        raise ValueError(f'y_true must be 1D, got shape {tuple(y_true.shape)}')
    if task.is_classification:
        y_true = y_true.long()

    def score_fn(y_pred: Tensor) -> Tensor:
        return score_pack(y_true, y_pred, task)

    return score_fn
