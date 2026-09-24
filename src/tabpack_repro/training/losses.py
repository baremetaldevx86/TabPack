"""Per-member losses (a19)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from tabpack_repro.types import BATCH_DIM, PACK_DIM, TaskType

# loss_fn(logits (K, B[, C]), y_true (K, B)) -> (K,) mean loss of each member.
PackLossFn = Callable[[Tensor, Tensor], Tensor]


def _check_shapes(task_type: TaskType, logits: Tensor, y_true: Tensor) -> None:
    expected_ndim = 3 if task_type == TaskType.MULTICLASS else 2
    if logits.ndim != expected_ndim:
        raise ValueError(
            f'{task_type} logits must have {expected_ndim} dims '
            f'({"(K, B, C)" if expected_ndim == 3 else "(K, B)"}), '
            f'got shape {tuple(logits.shape)}'
        )
    if y_true.shape != logits.shape[:2]:
        raise ValueError(
            f'y_true must have shape (K, B) = {tuple(logits.shape[:2])}, '
            f'got {tuple(y_true.shape)}'
        )


def make_pack_loss(task_type: TaskType) -> PackLossFn:
    """binclass: BCE-with-logits (targets cast to logits dtype); multiclass: CE;
    regression: MSE. Reduction: mean over the batch, per member. The trainer sums the
    (K,) vector, so gradient scale is independent of K (official semantics)."""
    try:
        task_type = TaskType(task_type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise ValueError(
            f'Unknown task type {task_type!r}; expected one of {valid}'
        ) from None

    def loss_fn(logits: Tensor, y_true: Tensor) -> Tensor:
        _check_shapes(task_type, logits, y_true)
        k, b = logits.shape[PACK_DIM], logits.shape[BATCH_DIM]
        # Flatten (K, B) -> (K * B,), compute unreduced per-object losses, then take
        # the mean over the batch of each member. The flattening makes every loss
        # term depend only on its own member's logits, so members stay independent.
        flat_logits = logits.flatten(0, 1)
        flat_y = y_true.flatten(0, 1)
        if task_type == TaskType.BINCLASS:
            # Numerically stable (log-sum-exp) BCE on raw logits; labels arrive as
            # int64 class ids and must share the logits' floating dtype.
            losses = F.binary_cross_entropy_with_logits(
                flat_logits, flat_y.to(flat_logits.dtype), reduction='none'
            )
        elif task_type == TaskType.MULTICLASS:
            losses = F.cross_entropy(
                flat_logits, flat_y.to(torch.long), reduction='none'
            )
        else:
            losses = F.mse_loss(
                flat_logits, flat_y.to(flat_logits.dtype), reduction='none'
            )
        return losses.reshape(k, b).mean(BATCH_DIM)

    return loss_fn
