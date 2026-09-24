"""Per-member losses (a19)."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from tabpack_repro.types import TaskType

# loss_fn(logits (K, B[, C]), y_true (K, B)) -> (K,) mean loss of each member.
PackLossFn = Callable[[Tensor, Tensor], Tensor]


def make_pack_loss(task_type: TaskType) -> PackLossFn:
    """binclass: BCE-with-logits (targets cast to logits dtype); multiclass: CE;
    regression: MSE. Reduction: mean over the batch, per member. The trainer sums the
    (K,) vector, so gradient scale is independent of K (official semantics)."""
    raise NotImplementedError
