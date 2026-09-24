"""Batched pack inference (a21).

Every member of a pack sees the same evaluation rows, so the inputs are passed to
the model as shared ``(B, f)`` slices of the dataset tensors (views, no copy); the
model expands them to ``(K, B, f)`` internally. Raw logits are converted to
aggregation-friendly float32 predictions (probabilities for classification, labels
for regression), see ``tabpack_repro.types``.

Differences from the official ``_evaluate`` (``project/tabpack.py``), all of which
leave the predictions unchanged:

* the previous train/eval mode of every submodule is restored afterwards (the
  official code leaves the model in eval mode and relies on the training loop to
  call ``model.train()``);
* rows are taken as contiguous slices instead of gathering them with an index tensor,
  and the batch outputs are written into one preallocated tensor instead of being
  concatenated (no 2x peak memory);
* on a GPU out-of-memory error, the batch size is halved and inference *resumes* from
  the failed batch (the official ``adjust_gpu_memory_usage`` restarts the whole
  evaluation); ``evaluate_pack`` keeps the reduced batch size for the next parts.
"""

from __future__ import annotations

import contextlib
import logging

import torch
from torch import Tensor, nn

from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.metrics import ScoreFn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.types import PartKey, TaskType

logger = logging.getLogger(__name__)

# Substrings of out-of-memory errors that are not raised as torch.OutOfMemoryError
# (same list as the official lib.utils.is_oom_exception).
_OOM_MESSAGES = (
    'CUDA out of memory',
    'CUBLAS_STATUS_ALLOC_FAILED',
    'CUDA error: out of memory',
)


def logits_to_predictions(logits: Tensor, task_type: str) -> Tensor:
    """binclass: sigmoid; multiclass: softmax(-1); regression: identity. float32."""
    task_type = TaskType(task_type)  # ValueError for an unknown task type.
    # Cast first: under autocast the logits may be bfloat16/float16.
    logits = logits.float()
    if task_type == TaskType.BINCLASS:
        return torch.sigmoid(logits)
    if task_type == TaskType.MULTICLASS:
        return torch.softmax(logits, dim=-1)
    return logits


def _is_oom_error(err: BaseException) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and any(m in str(err) for m in _OOM_MESSAGES)
    )


def _model_device(model: nn.Module, default: torch.device) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return default


def _slice(
    tensors: dict[PartKey, Tensor] | None,
    part: PartKey,
    start: int,
    end: int,
    device: torch.device,
) -> Tensor | None:
    if tensors is None:
        return None
    # A slice along dim 0 is a view; .to() is a no-op when already on `device`.
    return tensors[part][start:end].to(device)


def _predict_pack(
    model: nn.Module,
    dataset: PreparedDataset,
    part: PartKey,
    *,
    batch_size: int,
    autocast: contextlib.AbstractContextManager | None,
) -> tuple[Tensor, int]:
    """Inference loop of predict_pack; also returns the (possibly reduced) batch
    size that succeeded last, so that evaluate_pack can reuse it."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f'batch_size must be an int, got {batch_size!r}')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    n = dataset.size(part)
    device = _model_device(model, dataset.y[part].device)
    task_type = TaskType(dataset.task.type_)

    output: Tensor | None = None
    start = 0
    # `output is None` makes an empty part still run one (empty) batch, so that the
    # result has the right (K, 0[, C]) shape.
    while start < n or output is None:
        end = min(start + batch_size, n)
        try:
            x_num = _slice(dataset.x_num, part, start, end, device)
            x_cat = _slice(dataset.x_cat, part, start, end, device)
            with contextlib.nullcontext() if autocast is None else autocast:
                logits = model(x_num, x_cat)
            predictions = logits_to_predictions(logits, task_type)
            del logits, x_num, x_cat
        except RuntimeError as err:
            if not _is_oom_error(err) or batch_size == 1:
                raise
            # Retry outside of the `except` block: the traceback keeps the frames of
            # the failed forward (and their activations) alive until it is released.
        else:
            if output is None:
                output = predictions.new_empty(
                    (predictions.shape[0], n, *predictions.shape[2:])
                )
            output[:, start:end] = predictions
            del predictions
            start = end
            continue
        new_batch_size = batch_size // 2
        logger.warning(
            'Out of memory while predicting on %r with batch_size=%d;'
            ' retrying with batch_size=%d',
            part,
            batch_size,
            new_batch_size,
        )
        batch_size = new_batch_size
    return output, batch_size


@contextlib.contextmanager
def _eval_mode(model: nn.Module):
    """Switch every submodule to eval mode and restore each one's mode afterwards."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@torch.inference_mode()
def predict_pack(
    model: ModelPack,
    dataset: PreparedDataset,
    part: PartKey,
    *,
    batch_size: int = 32768,
    autocast: contextlib.AbstractContextManager | None = None,
) -> Tensor:
    """Predictions of all members on a part: (K, N[, C]) float32 on the model device.

    Puts the model in eval mode and restores the previous mode afterwards. On CUDA
    OOM, halve batch_size and retry (down to 1).
    """
    with _eval_mode(model):
        return _predict_pack(
            model, dataset, part, batch_size=batch_size, autocast=autocast
        )[0]


def evaluate_pack(
    model: ModelPack,
    dataset: PreparedDataset,
    parts: list[PartKey],
    score_fns: dict[PartKey, ScoreFn],
    *,
    batch_size: int = 32768,
    autocast: contextlib.AbstractContextManager | None = None,
) -> tuple[dict[PartKey, Tensor], dict[PartKey, Tensor]]:
    """Return (scores part -> (K,), predictions part -> (K, N[, C]))."""
    missing = [part for part in parts if part not in score_fns]
    if missing:
        raise KeyError(f'No score function for the parts {missing}')
    scores: dict[PartKey, Tensor] = {}
    predictions: dict[PartKey, Tensor] = {}
    with torch.inference_mode(), _eval_mode(model):
        for part in parts:
            predictions[part], batch_size = _predict_pack(
                model, dataset, part, batch_size=batch_size, autocast=autocast
            )
    # Outside of inference mode: the scores are ordinary tensors.
    for part in parts:
        scores[part] = score_fns[part](predictions[part])
    return scores, predictions
