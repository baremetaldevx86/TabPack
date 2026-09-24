"""Batched pack inference (a21)."""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor

from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.metrics import ScoreFn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.types import PartKey


def logits_to_predictions(logits: Tensor, task_type: str) -> Tensor:
    """binclass: sigmoid; multiclass: softmax(-1); regression: identity. float32."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
