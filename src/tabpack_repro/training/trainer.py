"""The pack training loop (a23), shared by the homogeneous ensemble and TabPack.

One epoch = every running member sees every training row once, in its own order.
After each epoch: evaluate val/test for every running member, update PackState,
stop members by early stopping, move stopped members (at their best checkpoint) to
the FinishedPool, remove them from model/optimizer/state, then update the online
ensemble (if any). The loop ends when no member is running, or when the online
ensemble's patience is exhausted (official semantics; unfinished members are then
simply dropped).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.types import PartKey


@dataclass
class PackTrainResult:
    # Finished members, in order of finishing.
    ids: np.ndarray
    best_steps: np.ndarray
    # part -> (M, N[, C]) numpy predictions at each finished member's best epoch;
    # parts: train, val, test.
    predictions: dict[PartKey, np.ndarray]
    # One dict per finished member: {'id', 'best_step', 'metrics': {part: {...}}}.
    members: list[dict[str, Any]]
    # One dict per epoch: {'epoch', 'step', 'time', 'n_running', 'n_finished',
    #  'train_loss', 'ensemble_val', 'ensemble_test'} (ensemble_* None if no ensemble)
    history: list[dict[str, Any]] = field(default_factory=list)
    n_epochs: int = 0
    n_steps: int = 0
    time_sec: float = 0.0
    # Online ensemble final report (OnlineGreedyEnsemble.report()) or None.
    online_ensemble: dict[str, Any] | None = None


def train_pack(
    *,
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    dataset: PreparedDataset,
    batch_size: int,
    patience: int,
    max_epochs: int,
    seed: int,
    eval_batch_size: int = 32768,
    autocast: contextlib.AbstractContextManager | None = None,
    online_ensemble: OnlineGreedyEnsemble | None = None,
    progress: bool = False,
) -> PackTrainResult:
    """Train every member of `model` with early stopping; see module docstring.

    `dataset` must already be on the model's device. The batch generator is a
    torch.Generator on that device seeded with `seed`. Uses make_pack_loss, the loss
    of the pack is the SUM of per-member mean losses. Metrics of finished members
    are computed with tabpack_repro.metrics.compute_metrics on train/val/test.
    """
    raise NotImplementedError


def _unused(_: Tensor) -> None:  # pragma: no cover - keeps imports stable for stubs
    pass
