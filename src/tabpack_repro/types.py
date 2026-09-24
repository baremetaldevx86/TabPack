"""Shared type aliases, enums and tensor-layout conventions.

Tensor layout conventions used everywhere in this package
---------------------------------------------------------
* A *pack* is a set of ``K`` independent MLPs trained simultaneously.
* Every packed tensor has the pack dimension first: ``PACK_DIM = 0``.
* Activations inside a pack have shape ``(K, B, d)`` with ``BATCH_DIM = 1``.
* Every ``nn.Parameter`` and every buffer of a pack module (``tabpack_repro.nn``)
  has ``shape[0] == K``. This invariant is what makes member removal a simple
  slice along dim 0 (see ``tabpack_repro.nn.pack_ops``).
* Predictions of a pack on a data part of size ``N``:
  - binclass:   ``(K, N)`` probabilities of the positive class, float32;
  - multiclass: ``(K, N, C)`` probabilities, float32;
  - regression: ``(K, N)`` predictions in the original label units, float32.
"""

from __future__ import annotations

import enum
from typing import Literal

PACK_DIM = 0
BATCH_DIM = 1

PartKey = Literal['train', 'val', 'test']
PARTS: tuple[PartKey, ...] = ('train', 'val', 'test')


class TaskType(enum.StrEnum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'


class PredictionType(enum.StrEnum):
    """Units in which predictions are stored for ensembling.

    Raw logits are *not* aggregation-friendly, so classification predictions are
    stored as probabilities and regression predictions as labels.
    """

    PROBS = 'probs'
    LABELS = 'labels'
