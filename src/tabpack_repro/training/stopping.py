"""Early stopping and the pool of finished members (a22)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from tabpack_repro.types import PartKey


def compute_stop_idx(
    n_bad_updates: np.ndarray,
    steps: np.ndarray,
    *,
    patience: int,
    epoch_size: int,
    max_epochs: int,
) -> np.ndarray | None:
    """Indices of members to stop now, or None if none (official semantics).

    stop if (patience >= 0 and n_bad_updates > patience) or
            (max_epochs >= 0 and steps // epoch_size >= max_epochs).

    ``n_bad_updates`` and ``steps`` are per-member arrays of shape (K,). A negative
    ``patience`` disables early stopping and a negative ``max_epochs`` disables the
    epoch budget; with both disabled the result is always None. Returned indices are
    sorted int64 positions in the current pack.
    """
    n_bad_updates = np.asarray(n_bad_updates)
    steps = np.asarray(steps)
    if n_bad_updates.ndim != 1 or n_bad_updates.shape != steps.shape:
        raise ValueError(
            'n_bad_updates and steps must be 1-D arrays of the same shape, got '
            f'{n_bad_updates.shape} and {steps.shape}'
        )

    stop_mask = np.zeros(len(steps), dtype=bool)
    enabled = False
    if patience >= 0:
        stop_mask |= n_bad_updates > patience
        enabled = True
    if max_epochs >= 0:
        if epoch_size <= 0:
            raise ValueError(f'epoch_size must be positive, got {epoch_size}')
        stop_mask |= steps // epoch_size >= max_epochs
        enabled = True
    if not enabled:
        return None

    stop_idx = np.flatnonzero(stop_mask).astype(np.int64, copy=False)
    return stop_idx if len(stop_idx) > 0 else None


@dataclass
class FinishedPool:
    """Finished members in order of finishing: ids, best steps, best predictions."""

    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    predictions: dict[PartKey, Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def extend(
        self, ids: np.ndarray, steps: np.ndarray, predictions: dict[PartKey, Tensor]
    ) -> None:
        """Append members (predictions: part -> (M, N[, C]) on any device)."""
        ids = np.asarray(ids)
        steps = np.asarray(steps)
        if ids.ndim != 1 or ids.shape != steps.shape:
            raise ValueError(
                'ids and steps must be 1-D arrays of the same shape, got '
                f'{ids.shape} and {steps.shape}'
            )
        n_new = len(ids)
        for part, value in predictions.items():
            if value.ndim < 1 or value.shape[0] != n_new:
                raise ValueError(
                    f'predictions[{part!r}] must have {n_new} rows (one per member), '
                    f'got shape {tuple(value.shape)}'
                )
        started = len(self.ids) > 0 or bool(self.predictions)
        if started and set(predictions) != set(self.predictions):
            raise ValueError(
                'predictions must cover the same parts on every extend: expected '
                f'{sorted(self.predictions)}, got {sorted(predictions)}'
            )

        self.ids = np.concatenate([self.ids, ids.astype(np.int64, copy=False)])
        self.steps = np.concatenate([self.steps, steps.astype(np.int64, copy=False)])
        if self.predictions:
            self.predictions = {
                part: torch.cat([old, predictions[part]], dim=0)
                for part, old in self.predictions.items()
            }
        else:
            # First extend: adopt the tensors as they are (device and dtype kept).
            self.predictions = dict(predictions)
