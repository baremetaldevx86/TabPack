"""Early stopping and the pool of finished members (a22)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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
    """
    raise NotImplementedError


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
        raise NotImplementedError
