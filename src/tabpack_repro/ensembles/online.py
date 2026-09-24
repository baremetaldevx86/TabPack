"""Online greedy ensemble maintained during training (a26).

Official semantics (project/tabpack.py::OnlineEnsemble with update_type='latest',
include_current_ensemble_in_pool=True, update_part='val'):
* pool = [current ensemble members] + [finished members at their best epoch]
  + [running members at their LATEST epoch]; each pool entry is (id, step, preds)
  and the same id may appear more than once (e.g. an older snapshot kept in the
  current ensemble and its newer latest snapshot);
* run greedy_ensemble on the pool's val predictions; the candidate ensemble
  prediction is the uniform average of the selected entries;
* accept it only if its val score is strictly better than the current ensemble's
  (the first update is always accepted): then store the selected entries' ids,
  steps and predictions for all parts, reset patience; else patience -= 1;
* is_running is False once the remaining patience drops below 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.metrics import ScoreFn
from tabpack_repro.types import PartKey


class OnlineGreedyEnsemble:
    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        task: TaskInfo,
        max_ensemble_size: int | None,
        patience: int,
    ) -> None:
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        raise NotImplementedError

    @property
    def ids(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def steps(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def score(self) -> float | None:
        """Current val score of the ensemble (None before the first update)."""
        raise NotImplementedError

    def predictions(self) -> dict[PartKey, Tensor]:
        """part -> (N[, C]) averaged prediction of the current ensemble."""
        raise NotImplementedError

    def update(
        self,
        *,
        running_ids: np.ndarray,
        running_steps: np.ndarray,
        running_predictions: dict[PartKey, Tensor],
        finished_ids: np.ndarray,
        finished_steps: np.ndarray,
        finished_predictions: dict[PartKey, Tensor],
    ) -> bool:
        """Update from the current pool (predictions include at least 'val', 'test').

        Returns True iff the ensemble improved.
        """
        raise NotImplementedError

    def report(self, y_true: dict[PartKey, np.ndarray]) -> dict[str, Any]:
        """{'ids', 'steps', 'size', 'n_unique', 'score_val', 'metrics': {part: ...}}
        with metrics from tabpack_repro.metrics.compute_metrics on each stored part."""
        raise NotImplementedError
