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

Implementation notes
--------------------
* The pool order matters: greedy_ensemble breaks ties by the first maximum, so a
  snapshot already in the ensemble wins over an equally good new entry.
* The ensemble stores its own copies of the selected predictions (a snapshot), so
  later in-place changes of the caller's tensors never leak into it.
* Pool parts are the parts shared by every source that contributes at least one
  entry (e.g. the 'train' part of finished members is dropped when the running
  members only provide 'val' and 'test'). 'val' is required.
* The pool lives on the device of the newest source (running, else finished, else
  the current ensemble); every other source is moved there.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.ensembles.aggregate import average_predictions
from tabpack_repro.ensembles.greedy import greedy_ensemble
from tabpack_repro.metrics import ScoreFn, compute_metrics
from tabpack_repro.types import PartKey

_UPDATE_PART: PartKey = 'val'


class _Source(NamedTuple):
    name: str
    ids: np.ndarray
    steps: np.ndarray
    predictions: dict[PartKey, Tensor]


def _as_int64_1d(x: Any, name: str) -> np.ndarray:
    array = np.asarray(x.cpu() if isinstance(x, Tensor) else x)
    if array.ndim != 1:
        raise ValueError(f'{name} must be 1D, got shape {array.shape}')
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f'{name} must hold integers, got dtype {array.dtype}')
    return array.astype(np.int64, copy=True)


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu()
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        return x.numpy()
    return np.asarray(x)


class OnlineGreedyEnsemble:
    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        task: TaskInfo,
        max_ensemble_size: int | None,
        patience: int,
    ) -> None:
        if max_ensemble_size is not None and max_ensemble_size < 1:
            raise ValueError(
                f'max_ensemble_size must be None or >= 1, got {max_ensemble_size}'
            )
        if patience < 0:
            raise ValueError(f'patience must be >= 0, got {patience}')
        self._score_fn = score_fn
        self._task = task
        self._max_ensemble_size = max_ensemble_size
        self._patience = patience
        self._remaining_patience = patience

        # The current ensemble: one entry per selected pool entry.
        self._ids = np.zeros(0, dtype=np.int64)
        self._steps = np.zeros(0, dtype=np.int64)
        # part -> (E, N[, C]) snapshots of the selected entries' predictions.
        self._predictions: dict[PartKey, Tensor] = {}
        # part -> (N[, C]) uniform average of self._predictions[part].
        self._mean_predictions: dict[PartKey, Tensor] = {}
        self._score: float | None = None

    @property
    def is_running(self) -> bool:
        return self._remaining_patience >= 0

    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def steps(self) -> np.ndarray:
        return self._steps.copy()

    @property
    def score(self) -> float | None:
        """Current val score of the ensemble (None before the first update)."""
        return self._score

    def predictions(self) -> dict[PartKey, Tensor]:
        """part -> (N[, C]) averaged prediction of the current ensemble."""
        return dict(self._mean_predictions)

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
        if not self.is_running:
            raise RuntimeError(
                'the online ensemble has stopped (its patience is exhausted); '
                'check is_running before calling update()'
            )

        pool_ids, pool_steps, pool_predictions = self._prepare_pool(
            [
                _Source('current', self._ids, self._steps, self._predictions),
                _Source(
                    'finished',
                    _as_int64_1d(finished_ids, 'finished_ids'),
                    _as_int64_1d(finished_steps, 'finished_steps'),
                    finished_predictions,
                ),
                _Source(
                    'running',
                    _as_int64_1d(running_ids, 'running_ids'),
                    _as_int64_1d(running_steps, 'running_steps'),
                    running_predictions,
                ),
            ]
        )

        pool_val = pool_predictions[_UPDATE_PART]
        idx = greedy_ensemble(
            pool_val,
            score_fn=self._score_fn,
            max_ensemble_size=self._max_ensemble_size,
        ).to(device=pool_val.device, dtype=torch.long)
        candidate_val = average_predictions(pool_val[idx])
        score = float(self._score_fn(candidate_val[None]).item())
        improved = self._score is None or score > self._score

        if improved:
            idx_numpy = idx.cpu().numpy()
            self._ids = pool_ids[idx_numpy]
            self._steps = pool_steps[idx_numpy]
            # Advanced indexing copies, so the ensemble owns its snapshots.
            self._predictions = {k: v[idx] for k, v in pool_predictions.items()}
            self._mean_predictions = {
                k: candidate_val if k == _UPDATE_PART else average_predictions(v)
                for k, v in self._predictions.items()
            }
            self._score = score
            self._remaining_patience = self._patience
        else:
            self._remaining_patience -= 1
        return improved

    def _prepare_pool(
        self, sources: list[_Source]
    ) -> tuple[np.ndarray, np.ndarray, dict[PartKey, Tensor]]:
        """Concatenate the non-empty sources (in the given order) into one pool."""
        for source in sources:
            if len(source.ids) != len(source.steps):
                raise ValueError(
                    f'{source.name}: {len(source.ids)} ids but '
                    f'{len(source.steps)} steps'
                )
        sources = [s for s in sources if len(s.ids) > 0]
        if not sources:
            raise ValueError(
                'the pool is empty: no current ensemble, running or finished members'
            )

        parts = [
            k
            for k in sources[-1].predictions
            if all(k in s.predictions for s in sources)
        ]
        if _UPDATE_PART not in parts:
            raise ValueError(
                f'every non-empty source must provide {_UPDATE_PART!r} predictions; '
                f'got parts {[list(s.predictions) for s in sources]}'
            )
        for source in sources:
            for k in parts:
                if source.predictions[k].shape[0] != len(source.ids):
                    raise ValueError(
                        f'{source.name}: predictions[{k!r}] has '
                        f'{source.predictions[k].shape[0]} rows but there are '
                        f'{len(source.ids)} ids'
                    )

        device = sources[-1].predictions[_UPDATE_PART].device
        pool_ids = np.concatenate([s.ids for s in sources])
        pool_steps = np.concatenate([s.steps for s in sources])
        pool_predictions = {
            k: torch.cat([s.predictions[k].detach().to(device) for s in sources])
            for k in parts
        }
        return pool_ids, pool_steps, pool_predictions

    def report(self, y_true: dict[PartKey, np.ndarray]) -> dict[str, Any]:
        """{'ids', 'steps', 'size', 'n_unique', 'score_val', 'metrics': {part: ...}}
        with metrics from tabpack_repro.metrics.compute_metrics on each stored part."""
        metrics: dict[str, dict[str, float]] = {}
        for part, predictions in self._predictions.items():
            if part not in y_true:
                raise KeyError(
                    f'y_true has no labels for the stored part {part!r} '
                    f'(stored parts: {list(self._predictions)})'
                )
            # numpy average of the snapshots, like the official report.
            y_pred = average_predictions(_to_numpy(predictions))
            metrics[part] = compute_metrics(_to_numpy(y_true[part]), y_pred, self._task)
        return {
            'ids': self._ids.tolist(),
            'steps': self._steps.tolist(),
            'size': len(self._ids),
            'n_unique': len(np.unique(self._ids)),
            'score_val': self._score,
            'metrics': metrics,
        }
