"""Raw dataset loading (a04).

On-disk format (official): ``info.json`` with ``{"task": {"type": ..., "score": ...}}``,
``x_num.npy`` (float32), ``x_bin.npy`` (float32 0/1), ``x_cat.npy`` (str),
``y.npy`` (int64 for classification), and ``splits/<split>/{train,val,test}.npy``
holding integer row indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabpack_repro.types import PartKey, TaskType


@dataclass(frozen=True, kw_only=True)
class TaskInfo:
    type_: TaskType
    # Name of the main metric, e.g. 'accuracy' (higher is better) or 'rmse'.
    score: str
    # Number of classes for classification tasks, None for regression.
    n_classes: int | None = None

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_classification(self) -> bool:
        return not self.is_regression


@dataclass(kw_only=True)
class RawDataset:
    """Unprocessed arrays, already split into parts. Missing feature kinds are None."""

    x_num: dict[PartKey, np.ndarray] | None
    x_bin: dict[PartKey, np.ndarray] | None
    x_cat: dict[PartKey, np.ndarray] | None
    y: dict[PartKey, np.ndarray]
    task: TaskInfo

    def size(self, part: PartKey) -> int:
        return len(self.y[part])


def load_raw_dataset(path: str | Path, split: str = 'default') -> RawDataset:
    """Load a dataset directory and apply the split (a04).

    * n_classes is inferred from the train labels for classification tasks.
    * Arrays are copied (fancy indexing), dtypes preserved.
    * Raise FileNotFoundError with a helpful message (mention
      ``tabpack-repro download``) when the directory is missing.
    """
    raise NotImplementedError
