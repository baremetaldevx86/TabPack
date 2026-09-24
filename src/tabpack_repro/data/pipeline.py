"""End-to-end dataset preparation (a07): raw arrays -> model-ready torch tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tabpack_repro.config import DataConfig
from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.types import PartKey


@dataclass(kw_only=True)
class PreparedDataset:
    """Model-ready data.

    * x_num: float32 ``(N, n_num)`` per part, or None.
    * x_cat: int64 ``(N, n_cat)`` ordinal codes per part, or None.
    * y: int64 ``(N,)`` for classification, float32 ``(N,)`` for regression.
    * cat_cardinalities: one entry per x_cat column ([] when x_cat is None).
    """

    x_num: dict[PartKey, Tensor] | None
    x_cat: dict[PartKey, Tensor] | None
    y: dict[PartKey, Tensor]
    task: TaskInfo
    cat_cardinalities: list[int]

    @property
    def n_num_features(self) -> int:
        return 0 if self.x_num is None else int(self.x_num['train'].shape[1])

    @property
    def n_cat_features(self) -> int:
        return len(self.cat_cardinalities)

    def size(self, part: PartKey) -> int:
        return int(self.y[part].shape[0])

    def to(self, device: torch.device | str) -> PreparedDataset:
        """Return a copy with every tensor moved to `device`."""
        raise NotImplementedError


def build_dataset(config: DataConfig) -> PreparedDataset:
    """Load + preprocess following the official order (see data/categorical.py).

    Steps: load_raw_dataset -> extract_bin_from_num (if enabled) ->
    numerical transform (num_policy, seed=config.seed) -> bin_policy -> cat_policy
    -> convert to CPU torch tensors. Regression label standardization is out of scope
    (Churn is binclass) and must raise NotImplementedError for regression tasks.
    """
    raise NotImplementedError
