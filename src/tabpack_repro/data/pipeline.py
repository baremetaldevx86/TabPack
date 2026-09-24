"""End-to-end dataset preparation (a07): raw arrays -> model-ready torch tensors."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tabpack_repro.config import DataConfig
from tabpack_repro.data.categorical import (
    bin_to_cat,
    extract_bin_from_num,
    merge_bin,
    ordinal_encode,
)
from tabpack_repro.data.dataset import TaskInfo, load_raw_dataset
from tabpack_repro.data.download import get_data_dir
from tabpack_repro.data.numerical import (
    drop_constant_columns,
    noisy_quantile_transform,
    standard_transform,
)
from tabpack_repro.types import PartKey

# Policies this pipeline can execute. The official code also knows cat_policy
# 'one-hot', but PreparedDataset.x_cat holds ordinal codes by contract (one-hot
# encoding happens inside the model), so it is rejected here.
_NUM_POLICIES: tuple[str | None, ...] = (None, 'noisy-quantile', 'standard')
_BIN_POLICIES: tuple[str | None, ...] = (None, 'convert-to-cat')
_CAT_POLICIES: tuple[str | None, ...] = (None, 'ordinal')

_Parts = dict[PartKey, np.ndarray]


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

        def move(parts: dict[PartKey, Tensor] | None) -> dict[PartKey, Tensor] | None:
            if parts is None:
                return None
            return {part: value.to(device) for part, value in parts.items()}

        y = move(self.y)
        assert y is not None
        return dataclasses.replace(
            self,
            x_num=move(self.x_num),
            x_cat=move(self.x_cat),
            y=y,
            cat_cardinalities=list(self.cat_cardinalities),
        )


def build_dataset(config: DataConfig) -> PreparedDataset:
    """Load + preprocess following the official order (see data/categorical.py).

    Steps: load_raw_dataset -> extract_bin_from_num (if enabled) ->
    numerical transform (num_policy, seed=config.seed) -> bin_policy -> cat_policy
    -> convert to CPU torch tensors. Regression label standardization is out of scope
    (Churn is binclass) and must raise NotImplementedError for regression tasks.

    Details (a07):
    * Policies are validated before any data is read: an unknown ``num_policy``,
      ``bin_policy`` or ``cat_policy`` raises ValueError.
    * A relative ``config.path`` is resolved against ``get_data_dir()``; an
      absolute one (after ``~`` expansion) is used as is.
    * ``num_policy=None`` still applies the official post-steps (NaN -> 0, drop
      columns constant on train, float32). If no numerical column survives, x_num
      becomes None.
    * Binary features that are left unconverted (``bin_policy=None``) cannot be
      represented by PreparedDataset (the official trainer asserts there are none),
      so they raise ValueError.
    * ``cat_policy=None`` is accepted only for integer-valued x_cat (e.g. binary
      features converted to categories without original x_cat); cardinalities are
      then the numbers of unique train values, as in the official code.
    """
    _validate_policies(config)
    raw = load_raw_dataset(_resolve_path(config.path))
    if raw.task.is_regression:
        raise NotImplementedError(
            'Regression datasets are not supported (label standardization is out'
            ' of scope for this reproduction)'
        )

    x_num, x_bin, x_cat = raw.x_num, raw.x_bin, raw.x_cat

    # 1. Move numerical columns with exactly two values to the binary features.
    if x_num is not None and config.extract_bin_from_num:
        extracted_x_bin, remaining_x_num = extract_bin_from_num(x_num)
        if extracted_x_bin is not None:
            x_num = remaining_x_num
            x_bin = merge_bin(extracted_x_bin, x_bin)

    # 2. Numerical features (checked again: step 1 may have consumed all of them).
    if x_num is not None:
        x_num = _transform_num(x_num, config.num_policy, config.seed)
        if x_num['train'].shape[1] == 0:
            x_num = None

    # 3. Binary features.
    if x_bin is not None:
        if config.bin_policy is None:
            raise ValueError(
                'The dataset has binary features, but bin_policy=None leaves them'
                " unconverted; use bin_policy='convert-to-cat'"
            )
        x_cat = bin_to_cat(x_bin, x_cat)

    # 4. Categorical features.
    cat_cardinalities: list[int] = []
    if x_cat is not None:
        if config.cat_policy is None:
            x_cat, cat_cardinalities = _keep_integer_cat(x_cat)
        else:
            x_cat, cat_cardinalities = ordinal_encode(x_cat)

    return PreparedDataset(
        x_num=None if x_num is None else _to_tensors(x_num, np.float32),
        x_cat=None if x_cat is None else _to_tensors(x_cat, np.int64),
        y=_to_tensors(raw.y, np.int64),
        task=raw.task,
        cat_cardinalities=[int(c) for c in cat_cardinalities],
    )


def _validate_policies(config: DataConfig) -> None:
    for name, value, allowed in (
        ('num_policy', config.num_policy, _NUM_POLICIES),
        ('bin_policy', config.bin_policy, _BIN_POLICIES),
        ('cat_policy', config.cat_policy, _CAT_POLICIES),
    ):
        if value not in allowed:
            raise ValueError(f'Unknown {name}={value!r}; expected one of {allowed}')


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else get_data_dir() / path


def _transform_num(x_num: _Parts, policy: str | None, seed: int) -> _Parts:
    if policy == 'noisy-quantile':
        return noisy_quantile_transform(x_num, seed=seed)
    if policy == 'standard':
        return standard_transform(x_num)
    assert policy is None, policy  # guaranteed by _validate_policies
    # Official transform_num without a normalizer: only the post-steps.
    x_num = {part: np.nan_to_num(value) for part, value in x_num.items()}
    x_num = drop_constant_columns(x_num)
    return {part: value.astype(np.float32) for part, value in x_num.items()}


def _keep_integer_cat(x_cat: _Parts) -> tuple[_Parts, list[int]]:
    if not all(np.issubdtype(value.dtype, np.integer) for value in x_cat.values()):
        raise ValueError(
            "cat_policy=None requires integer categorical features; use cat_policy='"
            "ordinal' to encode non-integer (e.g. string) categories"
        )
    cardinalities = [len(np.unique(column)) for column in x_cat['train'].T]
    return x_cat, cardinalities


def _to_tensors(parts: _Parts, dtype: type[np.generic]) -> dict[PartKey, Tensor]:
    # ascontiguousarray may return the input itself; copy read-only arrays so that
    # torch.from_numpy never shares memory it would treat as writable.
    result: dict[PartKey, Tensor] = {}
    for part, value in parts.items():
        array = np.ascontiguousarray(value, dtype=dtype)
        if not array.flags.writeable:
            array = array.copy()
        result[part] = torch.from_numpy(array)
    return result
