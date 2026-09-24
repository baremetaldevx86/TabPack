"""Raw dataset loading (a04).

On-disk format (official): ``info.json`` with ``{"task": {"type": ..., "score": ...}}``,
``x_num.npy`` (float32), ``x_bin.npy`` (float32 0/1), ``x_cat.npy`` (str),
``y.npy`` (int64 for classification), and ``splits/<split>/{train,val,test}.npy``
holding integer row indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tabpack_repro.types import PARTS, PartKey, TaskType


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


# Allowed numpy dtype kinds per array (the official loader asserts float32 for
# x_num/x_bin, int64 or str for x_cat, int64/float32 for classification/regression
# labels; we check kinds and keep the stored dtypes untouched).
_FEATURE_KINDS = {
    'x_num': ('f',),
    'x_bin': ('f', 'b', 'i', 'u'),
    'x_cat': ('U', 'i', 'u'),
}


def _download_hint(path: Path) -> str:
    return (
        f'run `tabpack-repro download --name {path.name}'
        f' --data-dir {path.parent}` to fetch it'
    )


def _load_npy(path: Path) -> np.ndarray:
    # allow_pickle=False: x_cat is a fixed-width unicode array, never an object one.
    return np.load(path, allow_pickle=False)


def _load_task_info(path: Path) -> tuple[TaskType, str]:
    info_path = path / 'info.json'
    if not info_path.is_file():
        raise FileNotFoundError(
            f'The dataset directory {path} has no info.json (incomplete download?);'
            f' {_download_hint(path)}'
        )
    info: Any = json.loads(info_path.read_text())
    task = info.get('task') if isinstance(info, dict) else None
    if not isinstance(task, dict) or 'type' not in task or 'score' not in task:
        raise ValueError(
            f'{info_path} must contain {{"task": {{"type": ..., "score": ...}}}},'
            f' got: {info!r}'
        )
    try:
        task_type = TaskType(task['type'])
    except ValueError:
        raise ValueError(
            f'Unknown task type {task["type"]!r} in {info_path};'
            f' expected one of {[t.value for t in TaskType]}'
        ) from None
    score = task['score']
    if not isinstance(score, str) or not score:
        raise ValueError(f'The task score in {info_path} must be a non-empty string')
    return task_type, score


def _load_split(path: Path, split: str, n_rows: int) -> dict[PartKey, np.ndarray]:
    """Read ``splits/<split>`` and validate the index arrays.

    Like the official ``load_split``, a nested split id such as ``'a/0'`` collects
    parts from every directory on the way (``splits/a/test.npy`` +
    ``splits/a/0/{train,val}.npy``); a part may be defined only once.
    """
    names = [x for x in split.split('/') if x]
    if not names or any(x in ('.', '..') for x in names):
        raise ValueError(f'Invalid split name: {split!r}')

    splits_dir = path / 'splits'
    files: dict[str, Path] = {}
    directory = splits_dir
    for name in names:
        directory = directory / name
        if not directory.is_dir():
            available = (
                sorted(x.name for x in splits_dir.iterdir() if x.is_dir())
                if splits_dir.is_dir()
                else []
            )
            raise FileNotFoundError(
                f'The split {split!r} does not exist in {path}'
                f' (missing directory {directory}; available splits: {available});'
                f' if the download is incomplete, {_download_hint(path)}'
            )
        for file in sorted(directory.iterdir()):
            # `._*` files are macOS AppleDouble junk shipped in the official archive.
            if file.suffix != '.npy' or file.is_dir() or file.name.startswith('._'):
                continue
            if file.stem in files:
                raise ValueError(
                    f'The part {file.stem!r} is defined more than once'
                    f' in the split {split!r} of {path}'
                )
            files[file.stem] = file

    missing = [p for p in PARTS if p not in files]
    unknown = sorted(set(files) - set(PARTS))
    if missing or unknown:
        raise ValueError(
            f'The split {split!r} of {path} must consist of exactly the parts'
            f' {list(PARTS)}; missing: {missing}, unexpected: {unknown}'
        )

    parts: dict[PartKey, np.ndarray] = {}
    for part in PARTS:
        idx = _load_npy(files[part])
        if idx.ndim != 1 or idx.dtype.kind not in 'iu':
            raise ValueError(
                f'Split indices must be a 1D integer array; {files[part]} has'
                f' shape {idx.shape} and dtype {idx.dtype}'
            )
        if idx.size == 0:
            raise ValueError(f'The part {part!r} of the split {split!r} is empty')
        if idx.min() < 0 or idx.max() >= n_rows:
            raise ValueError(
                f'The indices of the part {part!r} of the split {split!r} are out'
                f' of range [0, {n_rows}): min={idx.min()}, max={idx.max()}'
            )
        if len(np.unique(idx)) != len(idx):
            raise ValueError(
                f'The part {part!r} of the split {split!r} contains duplicate indices'
            )
        parts[part] = idx

    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1 :]:
            n_common = len(np.intersect1d(parts[a], parts[b]))
            if n_common:
                raise ValueError(
                    f'The parts {a!r} and {b!r} of the split {split!r} overlap'
                    f' ({n_common} shared indices); parts must be disjoint'
                )
    return parts


def _infer_n_classes(task_type: TaskType, y: dict[PartKey, np.ndarray]) -> int | None:
    if task_type == TaskType.REGRESSION:
        return None
    classes = np.unique(y['train'])
    n_classes = len(classes)
    # Labels are expected to be encoded as 0..n_classes-1 (as in the official data),
    # since they are used directly as class indices by the losses and metrics.
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(
            f'Classification labels must be encoded as 0..C-1 in the train part;'
            f' found classes {classes.tolist()}'
        )
    for part in PARTS:
        extra = np.setdiff1d(y[part], classes)
        if extra.size:
            raise ValueError(
                f'The {part!r} part has labels absent from the train part:'
                f' {extra.tolist()}'
            )
    if task_type == TaskType.BINCLASS and n_classes != 2:
        raise ValueError(
            f'A binclass task needs exactly 2 classes in the train part,'
            f' found {n_classes}'
        )
    if task_type == TaskType.MULTICLASS and n_classes < 2:
        raise ValueError(
            f'A multiclass task needs at least 2 classes in the train part,'
            f' found {n_classes}'
        )
    return n_classes


def load_raw_dataset(path: str | Path, split: str = 'default') -> RawDataset:
    """Load a dataset directory and apply the split (a04).

    * n_classes is inferred from the train labels for classification tasks.
    * Arrays are copied (fancy indexing), dtypes preserved.
    * Raise FileNotFoundError with a helpful message (mention
      ``tabpack-repro download``) when the directory is missing.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f'The dataset directory does not exist: {path}; {_download_hint(path)}'
        )
    task_type, score = _load_task_info(path)

    y_path = path / 'y.npy'
    if not y_path.is_file():
        raise FileNotFoundError(
            f'The dataset directory {path} has no y.npy (incomplete download?);'
            f' {_download_hint(path)}'
        )
    y_all = _load_npy(y_path)
    if y_all.ndim != 1:
        raise ValueError(f'y.npy must be 1D, got shape {y_all.shape}')
    y_kinds = 'f' if task_type == TaskType.REGRESSION else 'iu'
    if y_all.dtype.kind not in y_kinds:
        raise ValueError(
            f'Labels of a {task_type.value} task must have a'
            f' {"floating" if task_type == TaskType.REGRESSION else "integer"}'
            f' dtype, got {y_all.dtype}'
        )
    n_rows = len(y_all)

    features: dict[str, np.ndarray | None] = {}
    for key, kinds in _FEATURE_KINDS.items():
        file = path / f'{key}.npy'
        if not file.is_file():
            features[key] = None
            continue
        x = _load_npy(file)
        if x.ndim != 2 or x.shape[0] != n_rows:
            raise ValueError(
                f'{file.name} must have shape ({n_rows}, n_features), got {x.shape}'
            )
        if x.dtype.kind not in kinds:
            raise ValueError(f'Unsupported dtype of {file.name}: {x.dtype}')
        features[key] = x

    parts = _load_split(path, split, n_rows)

    def apply(x: np.ndarray | None) -> dict[PartKey, np.ndarray] | None:
        # Integer-array indexing always returns a copy with the same dtype.
        return None if x is None else {p: x[idx] for p, idx in parts.items()}

    y = apply(y_all)
    assert y is not None
    return RawDataset(
        x_num=apply(features['x_num']),
        x_bin=apply(features['x_bin']),
        x_cat=apply(features['x_cat']),
        y=y,
        task=TaskInfo(
            type_=task_type, score=score, n_classes=_infer_n_classes(task_type, y)
        ),
    )
