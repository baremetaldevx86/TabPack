"""Tests for tabpack_repro.data.dataset (a04)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from tabpack_repro.data.dataset import RawDataset, TaskInfo, load_raw_dataset
from tabpack_repro.types import PARTS, TaskType

N_ROWS = 40
# Deliberately non-contiguous and unsorted indices to catch slicing shortcuts.
DEFAULT_SPLIT = {
    'train': np.array([5, 0, 7, 9, 11, 13, 1, 2, 3, 4, 6, 8, 15, 17, 19, 21, 23, 25]),
    'val': np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 27]),
    'test': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
}
CATEGORIES = np.array(['France', 'Germany', 'Spain'])


def _make_arrays(task_type: str, n_classes: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if task_type == 'regression':
        y = rng.standard_normal(N_ROWS).astype(np.float32)
    else:
        # Every class appears in train (rows 0..9 all belong to the train part).
        y = (np.arange(N_ROWS) % n_classes).astype(np.int64)
    return {
        'x_num': rng.standard_normal((N_ROWS, 4)).astype(np.float32),
        'x_bin': rng.integers(0, 2, (N_ROWS, 3)).astype(np.float32),
        'x_cat': CATEGORIES[rng.integers(0, 3, (N_ROWS, 2))],
        'y': y,
    }


def write_dataset(
    root: Path,
    *,
    task_type: str = 'binclass',
    score: str = 'accuracy',
    n_classes: int = 2,
    features: tuple[str, ...] = ('x_num', 'x_bin', 'x_cat'),
    split: dict[str, np.ndarray] | None = None,
    split_name: str = 'default',
    arrays: dict | None = None,
) -> Path:
    """Write a dataset directory in the official on-disk format."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'info.json').write_text(
        json.dumps({'task': {'type': task_type, 'score': score}}, indent=4)
    )
    arrays = _make_arrays(task_type, n_classes) if arrays is None else arrays
    for key in (*features, 'y'):
        np.save(root / f'{key}.npy', arrays[key])
    split_dir = root / 'splits' / split_name
    split_dir.mkdir(parents=True)
    for part, idx in (DEFAULT_SPLIT if split is None else split).items():
        np.save(split_dir / f'{part}.npy', np.asarray(idx).astype(np.int32))
    return root


# >>> Feature-file combinations x task types

FEATURE_COMBOS = [
    tuple(
        k
        for k, present in zip(('x_num', 'x_bin', 'x_cat'), mask, strict=True)
        if present
    )
    for mask in itertools.product([True, False], repeat=3)
]
TASKS = [
    ('regression', 'rmse', None),
    ('binclass', 'accuracy', 2),
    ('multiclass', 'accuracy', 4),
]


@pytest.mark.parametrize(
    'features', FEATURE_COMBOS, ids=lambda f: '+'.join(f) or 'none'
)
@pytest.mark.parametrize('task', TASKS, ids=lambda t: t[0])
def test_load_all_feature_combinations(tmp_path, features, task):
    task_type, score, n_classes = task
    arrays = _make_arrays(task_type, n_classes or 2)
    path = write_dataset(
        tmp_path / 'ds',
        task_type=task_type,
        score=score,
        n_classes=n_classes or 2,
        features=features,
        arrays=arrays,
    )
    ds = load_raw_dataset(path)

    assert isinstance(ds, RawDataset)
    assert ds.task == TaskInfo(
        type_=TaskType(task_type), score=score, n_classes=n_classes
    )
    assert ds.task.is_regression == (task_type == 'regression')
    for key in ('x_num', 'x_bin', 'x_cat'):
        value = getattr(ds, key)
        if key not in features:
            assert value is None
            continue
        assert value is not None
        assert list(value) == list(PARTS)
        for part in PARTS:
            np.testing.assert_array_equal(value[part], arrays[key][DEFAULT_SPLIT[part]])
            assert value[part].dtype == arrays[key].dtype
    for part in PARTS:
        np.testing.assert_array_equal(ds.y[part], arrays['y'][DEFAULT_SPLIT[part]])
        assert ds.y[part].dtype == arrays['y'].dtype
        assert ds.size(part) == len(DEFAULT_SPLIT[part])


def test_dtypes_are_preserved(tmp_path):
    ds = load_raw_dataset(write_dataset(tmp_path / 'ds'))
    assert ds.x_num is not None and ds.x_bin is not None and ds.x_cat is not None
    assert ds.x_num['train'].dtype == np.float32
    assert ds.x_bin['train'].dtype == np.float32
    # x_cat stays a fixed-width unicode array (no ordinal encoding at this stage).
    assert isinstance(ds.x_cat['train'].dtype, np.dtypes.StrDType)
    assert set(np.unique(ds.x_cat['train'])) <= set(CATEGORIES)
    assert ds.y['train'].dtype == np.int64


def test_integer_x_cat_is_accepted(tmp_path):
    arrays = _make_arrays('binclass', 2)
    arrays['x_cat'] = np.arange(N_ROWS * 2, dtype=np.int64).reshape(N_ROWS, 2) % 5
    ds = load_raw_dataset(write_dataset(tmp_path / 'ds', arrays=arrays))
    assert ds.x_cat is not None
    assert ds.x_cat['val'].dtype == np.int64


def test_returned_arrays_are_independent_copies(tmp_path):
    path = write_dataset(tmp_path / 'ds')
    ds = load_raw_dataset(path)
    assert ds.x_num is not None
    val_before = ds.x_num['val'].copy()
    ds.x_num['train'][:] = -1.0
    ds.y['train'][:] = 0
    # Mutating one part touches neither the other parts nor the files on disk.
    np.testing.assert_array_equal(ds.x_num['val'], val_before)
    fresh = load_raw_dataset(path)
    assert fresh.x_num is not None
    assert not np.all(fresh.x_num['train'] == -1.0)
    assert fresh.y['train'].any()


def test_accepts_str_path_and_named_split(tmp_path):
    path = write_dataset(tmp_path / 'ds', split_name='other')
    ds = load_raw_dataset(str(path), split='other')
    assert ds.size('train') == len(DEFAULT_SPLIT['train'])


def test_nested_split_collects_parts_from_parent_dirs(tmp_path):
    # Official layout: splits/a/test.npy is shared by the splits a/0 and a/1.
    path = write_dataset(tmp_path / 'ds')
    arrays = _make_arrays('binclass', 2)
    shared = path / 'splits' / 'a'
    shared.mkdir()
    np.save(shared / 'test.npy', DEFAULT_SPLIT['test'].astype(np.int32))
    folds = {
        '0': (np.arange(0, 20), np.arange(20, 28)),
        '1': (np.arange(8, 28), np.arange(0, 8)),
    }
    for name, (train, val) in folds.items():
        (shared / name).mkdir()
        np.save(shared / name / 'train.npy', train.astype(np.int32))
        np.save(shared / name / 'val.npy', val.astype(np.int32))
    for name, (train, val) in folds.items():
        ds = load_raw_dataset(path, split=f'a/{name}')
        assert ds.x_num is not None
        test = DEFAULT_SPLIT['test']
        np.testing.assert_array_equal(ds.x_num['train'], arrays['x_num'][train])
        np.testing.assert_array_equal(ds.x_num['val'], arrays['x_num'][val])
        np.testing.assert_array_equal(ds.x_num['test'], arrays['x_num'][test])


def test_appledouble_files_are_ignored(tmp_path):
    path = write_dataset(tmp_path / 'ds')
    (path / 'splits' / 'default' / '._train.npy').write_bytes(b'\x00\x05\x16\x07junk')
    ds = load_raw_dataset(path)
    assert ds.size('train') == len(DEFAULT_SPLIT['train'])


# >>> Missing files


def test_missing_directory_mentions_download(tmp_path):
    with pytest.raises(FileNotFoundError, match='tabpack-repro download'):
        load_raw_dataset(tmp_path / 'churn')


@pytest.mark.parametrize('name', ['info.json', 'y.npy'])
def test_missing_required_file_mentions_download(tmp_path, name):
    path = write_dataset(tmp_path / 'ds')
    (path / name).unlink()
    with pytest.raises(FileNotFoundError, match=rf'{name}.*tabpack-repro download'):
        load_raw_dataset(path)


def test_missing_split_lists_available_splits(tmp_path):
    path = write_dataset(tmp_path / 'ds')
    with pytest.raises(FileNotFoundError, match=r"'nope'.*\['default'\]"):
        load_raw_dataset(path, split='nope')


@pytest.mark.parametrize('split', ['', '/', '..', 'default/..'])
def test_invalid_split_name(tmp_path, split):
    path = write_dataset(tmp_path / 'ds')
    with pytest.raises(ValueError, match='Invalid split name'):
        load_raw_dataset(path, split=split)


# >>> Bad splits


def _bad_split(**changes: np.ndarray | None) -> dict[str, np.ndarray]:
    split = dict(DEFAULT_SPLIT)
    for part, idx in changes.items():
        if idx is None:
            del split[part]
        else:
            split[part] = idx
    return split


@pytest.mark.parametrize(
    ('split', 'match'),
    [
        (_bad_split(test=np.array([27, 28, 29])), "'val' and 'test'.*overlap"),
        (_bad_split(val=np.array([0, 10, 12])), "'train' and 'val'.*overlap"),
        (_bad_split(test=np.array([38, 39, N_ROWS])), 'out of range'),
        (_bad_split(train=np.array([-1, 0, 1])), 'out of range'),
        (_bad_split(val=np.array([10, 10, 12])), 'duplicate'),
        (_bad_split(val=np.array([], dtype=np.int32)), 'empty'),
        (_bad_split(val=None), r"missing: \['val'\]"),
        ({**DEFAULT_SPLIT, 'extra': np.array([0])}, r"unexpected: \['extra'\]"),
    ],
    ids=[
        'val-test-overlap',
        'train-val-overlap',
        'too-large',
        'negative',
        'duplicate',
        'empty',
        'missing-part',
        'extra-part',
    ],
)
def test_bad_split_raises(tmp_path, split, match):
    path = write_dataset(tmp_path / 'ds', split=split)
    with pytest.raises(ValueError, match=match):
        load_raw_dataset(path)


def test_float_split_indices_raise(tmp_path):
    path = write_dataset(tmp_path / 'ds')
    np.save(
        path / 'splits' / 'default' / 'val.npy', DEFAULT_SPLIT['val'].astype(np.float64)
    )
    with pytest.raises(ValueError, match='1D integer array'):
        load_raw_dataset(path)


def test_part_defined_twice_in_nested_split_raises(tmp_path):
    path = write_dataset(tmp_path / 'ds')
    nested = path / 'splits' / 'default' / 'inner'
    nested.mkdir()
    np.save(nested / 'test.npy', DEFAULT_SPLIT['test'].astype(np.int32))
    with pytest.raises(ValueError, match='more than once'):
        load_raw_dataset(path, split='default/inner')


# >>> Labels and task info


def _with_labels(y: np.ndarray, task_type: str = 'binclass') -> dict:
    arrays = _make_arrays(task_type, 2)
    arrays['y'] = y
    return arrays


@pytest.mark.parametrize(
    ('task_type', 'y', 'match'),
    [
        ('binclass', np.arange(N_ROWS) % 3, 'exactly 2 classes'),
        ('binclass', np.zeros(N_ROWS, dtype=np.int64), 'exactly 2 classes'),
        ('multiclass', (np.arange(N_ROWS) % 3) * 2, r'0\.\.C-1'),
        (
            'multiclass',
            np.where(np.arange(N_ROWS) < 28, np.arange(N_ROWS) % 3, 3),
            'absent',
        ),
        ('binclass', (np.arange(N_ROWS) % 2).astype(np.float32), 'integer dtype'),
        ('regression', np.arange(N_ROWS), 'floating dtype'),
    ],
    ids=[
        'binclass-3-classes',
        'binclass-1-class',
        'non-contiguous',
        'unseen-test-label',
        'float-clf-labels',
        'int-reg-labels',
    ],
)
def test_bad_labels_raise(tmp_path, task_type, y, match):
    arrays = _with_labels(y, task_type)
    path = write_dataset(tmp_path / 'ds', task_type=task_type, arrays=arrays)
    with pytest.raises(ValueError, match=match):
        load_raw_dataset(path)


def test_n_classes_is_inferred_from_train_labels(tmp_path):
    # 5 classes, all present in train, only a subset in val/test.
    y = np.where(np.arange(N_ROWS) < 26, np.arange(N_ROWS) % 5, 0).astype(np.int64)
    path = write_dataset(
        tmp_path / 'ds', task_type='multiclass', arrays=_with_labels(y)
    )
    ds = load_raw_dataset(path)
    assert ds.task.n_classes == 5
    assert ds.task.is_classification


@pytest.mark.parametrize(
    'info',
    [
        {},
        {'task': {'type': 'binclass'}},
        {'task': {'type': 'ranking', 'score': 'ndcg'}},
        {'task': {'type': 'binclass', 'score': ''}},
        [],
    ],
    ids=['no-task', 'no-score', 'unknown-type', 'empty-score', 'not-a-dict'],
)
def test_bad_info_json_raises(tmp_path, info):
    path = write_dataset(tmp_path / 'ds')
    (path / 'info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError):
        load_raw_dataset(path)


@pytest.mark.parametrize(
    ('key', 'value', 'match'),
    [
        ('x_num', np.zeros((N_ROWS - 1, 4), np.float32), 'must have shape'),
        ('x_bin', np.zeros(N_ROWS, np.float32), 'must have shape'),
        ('x_num', np.zeros((N_ROWS, 4), np.int64), 'Unsupported dtype'),
    ],
    ids=['rows-mismatch', 'not-2d', 'int-x-num'],
)
def test_bad_feature_arrays_raise(tmp_path, key, value, match):
    arrays = _make_arrays('binclass', 2)
    arrays[key] = value
    path = write_dataset(tmp_path / 'ds', arrays=arrays)
    with pytest.raises(ValueError, match=match):
        load_raw_dataset(path)


# >>> Real Churn data


@pytest.mark.data
def test_churn(churn_dir):
    ds = load_raw_dataset(churn_dir)
    assert ds.task == TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)
    assert {p: ds.size(p) for p in PARTS} == {'train': 6400, 'val': 1600, 'test': 2000}
    assert ds.x_num is not None and ds.x_bin is not None and ds.x_cat is not None
    for part in PARTS:
        n = ds.size(part)
        assert ds.x_num[part].shape == (n, 7) and ds.x_num[part].dtype == np.float32
        assert ds.x_bin[part].shape == (n, 3)
        assert set(np.unique(ds.x_bin[part])) <= {0.0, 1.0}
        assert ds.x_cat[part].shape == (n, 1)
        assert isinstance(ds.x_cat[part].dtype, np.dtypes.StrDType)
        assert ds.y[part].shape == (n,) and ds.y[part].dtype == np.int64
    assert set(np.unique(ds.x_cat['train'])) == {'France', 'Germany', 'Spain'}


@pytest.mark.data
def test_churn_matches_raw_files(churn_dir):
    ds = load_raw_dataset(churn_dir, split='default')
    y_all = np.load(churn_dir / 'y.npy')
    for part in PARTS:
        idx = np.load(churn_dir / 'splits' / 'default' / f'{part}.npy')
        np.testing.assert_array_equal(ds.y[part], y_all[idx])
