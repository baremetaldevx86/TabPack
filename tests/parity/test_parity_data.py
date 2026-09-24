"""Parity of our data pipeline with the official ``lib/data.py`` (a42).

Everything is compared *bitwise* (same dtype, shape and bytes, so even ``-0.0`` vs
``0.0`` would count as a mismatch): our implementation re-derives the official
recipe step by step, and every agent that owns a step reported bit-identity.

* End to end: ``build_dataset(DataConfig(path=...))`` vs the official
  ``build_dataset(...)`` on the real Churn dataset (official Churn config and other
  policies/seeds), on a Churn copy whose binary features are stored as numerical
  columns, and on synthetic dataset directories in the official on-disk format
  (NaNs, constant columns, binary/all-binary numerical features, unknown
  categories, no x_cat, integer x_cat, multiclass).
* Per step: ``noisy_quantile_transform``/``standard_transform``/
  ``drop_constant_columns`` vs ``transform_num``; ``extract_bin_from_num`` vs
  ``_extract_bin_from_num``; ``bin_to_cat`` vs
  ``Dataset.convert_bin_features_to_cat_``; ``ordinal_encode`` vs ``transform_cat``
  + ``Dataset.compute_cat_cardinalities``.

The official module needs no environment setup as long as ``cache=False``; the
``od`` fixture replaces ``lib.env``'s directory helpers with a failing stub so that
no test can create a cache/project directory inside the official clone.

Known, intentional differences that are therefore not tested as parity:
* if every numerical column is dropped as constant, the official Dataset keeps an
  ``(N, 0)`` x_num while ``PreparedDataset.x_num`` is None (a07's documented
  decision); this equivalence is asserted explicitly;
* a two-valued numerical column containing ``inf``: the official OrdinalEncoder
  raises, ours extracts it (unreachable for Churn, which has no ``inf``).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from tabpack_repro.config import DataConfig
from tabpack_repro.data.categorical import (
    bin_to_cat,
    extract_bin_from_num,
    ordinal_encode,
)
from tabpack_repro.data.dataset import load_raw_dataset
from tabpack_repro.data.numerical import (
    drop_constant_columns,
    noisy_quantile_transform,
    standard_transform,
)
from tabpack_repro.data.pipeline import PreparedDataset, build_dataset
from tabpack_repro.types import PARTS

# Both implementations emit these for columns that are all-NaN on train.
pytestmark = [
    pytest.mark.filterwarnings('ignore:All-NaN slice:RuntimeWarning'),
    pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning'),
]

Parts = dict[str, np.ndarray]

# The arguments the official Churn experiments pass to lib.data.build_dataset
# (experiments/tabpack/churn/main/config.json, 'data' section, + the default seed),
# except the cache which must stay off in tests.
OFFICIAL_CHURN_KWARGS: dict[str, Any] = {
    'num_policy': 'noisy-quantile',
    'extract_bin_from_num': True,
    'bin_policy': 'convert-to-cat',
    'cat_policy': 'ordinal',
    'seed': 0,
    'cache': False,
}
CHURN_SIZES = {'train': 6400, 'val': 1600, 'test': 2000}
CHURN_CAT_CARDINALITIES = [3, 2, 2, 2]


# >>> Fixtures and helpers


@pytest.fixture
def od(official: Callable[[str], ModuleType], monkeypatch) -> ModuleType:
    """The official ``lib.data`` module, guarded against touching its env dirs."""
    module = official('lib.data')

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            'parity tests must not use the official cache/project directories'
        )

    for name in ('get_project_dir', 'get_cache_dir', 'get_data_dir'):
        monkeypatch.setattr(module.env, name, forbidden)
    return module


def _assert_same_array(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    """Exact equality: same dtype, same shape and the same bytes."""
    np.testing.assert_array_equal(actual, expected, err_msg=what, strict=True)
    assert (
        np.ascontiguousarray(actual).tobytes()
        == np.ascontiguousarray(expected).tobytes()
    ), f'{what}: equal values but different bytes (e.g. -0.0 vs 0.0)'


def _assert_same_parts(actual: Parts | None, expected: Parts | None, what: str) -> None:
    if expected is None:
        assert actual is None, f'{what}: expected None'
        return
    assert actual is not None, f'{what}: unexpected None'
    assert set(actual) == set(expected), what
    for part in expected:
        _assert_same_array(actual[part], expected[part], f'{what}[{part!r}]')


def _copy(parts: Parts) -> Parts:
    return {part: value.copy() for part, value in parts.items()}


def _numpy(parts: dict[str, torch.Tensor] | None) -> Parts | None:
    if parts is None:
        return None
    for part, tensor in parts.items():
        assert tensor.device.type == 'cpu', part
        assert tensor.is_contiguous(), part
    return {part: tensor.numpy() for part, tensor in parts.items()}


def _assert_dataset_matches(ours: PreparedDataset, theirs: Any) -> None:
    """Compare our PreparedDataset with an official ``Dataset[np.ndarray]``."""
    data = theirs.data
    assert 'x_bin' not in data, 'the official run left binary features unconverted'
    assert set(data['y']) == set(PARTS)

    # Features: exact values, dtypes and column order.
    x_num = _numpy(ours.x_num)
    if x_num is None and 'x_num' in data:
        # All numerical columns were dropped as constant: official keeps (N, 0).
        assert all(value.shape[1] == 0 for value in data['x_num'].values())
    else:
        _assert_same_parts(x_num, data.get('x_num'), 'x_num')
    _assert_same_parts(_numpy(ours.x_cat), data.get('x_cat'), 'x_cat')
    _assert_same_parts(_numpy(ours.y), data['y'], 'y')

    # Derived information.
    assert ours.cat_cardinalities == theirs.compute_cat_cardinalities()
    assert all(isinstance(c, int) for c in ours.cat_cardinalities)
    assert ours.n_num_features == theirs.n_num_features
    assert ours.n_cat_features == theirs.n_cat_features
    for part in PARTS:
        assert ours.size(part) == theirs.size(part), part

    # Task.
    assert ours.task.type_.value == theirs.task.type_.value
    assert ours.task.score == theirs.task.score.value
    assert ours.task.n_classes == theirs.task.try_compute_n_classes()


def _official_dataset(od: ModuleType, data: dict[str, Parts]) -> Any:
    """An official in-memory Dataset (its methods need ``data['y']``)."""
    y = data['y']
    task = od.Task(labels=y, type_=od.TaskType.BINCLASS, score=od.Score.ACCURACY)
    return od.Dataset({key: dict(parts) for key, parts in data.items()}, task)


def _split_rows(x: np.ndarray, sizes: dict[str, int]) -> Parts:
    """Split rows into consecutive parts (train first)."""
    bounds = np.cumsum([0, *sizes.values()])
    return {part: x[bounds[i] : bounds[i + 1]].copy() for i, part in enumerate(sizes)}


def _sizes(n_train: int, n_val: int, n_test: int) -> dict[str, int]:
    return {'train': n_train, 'val': n_val, 'test': n_test}


def _labels(rng: np.random.Generator, sizes: dict[str, int], n_classes=2) -> Parts:
    return _split_rows(rng.integers(0, n_classes, sum(sizes.values())), sizes)


# >>> Synthetic data


def _numerical_matrix(
    rng: np.random.Generator, n_train: int, n: int, dtype: type[np.floating]
) -> np.ndarray:
    """Columns covering every numerical edge case, in a random order.

    Rows ``[:n_train]`` are the train part.
    """
    normal = rng.standard_normal(n)
    columns = [
        normal,
        rng.integers(0, 5, n).astype(np.float64),  # many ties
        np.exp(3.0 * rng.standard_normal(n)),  # heavy tail
        np.where(rng.random(n) < 0.1, np.nan, rng.standard_normal(n)),  # NaNs
        rng.choice([-1.5, 4.0], n),  # binary
        np.full(n, 3.25),  # constant everywhere
        np.r_[np.full(n_train, 0.5), rng.standard_normal(n - n_train)],  # train-const
        np.r_[np.full(n_train, np.nan), rng.standard_normal(n - n_train)],  # NaN train
        np.r_[rng.standard_normal(n_train), np.full(n - n_train, np.nan)],  # NaN eval
        -normal,  # a duplicate (up to the sign) of another column
    ]
    x = np.stack(columns, axis=1)[:, rng.permutation(len(columns))]
    return x.astype(dtype)


# Kinds of columns for the binary extraction; `n_train` rows come first.
def _bin_column(rng: np.random.Generator, kind: str, n_train: int, n: int):
    if kind == 'binary':
        return rng.choice(rng.normal(size=2) * 10, n)
    if kind == 'binary01':
        return rng.integers(0, 2, n).astype(np.float64)
    if kind == 'binary_nan':
        return np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 2, n))
    if kind == 'single_nan':
        return np.where(rng.random(n) < 0.2, np.nan, 7.0)
    if kind == 'constant':
        return np.full(n, -2.0)
    if kind == 'continuous':
        return rng.standard_normal(n)
    if kind == 'three':
        return rng.integers(0, 3, n).astype(np.float64)
    if kind == 'binary_eval_only':  # the second value never occurs in train
        column = np.full(n, 1.0)
        column[n_train + rng.integers(0, n - n_train, 3)] = -1.0
        return column
    if kind == 'negative_zero':  # -0.0 and 0.0 are one value; binary with 1.0
        return rng.choice([-0.0, 0.0, 1.0], n)
    raise ValueError(kind)


BIN_COLUMN_KINDS = (
    'binary',
    'binary01',
    'binary_nan',
    'single_nan',
    'constant',
    'continuous',
    'three',
    'binary_eval_only',
    'negative_zero',
)


def _bin_matrix(
    rng: np.random.Generator,
    kinds: list[str],
    n_train: int,
    n: int,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    columns = [_bin_column(rng, kind, n_train, n) for kind in kinds]
    return np.stack(columns, axis=1).astype(dtype)


def _string_categories(
    rng: np.random.Generator, sizes: dict[str, int], dtype: str = '<U7'
) -> Parts:
    """Two str columns (incl. a 'nan' string) with categories unseen in train."""
    n = sum(sizes.values())
    x = np.stack(
        [
            rng.choice(['France', 'Germany', 'Spain'], n),
            rng.choice(['a', 'bb', 'nan', 'z'], n),
            np.full(n, 'same'),
        ],
        axis=1,
    ).astype(dtype)
    parts = _split_rows(x, sizes)
    parts['val'][: 3 if sizes['val'] > 3 else 1, 0] = 'Unseen'
    parts['test'][-2:, 0] = 'Brazil'
    parts['test'][:4, 1] = 'new'
    parts['val'][-1, 2] = 'other'
    return parts


def _integer_categories(rng: np.random.Generator, sizes: dict[str, int]) -> Parts:
    n = sum(sizes.values())
    x = np.stack(
        [rng.choice([-3, 0, 10, 100], n), rng.integers(0, 2, n), np.full(n, 5)],
        axis=1,
    ).astype(np.int64)
    parts = _split_rows(x, sizes)
    parts['test'][:2, 0] = 99
    parts['val'][:1, 1] = -7
    parts['test'][-1, 2] = 6
    return parts


# >>> Raw loading


@pytest.mark.data
def test_load_raw_dataset_churn(od: ModuleType, churn_dir: Path) -> None:
    ours = load_raw_dataset(churn_dir)
    theirs = od.Dataset.from_dir(churn_dir, od.DEFAULT_SPLIT_ID)

    assert set(theirs.data) == {'x_num', 'x_bin', 'x_cat', 'y'}
    _assert_same_parts(ours.x_num, theirs.data['x_num'], 'x_num')
    _assert_same_parts(ours.x_bin, theirs.data['x_bin'], 'x_bin')
    _assert_same_parts(ours.x_cat, theirs.data['x_cat'], 'x_cat')
    _assert_same_parts(ours.y, theirs.data['y'], 'y')
    assert ours.task.type_.value == theirs.task.type_.value == 'binclass'
    assert ours.task.score == theirs.task.score.value == 'accuracy'
    assert ours.task.n_classes == theirs.task.compute_n_classes() == 2
    assert {part: ours.size(part) for part in PARTS} == CHURN_SIZES


# >>> End to end on Churn


def test_official_churn_kwargs_are_the_official_config_and_our_defaults(
    od: ModuleType,
) -> None:
    """OFFICIAL_CHURN_KWARGS == the paper's Churn data config == DataConfig()."""
    config_path = (
        Path(od.__file__).resolve().parents[2]
        / 'experiments'
        / 'tabpack'
        / 'churn'
        / 'main'
        / 'config.json'
    )
    if not config_path.exists():
        pytest.skip(f'{config_path} is missing in the official clone')
    official_data = json.loads(config_path.read_text())['data']
    assert official_data.pop('path') == 'data/churn'
    assert official_data.pop('cache') is True  # the experiments cache; tests do not
    assert 'seed' not in official_data  # -> the build_dataset default is used
    seed_default = inspect.signature(od.build_dataset).parameters['seed'].default
    expected = {**official_data, 'seed': seed_default, 'cache': False}
    assert expected == OFFICIAL_CHURN_KWARGS

    ours = DataConfig()
    assert Path(ours.path).name == 'churn'
    assert {
        field.name: getattr(ours, field.name)
        for field in dataclasses.fields(ours)
        if field.name != 'path'
    } == {k: v for k, v in OFFICIAL_CHURN_KWARGS.items() if k != 'cache'}


@pytest.mark.data
def test_build_dataset_churn_official_config(od: ModuleType, churn_dir: Path) -> None:
    ours = build_dataset(DataConfig(path=str(churn_dir.resolve())))
    theirs = od.build_dataset(churn_dir, **OFFICIAL_CHURN_KWARGS)

    _assert_dataset_matches(ours, theirs)
    # Pin the facts, so that a change on both sides is noticed too.
    assert {part: ours.size(part) for part in PARTS} == CHURN_SIZES
    assert ours.cat_cardinalities == CHURN_CAT_CARDINALITIES
    assert ours.n_num_features == 7
    assert ours.task.n_classes == 2
    assert ours.x_num is not None and ours.x_num['train'].dtype == torch.float32
    assert ours.x_cat is not None and ours.x_cat['train'].dtype == torch.int64
    assert ours.y['train'].dtype == torch.int64


@pytest.mark.data
@pytest.mark.parametrize(
    ('num_policy', 'seed', 'extract_bin_from_num'),
    [
        ('noisy-quantile', 1, True),
        ('noisy-quantile', 2026, True),
        ('noisy-quantile', 0, False),
        ('standard', 0, True),
        (None, 0, True),
    ],
)
def test_build_dataset_churn_other_settings(
    od: ModuleType,
    churn_dir: Path,
    num_policy: str | None,
    seed: int,
    extract_bin_from_num: bool,
) -> None:
    settings = {
        'num_policy': num_policy,
        'seed': seed,
        'extract_bin_from_num': extract_bin_from_num,
    }
    ours = build_dataset(DataConfig(path=str(churn_dir.resolve()), **settings))
    theirs = od.build_dataset(churn_dir, **{**OFFICIAL_CHURN_KWARGS, **settings})
    _assert_dataset_matches(ours, theirs)


@pytest.mark.data
def test_build_dataset_churn_binary_features_stored_as_numerical(
    od: ModuleType, churn_dir: Path, tmp_path: Path
) -> None:
    """Churn with its 3 binary features interleaved into x_num (no x_bin.npy).

    Both pipelines must extract them again (bool -> 'False'/'True' categories) and
    end up with exactly the dataset built from the original Churn files.
    """
    x_num = np.load(churn_dir / 'x_num.npy')
    x_bin = np.load(churn_dir / 'x_bin.npy')
    assert x_num.shape[1] == 7 and x_bin.shape[1] == 3
    columns = [x_bin[:, 0], *x_num[:, :3].T, x_bin[:, 1], *x_num[:, 3:].T, x_bin[:, 2]]
    root = tmp_path / 'churn-bins-in-num'
    (root / 'splits' / 'default').mkdir(parents=True)
    np.save(root / 'x_num.npy', np.stack(columns, axis=1))
    for name in ('x_cat', 'y'):
        np.save(root / f'{name}.npy', np.load(churn_dir / f'{name}.npy'))
    for part in PARTS:
        file = Path('splits') / 'default' / f'{part}.npy'
        np.save(root / file, np.load(churn_dir / file))
    (root / 'info.json').write_text((churn_dir / 'info.json').read_text())

    ours = build_dataset(DataConfig(path=str(root)))
    theirs = od.build_dataset(root, **OFFICIAL_CHURN_KWARGS)
    _assert_dataset_matches(ours, theirs)

    original = build_dataset(DataConfig(path=str(churn_dir.resolve())))
    _assert_same_parts(_numpy(ours.x_num), _numpy(original.x_num), 'x_num')
    _assert_same_parts(_numpy(ours.x_cat), _numpy(original.x_cat), 'x_cat')
    assert ours.cat_cardinalities == original.cat_cardinalities


# >>> Per step on Churn


@pytest.fixture
def churn_raw(od: ModuleType, churn_dir: Path) -> dict[str, Parts]:
    """Raw Churn arrays, loaded by the official loader."""
    return od.Dataset.from_dir(churn_dir, od.DEFAULT_SPLIT_ID).data


@pytest.mark.data
def test_extract_bin_from_num_churn(od: ModuleType, churn_raw) -> None:
    # Churn has no two-valued numerical column, so nothing is extracted.
    x_num = churn_raw['x_num']
    ours = extract_bin_from_num(_copy(x_num))
    theirs = od._extract_bin_from_num(_copy(x_num))
    assert ours[0] is None and theirs[0] is None
    _assert_same_parts(ours[1], theirs[1], 'remaining x_num')

    # With the binary features moved into x_num, all three are extracted.
    columns = {
        part: np.c_[churn_raw['x_bin'][part], value] for part, value in x_num.items()
    }
    ours = extract_bin_from_num(_copy(columns))
    theirs = od._extract_bin_from_num(_copy(columns))
    _assert_same_parts(ours[0], theirs[0], 'extracted')
    _assert_same_parts(ours[1], theirs[1], 'remaining x_num')
    _assert_same_parts(ours[1], x_num, 'remaining x_num vs original')


@pytest.mark.data
@pytest.mark.parametrize('seed', [0, 1, 17])
def test_noisy_quantile_transform_churn(od: ModuleType, churn_raw, seed: int) -> None:
    x_num = churn_raw['x_num']
    expected = od.transform_num(_copy(x_num), 'noisy-quantile', seed)
    actual = noisy_quantile_transform(x_num, seed=seed)
    _assert_same_parts(actual, expected, 'x_num')
    assert actual['train'].shape == (CHURN_SIZES['train'], 7)
    _assert_same_parts(x_num, churn_raw['x_num'], 'input (not modified)')


@pytest.mark.data
def test_standard_transform_and_drop_constant_columns_churn(
    od: ModuleType, churn_raw
) -> None:
    x_num = churn_raw['x_num']
    _assert_same_parts(
        standard_transform(x_num),
        od.transform_num(_copy(x_num), 'standard', None),
        'standard',
    )
    # No NaNs and float32: the policy-None transform is only the column filter.
    _assert_same_parts(
        drop_constant_columns(x_num), od.transform_num(_copy(x_num), None, None), 'None'
    )


@pytest.mark.data
def test_bin_to_cat_and_ordinal_encode_churn(od: ModuleType, churn_raw) -> None:
    dataset = _official_dataset(od, churn_raw)
    dataset.convert_bin_features_to_cat_()
    x_cat_str = bin_to_cat(churn_raw['x_bin'], churn_raw['x_cat'])
    _assert_same_parts(x_cat_str, dataset.data['x_cat'], 'x_cat (str)')
    assert x_cat_str['train'].dtype == np.dtype('<U7')

    codes, cardinalities = ordinal_encode(x_cat_str)
    dataset.data['x_cat'] = od.transform_cat(dataset.data['x_cat'], 'ordinal')
    _assert_same_parts(codes, dataset.data['x_cat'], 'x_cat (codes)')
    assert cardinalities == dataset.compute_cat_cardinalities()
    assert cardinalities == CHURN_CAT_CARDINALITIES


# >>> Per step on synthetic arrays


@pytest.mark.filterwarnings('ignore:n_quantiles:UserWarning')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('n_train', [7, 64, 450, 30_030])
@pytest.mark.parametrize('seed', [0, 3])
def test_numerical_transforms_synthetic(
    od: ModuleType, seed: int, n_train: int, dtype: type[np.floating]
) -> None:
    """NaNs, ties, heavy tails, constant/train-constant/all-NaN-train columns.

    n_train=7 (fewer rows than quantiles), 64 and 450 (the minimum of 10 and a
    regular n_train // 30), 30_030 (capped at 1000 quantiles).
    """
    rng = np.random.default_rng(seed)
    sizes = _sizes(n_train, max(n_train // 4, 3), max(n_train // 3, 3))
    x_num = _split_rows(
        _numerical_matrix(rng, n_train, sum(sizes.values()), dtype), sizes
    )
    snapshot = _copy(x_num)

    for noise_seed in (seed, seed + 1):
        _assert_same_parts(
            noisy_quantile_transform(x_num, seed=noise_seed),
            od.transform_num(_copy(x_num), 'noisy-quantile', noise_seed),
            f'noisy-quantile (seed={noise_seed})',
        )
    _assert_same_parts(
        standard_transform(x_num),
        od.transform_num(_copy(x_num), 'standard', None),
        'standard',
    )
    _assert_same_parts(x_num, snapshot, 'input (not modified)')


@pytest.mark.parametrize('seed', range(3))
def test_drop_constant_columns_synthetic(od: ModuleType, seed: int) -> None:
    """policy=None on NaN-free float32 data = only the train-constant filter."""
    rng = np.random.default_rng(seed)
    sizes = _sizes(50, 20, 20)
    x = _numerical_matrix(rng, 50, 90, np.float32)
    x_num = _split_rows(x[:, ~np.isnan(x).any(axis=0)], sizes)
    expected = od.transform_num(_copy(x_num), None, None)
    assert expected['train'].shape[1] < x_num['train'].shape[1]  # something dropped
    _assert_same_parts(drop_constant_columns(x_num), expected, 'x_num')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('seed', range(25))
def test_extract_bin_from_num_synthetic(
    od: ModuleType, seed: int, dtype: type[np.floating]
) -> None:
    rng = np.random.default_rng(seed)
    n_columns = int(rng.integers(1, 9))
    kinds = [str(k) for k in rng.choice(BIN_COLUMN_KINDS, n_columns)]
    sizes = _sizes(40, 10, 12)
    x_num = _split_rows(_bin_matrix(rng, kinds, 40, 62, dtype), sizes)

    ours = extract_bin_from_num(_copy(x_num))
    theirs = od._extract_bin_from_num(_copy(x_num))
    _assert_same_parts(ours[0], theirs[0], f'extracted {kinds}')
    _assert_same_parts(ours[1], theirs[1], f'remaining {kinds}')


@pytest.mark.parametrize(
    'kinds',
    [
        pytest.param(['binary', 'binary01', 'binary_eval_only'], id='all-binary'),
        pytest.param(['negative_zero', 'binary'], id='all-binary-negative-zero'),
        pytest.param(['continuous', 'binary_nan', 'single_nan'], id='no-binary'),
        pytest.param(['constant', 'three', 'binary', 'continuous'], id='mixed'),
    ],
)
def test_extract_bin_from_num_edge_cases(od: ModuleType, kinds: list[str]) -> None:
    rng = np.random.default_rng(0)
    x_num = _split_rows(_bin_matrix(rng, kinds, 30, 50), _sizes(30, 10, 10))
    ours = extract_bin_from_num(_copy(x_num))
    theirs = od._extract_bin_from_num(_copy(x_num))
    _assert_same_parts(ours[0], theirs[0], 'extracted')
    _assert_same_parts(ours[1], theirs[1], 'remaining')


def _x_bin(rng: np.random.Generator, kind: str, sizes: dict[str, int]) -> Parts:
    n = sum(sizes.values())
    x = rng.integers(0, 2, (n, 3)).astype(np.float32)
    if kind == 'float_nan':
        x[rng.random((n, 3)) < 0.15] = np.nan
    elif kind == 'bool':
        x = x.astype(bool)
    return _split_rows(x, sizes)


def _x_cat(rng: np.random.Generator, kind: str | None, sizes: dict[str, int]):
    if kind is None:
        return None
    if kind == 'int':
        return _integer_categories(rng, sizes)
    return _string_categories(rng, sizes, '<U1' if kind == 'str_short' else '<U7')


@pytest.mark.parametrize('cat_kind', [None, 'str', 'str_short', 'int'])
@pytest.mark.parametrize('bin_kind', ['float', 'float_nan', 'bool'])
def test_bin_to_cat_and_ordinal_encode_synthetic(
    od: ModuleType, bin_kind: str, cat_kind: str | None
) -> None:
    """'str_short' truncates the cast bins ('1.0' -> '1', 'True' -> 'T')."""
    if bin_kind == 'float_nan' and cat_kind == 'int':
        pytest.skip('casting NaN to int64 is undefined behavior in numpy')
    rng = np.random.default_rng(0)
    sizes = _sizes(60, 15, 20)
    x_bin = _x_bin(rng, bin_kind, sizes)
    x_cat = _x_cat(rng, cat_kind, sizes)
    data = {'x_bin': _copy(x_bin), 'y': _labels(rng, sizes)}
    if x_cat is not None:
        data['x_cat'] = _copy(x_cat)

    dataset = _official_dataset(od, data)
    dataset.convert_bin_features_to_cat_()
    converted = bin_to_cat(x_bin, x_cat)
    _assert_same_parts(converted, dataset.data['x_cat'], 'converted x_cat')

    codes, cardinalities = ordinal_encode(converted)
    dataset.data['x_cat'] = od.transform_cat(dataset.data['x_cat'], 'ordinal')
    _assert_same_parts(codes, dataset.data['x_cat'], 'x_cat codes')
    assert cardinalities == dataset.compute_cat_cardinalities()


@pytest.mark.parametrize('kind', ['str', 'str_short', 'int'])
@pytest.mark.parametrize('seed', range(4))
def test_ordinal_encode_unknown_categories(od: ModuleType, kind: str, seed: int):
    """Categories unseen in train (in val, test or both) -> train max + 1."""
    rng = np.random.default_rng(seed)
    sizes = _sizes(int(rng.integers(8, 80)), 10, 10)
    x_cat = _x_cat(rng, kind, sizes)
    assert x_cat is not None
    snapshot = _copy(x_cat)

    codes, cardinalities = ordinal_encode(x_cat)
    expected = od.transform_cat(_copy(x_cat), 'ordinal')
    _assert_same_parts(codes, expected, 'x_cat codes')
    dataset = _official_dataset(od, {'x_cat': expected, 'y': _labels(rng, sizes)})
    assert cardinalities == dataset.compute_cat_cardinalities()
    _assert_same_parts(x_cat, snapshot, 'input (not modified)')
    # The unknown categories really occur and get codes beyond the train codes.
    assert any(
        (codes[part] > codes['train'].max(axis=0)).any() for part in ('val', 'test')
    )


# >>> End to end on synthetic datasets in the official on-disk format


def _write_dataset_dir(
    path: Path,
    *,
    sizes: dict[str, int],
    task_type: str,
    y: Parts,
    x_num: Parts | None = None,
    x_bin: Parts | None = None,
    x_cat: Parts | None = None,
) -> Path:
    """Official format: arrays over all rows + int32 index files per part.

    Rows are stored in a shuffled order, so the split indices are non-trivial.
    """
    n = sum(sizes.values())
    order = np.random.default_rng(len(str(path))).permutation(n)
    bounds = np.cumsum([0, *sizes.values()])
    split = {part: order[bounds[i] : bounds[i + 1]] for i, part in enumerate(sizes)}

    (path / 'splits' / 'default').mkdir(parents=True)
    (path / 'info.json').write_text(
        json.dumps({'task': {'type': task_type, 'score': 'accuracy'}})
    )
    for name, parts in (('x_num', x_num), ('x_bin', x_bin), ('x_cat', x_cat), ('y', y)):
        if parts is None:
            continue
        first = parts['train']
        array = np.empty((n, *first.shape[1:]), dtype=first.dtype)
        for part, idx in split.items():
            array[idx] = parts[part]
        np.save(path / f'{name}.npy', array)
    for part, idx in split.items():
        np.save(path / 'splits' / 'default' / f'{part}.npy', idx.astype(np.int32))
    return path


def _synthetic_dataset_dir(root: Path, scenario: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    sizes = _sizes(300, 80, 90)
    n_train, n = sizes['train'], sum(sizes.values())

    def num(*kinds: str) -> Parts:
        return _split_rows(_bin_matrix(rng, list(kinds), n_train, n), sizes)

    kwargs: dict[str, Any] = {'task_type': 'binclass', 'y': _labels(rng, sizes)}
    if scenario == 'all-kinds':
        # Numerical edge cases + binary numerical columns + x_bin with NaNs (a
        # 'nan' category, since x_cat is str) + unknown string categories.
        x_num = _numerical_matrix(rng, n_train, n, np.float32)
        kwargs['x_num'] = _split_rows(x_num, sizes)
        kwargs['x_bin'] = _x_bin(rng, 'float_nan', sizes)
        kwargs['x_cat'] = _string_categories(rng, sizes)
    elif scenario == 'no-x-cat-multiclass':
        # Merged bool + float32 bins -> float32 -> NaN becomes code 2 (no x_cat).
        kwargs['task_type'] = 'multiclass'
        kwargs['y'] = _labels(rng, sizes, n_classes=3)
        kwargs['x_num'] = num('continuous', 'binary', 'three', 'binary_nan')
        kwargs['x_bin'] = _x_bin(rng, 'float_nan', sizes)
    elif scenario == 'num-all-binary':
        # x_num disappears entirely.
        kwargs['x_num'] = num('binary', 'binary01', 'negative_zero')
        kwargs['x_cat'] = _string_categories(rng, sizes)
    elif scenario == 'num-only':
        # Extracted bool bins with neither x_bin nor x_cat.
        kwargs['x_num'] = num('continuous', 'binary', 'three', 'single_nan')
    elif scenario == 'int-x-cat':
        kwargs['x_num'] = num('continuous', 'binary_nan', 'binary_eval_only')
        kwargs['x_bin'] = _x_bin(rng, 'float', sizes)
        kwargs['x_cat'] = _integer_categories(rng, sizes)
    elif scenario == 'num-all-constant':
        # Every numerical column is dropped: official (N, 0) vs our None.
        x_num = num('constant', 'constant')
        x_num['val'][:, 1] = 1.0  # constant on train only
        kwargs['x_num'] = x_num
        kwargs['x_cat'] = _string_categories(rng, sizes)
    else:
        raise ValueError(scenario)
    return _write_dataset_dir(root / scenario, sizes=sizes, **kwargs)


SCENARIOS = (
    'all-kinds',
    'no-x-cat-multiclass',
    'num-all-binary',
    'num-only',
    'int-x-cat',
    'num-all-constant',
)


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_build_dataset_synthetic(
    od: ModuleType, tmp_path: Path, scenario: str, seed: int
) -> None:
    path = _synthetic_dataset_dir(tmp_path, scenario, seed)
    ours = build_dataset(DataConfig(path=str(path), seed=seed))
    theirs = od.build_dataset(path, **{**OFFICIAL_CHURN_KWARGS, 'seed': seed})
    _assert_dataset_matches(ours, theirs)

    if scenario in ('num-all-binary', 'num-all-constant'):
        assert ours.x_num is None
    if scenario == 'num-all-constant':
        assert theirs.data['x_num']['train'].shape == (300, 0)
    if scenario == 'num-only':
        assert ours.x_cat is not None and ours.cat_cardinalities == [2]


@pytest.mark.parametrize('extract_bin_from_num', [True, False])
@pytest.mark.parametrize('num_policy', ['noisy-quantile', 'standard', None])
def test_build_dataset_synthetic_policies(
    od: ModuleType,
    tmp_path: Path,
    num_policy: str | None,
    extract_bin_from_num: bool,
) -> None:
    path = _synthetic_dataset_dir(tmp_path, 'all-kinds', 2)
    settings = {'num_policy': num_policy, 'extract_bin_from_num': extract_bin_from_num}
    ours = build_dataset(DataConfig(path=str(path), **settings))
    theirs = od.build_dataset(path, **{**OFFICIAL_CHURN_KWARGS, **settings})
    _assert_dataset_matches(ours, theirs)
