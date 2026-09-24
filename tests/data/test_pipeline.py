"""Tests for data/pipeline.py (a07).

The orchestration tests replace the functions owned by a03-a06 with fakes, so they
check the pipeline in isolation: call order, data flow between the steps, policy
validation, path resolution and the output tensor dtypes.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _helpers import make_synthetic_dataset

from tabpack_repro.config import DataConfig
from tabpack_repro.data import pipeline
from tabpack_repro.data.dataset import RawDataset, TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset, build_dataset
from tabpack_repro.types import PARTS, TaskType

SIZES = {'train': 6, 'val': 3, 'test': 2}
BINCLASS = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)

# ---------------------------------------------------------------------------
# PreparedDataset.to


def _tensors(ds: PreparedDataset) -> list[torch.Tensor]:
    out = list(ds.y.values())
    for parts in (ds.x_num, ds.x_cat):
        if parts is not None:
            out.extend(parts.values())
    return out


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu'), 'meta'])
def test_to_moves_every_tensor_and_returns_a_copy(device: Any) -> None:
    ds = make_synthetic_dataset(n_train=8, n_val=4, n_test=4)
    moved = ds.to(device)

    assert moved is not ds
    assert all(t.device == torch.device(device) for t in _tensors(moved))
    for name in ('x_num', 'x_cat', 'y'):
        src, dst = getattr(ds, name), getattr(moved, name)
        assert dst is not src
        assert dst.keys() == src.keys()
        for part in PARTS:
            assert dst[part].dtype == src[part].dtype
            assert dst[part].shape == src[part].shape
    assert moved.task == ds.task
    assert moved.cat_cardinalities == ds.cat_cardinalities
    assert moved.cat_cardinalities is not ds.cat_cardinalities
    # The source is left untouched.
    assert all(t.device.type == 'cpu' for t in _tensors(ds))


def test_to_cpu_round_trip_preserves_values() -> None:
    ds = make_synthetic_dataset(n_train=8, n_val=4, n_test=4)
    moved = ds.to('cpu')
    assert ds.x_num is not None and moved.x_num is not None
    assert ds.x_cat is not None and moved.x_cat is not None
    for part in PARTS:
        assert torch.equal(moved.x_num[part], ds.x_num[part])
        assert torch.equal(moved.x_cat[part], ds.x_cat[part])
        assert torch.equal(moved.y[part], ds.y[part])
    assert moved.n_num_features == ds.n_num_features
    assert moved.n_cat_features == ds.n_cat_features
    assert all(moved.size(p) == ds.size(p) for p in PARTS)


def test_to_keeps_missing_feature_kinds_none() -> None:
    ds = make_synthetic_dataset(n_train=8, n_val=4, n_test=4, cat_cardinalities=())
    ds = dataclasses.replace(ds, x_num=None)
    moved = ds.to('meta')
    assert moved.x_num is None
    assert moved.x_cat is None
    assert moved.cat_cardinalities == []
    assert all(t.device.type == 'meta' for t in moved.y.values())


@pytest.mark.gpu
def test_to_cuda_and_back(cuda_device: torch.device) -> None:
    ds = make_synthetic_dataset(n_train=8, n_val=4, n_test=4)
    on_gpu = ds.to(cuda_device)
    assert all(t.is_cuda for t in _tensors(on_gpu))
    back = on_gpu.to('cpu')
    for a, b in zip(_tensors(back), _tensors(ds), strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# build_dataset with fakes of the a03-a06 functions


def _parts(fn) -> dict[str, np.ndarray]:
    return {part: fn(part, n) for part, n in SIZES.items()}


def _make_raw(
    *,
    x_num: bool = True,
    x_bin: bool = True,
    x_cat: bool = True,
    task: TaskInfo = BINCLASS,
) -> RawDataset:
    rng = np.random.default_rng(0)
    return RawDataset(
        x_num=_parts(lambda p, n: rng.standard_normal((n, 3))) if x_num else None,
        x_bin=_parts(lambda p, n: rng.integers(0, 2, (n, 1)).astype(np.float32))
        if x_bin
        else None,
        x_cat=_parts(lambda p, n: rng.choice(['a', 'b', 'c'], (n, 2)))
        if x_cat
        else None,
        y=_parts(lambda p, n: rng.integers(0, 2, n).astype(np.int64)),
        task=task,
    )


class Fakes:
    """Replaces every dependency of the pipeline and records what it received."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, raw: RawDataset) -> None:
        self.raw = raw
        self.calls: list[str] = []
        self.args: dict[str, Any] = {}
        self.out: dict[str, Any] = {}
        self.data_dir = Path('/fake/data/root')
        # What the fake extract_bin_from_num returns: 'split' (1 binary + 2 numerical
        # columns), 'none' (nothing extracted) or 'all' (every column is binary).
        self.extract_mode = 'split'
        for name in (
            'load_raw_dataset',
            'get_data_dir',
            'extract_bin_from_num',
            'merge_bin',
            'noisy_quantile_transform',
            'standard_transform',
            'drop_constant_columns',
            'bin_to_cat',
            'ordinal_encode',
        ):
            monkeypatch.setattr(pipeline, name, getattr(self, name))

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(name)
        self.args[name] = (args, kwargs)

    def load_raw_dataset(self, path, split='default'):
        self._record('load_raw_dataset', path, split=split)
        return self.raw

    def get_data_dir(self):
        self._record('get_data_dir')
        return self.data_dir

    def extract_bin_from_num(self, x_num):
        self._record('extract_bin_from_num', x_num)
        if self.extract_mode == 'none':
            out = (None, x_num)
        elif self.extract_mode == 'all':
            out = ({k: v > 0 for k, v in x_num.items()}, None)
        else:
            out = (
                {k: v[:, :1] > 0 for k, v in x_num.items()},
                {k: v[:, 1:] for k, v in x_num.items()},
            )
        self.out['extract_bin_from_num'] = out
        return out

    def merge_bin(self, extracted, x_bin):
        self._record('merge_bin', extracted, x_bin)
        if x_bin is None:
            out = dict(extracted)
        else:
            out = {
                k: np.concatenate([extracted[k], x_bin[k]], axis=1) for k in extracted
            }
        self.out['merge_bin'] = out
        return out

    def _num(self, name, x_num):
        out = {k: (10 * v).astype(np.float32) for k, v in x_num.items()}
        self.out[name] = out
        return out

    def noisy_quantile_transform(self, x_num, *, seed, noise_std=1e-5):
        self._record('noisy_quantile_transform', x_num, seed=seed)
        return self._num('noisy_quantile_transform', x_num)

    def standard_transform(self, x_num):
        self._record('standard_transform', x_num)
        return self._num('standard_transform', x_num)

    def drop_constant_columns(self, x_num):
        self._record('drop_constant_columns', x_num)
        mask = np.array([len(np.unique(c)) > 1 for c in x_num['train'].T], dtype=bool)
        out = {k: v[:, mask] for k, v in x_num.items()}
        self.out['drop_constant_columns'] = out
        return out

    def bin_to_cat(self, x_bin, x_cat):
        self._record('bin_to_cat', x_bin, x_cat)
        if x_cat is None:
            out = {k: v.astype(np.int64) for k, v in x_bin.items()}
        else:
            out = {k: np.column_stack([x_cat[k], x_bin[k].astype(str)]) for k in x_cat}
        self.out['bin_to_cat'] = out
        return out

    def ordinal_encode(self, x_cat):
        self._record('ordinal_encode', x_cat)
        n_cols = x_cat['train'].shape[1]
        codes = {
            k: np.tile(np.arange(n_cols, dtype=np.int64), (len(v), 1))
            for k, v in x_cat.items()
        }
        out = (codes, [c + 2 for c in range(n_cols)])
        self.out['ordinal_encode'] = out
        return out


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    return Fakes(monkeypatch, _make_raw())


def test_default_config_follows_official_order_and_data_flow(fakes: Fakes) -> None:
    ds = build_dataset(DataConfig(seed=7))

    assert fakes.calls == [
        'get_data_dir',
        'load_raw_dataset',
        'extract_bin_from_num',
        'merge_bin',
        'noisy_quantile_transform',
        'bin_to_cat',
        'ordinal_encode',
    ]
    raw = fakes.raw
    assert fakes.args['load_raw_dataset'][0] == (fakes.data_dir / 'churn',)
    assert fakes.args['extract_bin_from_num'][0][0] is raw.x_num
    extracted, remaining = fakes.out['extract_bin_from_num']
    # The extracted binary features go *before* the original x_bin columns.
    assert fakes.args['merge_bin'][0] == (extracted, raw.x_bin)
    # The numerical transform sees what extraction left, with the configured seed.
    assert fakes.args['noisy_quantile_transform'] == ((remaining,), {'seed': 7})
    assert fakes.args['bin_to_cat'][0] == (fakes.out['merge_bin'], raw.x_cat)
    assert fakes.args['ordinal_encode'][0][0] is fakes.out['bin_to_cat']

    codes, cardinalities = fakes.out['ordinal_encode']
    num = fakes.out['noisy_quantile_transform']
    assert ds.x_num is not None and ds.x_cat is not None
    for part in PARTS:
        np.testing.assert_array_equal(ds.x_num[part].numpy(), num[part])
        np.testing.assert_array_equal(ds.x_cat[part].numpy(), codes[part])
        np.testing.assert_array_equal(ds.y[part].numpy(), raw.y[part])
    assert ds.cat_cardinalities == cardinalities
    assert ds.task is raw.task
    assert ds.n_num_features == 2
    assert ds.n_cat_features == 2 + 1 + 1  # x_cat + original x_bin + extracted
    assert [ds.size(p) for p in PARTS] == list(SIZES.values())


def test_output_tensors_are_cpu_with_documented_dtypes(fakes: Fakes) -> None:
    # Raw labels in a narrower integer dtype must still come out as int64.
    fakes.raw.y = {k: v.astype(np.int32) for k, v in fakes.raw.y.items()}
    ds = build_dataset(DataConfig())
    assert ds.x_num is not None and ds.x_cat is not None
    assert ds.x_num.keys() == ds.x_cat.keys() == ds.y.keys() == set(PARTS)
    for part in PARTS:
        assert ds.x_num[part].dtype == torch.float32
        assert ds.x_cat[part].dtype == torch.int64
        assert ds.y[part].dtype == torch.int64
        assert ds.y[part].shape == (SIZES[part],)
        for t in (ds.x_num[part], ds.x_cat[part], ds.y[part]):
            assert t.device.type == 'cpu'
            assert t.is_contiguous()
    assert all(isinstance(c, int) for c in ds.cat_cardinalities)


def test_read_only_inputs_do_not_warn(fakes: Fakes) -> None:
    for parts in (fakes.raw.y, fakes.raw.x_num):
        for value in parts.values():
            value.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ds = build_dataset(DataConfig())
    # The tensors own writable memory: in-place ops must not touch the raw arrays.
    ds.y['train'] += 1
    assert not np.array_equal(ds.y['train'].numpy(), fakes.raw.y['train'])


def test_extraction_disabled(fakes: Fakes) -> None:
    ds = build_dataset(DataConfig(extract_bin_from_num=False))
    assert 'extract_bin_from_num' not in fakes.calls
    assert 'merge_bin' not in fakes.calls
    assert fakes.args['noisy_quantile_transform'][0][0] is fakes.raw.x_num
    assert fakes.args['bin_to_cat'][0] == (fakes.raw.x_bin, fakes.raw.x_cat)
    assert ds.n_num_features == 3
    assert ds.n_cat_features == 3


def test_nothing_extracted_skips_merge(fakes: Fakes) -> None:
    fakes.extract_mode = 'none'
    build_dataset(DataConfig())
    assert 'merge_bin' not in fakes.calls
    assert fakes.args['noisy_quantile_transform'][0][0] is fakes.raw.x_num
    assert fakes.args['bin_to_cat'][0][0] is fakes.raw.x_bin


def test_all_numerical_extracted_skips_num_transform(fakes: Fakes) -> None:
    fakes.extract_mode = 'all'
    ds = build_dataset(DataConfig())
    assert 'noisy_quantile_transform' not in fakes.calls
    assert ds.x_num is None
    assert ds.n_num_features == 0
    assert ds.n_cat_features == 2 + 1 + 3


def test_extracted_bin_without_raw_x_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_bin=False))
    build_dataset(DataConfig())
    extracted, _ = fakes.out['extract_bin_from_num']
    # merge_bin is None-aware; the pipeline passes the missing x_bin through.
    assert fakes.args['merge_bin'][0] == (extracted, None)
    assert fakes.args['bin_to_cat'][0][0] is fakes.out['merge_bin']


def test_no_numerical_features(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_num=False))
    ds = build_dataset(DataConfig())
    assert fakes.calls == [
        'get_data_dir',
        'load_raw_dataset',
        'bin_to_cat',
        'ordinal_encode',
    ]
    assert ds.x_num is None


def test_no_binary_features(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_bin=False))
    ds = build_dataset(DataConfig(extract_bin_from_num=False, bin_policy=None))
    assert 'bin_to_cat' not in fakes.calls
    assert fakes.args['ordinal_encode'][0][0] is fakes.raw.x_cat
    assert ds.n_cat_features == 2


def test_binary_only_categorical_features(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_cat=False))
    ds = build_dataset(DataConfig())
    assert fakes.args['bin_to_cat'][0] == (fakes.out['merge_bin'], None)
    assert ds.n_cat_features == 2


def test_no_categorical_features_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_bin=False, x_cat=False))
    ds = build_dataset(DataConfig(extract_bin_from_num=False))
    assert fakes.calls == [
        'get_data_dir',
        'load_raw_dataset',
        'noisy_quantile_transform',
    ]
    assert ds.x_cat is None
    assert ds.cat_cardinalities == []
    assert ds.n_cat_features == 0


def test_standard_num_policy(fakes: Fakes) -> None:
    build_dataset(DataConfig(num_policy='standard'))
    assert 'standard_transform' in fakes.calls
    assert 'noisy_quantile_transform' not in fakes.calls


def test_no_num_policy_applies_only_the_post_steps(fakes: Fakes) -> None:
    x_num = fakes.raw.x_num
    assert x_num is not None
    for value in x_num.values():
        value[:, 1] = 5.0  # constant on train -> dropped
        value[0, 2] = np.nan  # NaN -> 0
    ds = build_dataset(DataConfig(num_policy=None, extract_bin_from_num=False))
    assert 'noisy_quantile_transform' not in fakes.calls
    assert 'standard_transform' not in fakes.calls
    received = fakes.args['drop_constant_columns'][0][0]
    assert not any(np.isnan(v).any() for v in received.values())
    assert ds.x_num is not None
    for part in PARTS:
        expected = np.nan_to_num(x_num[part][:, [0, 2]]).astype(np.float32)
        assert ds.x_num[part].dtype == torch.float32
        np.testing.assert_array_equal(ds.x_num[part].numpy(), expected)


def test_all_numerical_columns_dropped_gives_none(fakes: Fakes) -> None:
    for value in (fakes.raw.x_num or {}).values():
        value[:] = 1.0
    ds = build_dataset(DataConfig(num_policy=None, extract_bin_from_num=False))
    assert ds.x_num is None
    assert ds.n_num_features == 0


@pytest.mark.parametrize('field', ['num_policy', 'bin_policy', 'cat_policy'])
def test_unknown_policy_raises_before_loading(fakes: Fakes, field: str) -> None:
    config = DataConfig(**{field: 'no-such-policy'})
    with pytest.raises(ValueError, match=field):
        build_dataset(config)
    assert fakes.calls == []


def test_one_hot_cat_policy_is_rejected(fakes: Fakes) -> None:
    with pytest.raises(ValueError, match='cat_policy'):
        build_dataset(DataConfig(cat_policy='one-hot'))
    assert fakes.calls == []


def test_unconverted_binary_features_raise(fakes: Fakes) -> None:
    with pytest.raises(ValueError, match='bin_policy'):
        build_dataset(DataConfig(bin_policy=None))


def test_no_cat_policy_keeps_integer_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(monkeypatch, _make_raw(x_cat=False))
    ds = build_dataset(DataConfig(cat_policy=None))
    assert 'ordinal_encode' not in fakes.calls
    x_cat = fakes.out['bin_to_cat']
    assert ds.x_cat is not None
    for part in PARTS:
        assert ds.x_cat[part].dtype == torch.int64
        np.testing.assert_array_equal(ds.x_cat[part].numpy(), x_cat[part])
    assert ds.cat_cardinalities == [len(np.unique(c)) for c in x_cat['train'].T]


def test_no_cat_policy_rejects_string_categories(fakes: Fakes) -> None:
    with pytest.raises(ValueError, match='cat_policy'):
        build_dataset(DataConfig(cat_policy=None))


def test_regression_is_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskInfo(type_=TaskType.REGRESSION, score='rmse')
    Fakes(monkeypatch, _make_raw(task=task))
    with pytest.raises(NotImplementedError, match='Regression'):
        build_dataset(DataConfig())


def test_relative_path_is_resolved_against_data_dir(fakes: Fakes) -> None:
    build_dataset(DataConfig(path='sub/churn'))
    assert fakes.args['load_raw_dataset'] == (
        (fakes.data_dir / 'sub' / 'churn',),
        {'split': 'default'},
    )


def test_absolute_path_is_used_as_is(fakes: Fakes, tmp_path: Path) -> None:
    build_dataset(DataConfig(path=str(tmp_path)))
    assert 'get_data_dir' not in fakes.calls
    assert fakes.args['load_raw_dataset'][0] == (tmp_path,)


# ---------------------------------------------------------------------------
# Integration: build_dataset with the real a03-a06 functions


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], splits: dict) -> Path:
    (path / 'splits' / 'default').mkdir(parents=True)
    (path / 'info.json').write_text(
        '{"task": {"type": "binclass", "score": "accuracy"}}'
    )
    for name, value in arrays.items():
        np.save(path / f'{name}.npy', value)
    for part, idx in splits.items():
        np.save(path / 'splits' / 'default' / f'{part}.npy', idx)
    return path


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    """A small dataset in the official on-disk format exercising every step.

    x_num columns: [continuous, binary {3, 7} (extracted), constant (dropped),
    continuous with a NaN in test (-> 0)]; x_bin: one 0/1 column; x_cat: one
    string column whose test part contains a category unseen in train.
    """
    rng = np.random.default_rng(0)
    n = 40
    x_num = np.stack(
        [
            rng.standard_normal(n),
            np.where(np.arange(n) % 2 == 0, 3.0, 7.0),
            np.full(n, 1.5),
            rng.standard_normal(n),
        ],
        axis=1,
    ).astype(np.float32)
    x_num[-1, 3] = np.nan
    x_cat = rng.choice(['b', 'a', 'c'], (n, 1))
    x_cat[-2, 0] = 'unseen'
    arrays = {
        'x_num': x_num,
        'x_bin': (np.arange(n) % 3 == 0).astype(np.float32)[:, None],
        'x_cat': x_cat,
        'y': (np.arange(n) % 2).astype(np.int64),
    }
    order = rng.permutation(n - 2)  # the last two rows (NaN, unseen) go to test
    splits = {
        'train': order[:24],
        'val': order[24:32],
        'test': np.concatenate([order[32:], [n - 2, n - 1]]),
    }
    return _write_dataset(tmp_path / 'tiny', arrays, splits)


def test_real_pipeline_on_tiny_dataset(tiny_dir: Path) -> None:
    ds = build_dataset(DataConfig(path=str(tiny_dir)))
    x_num = np.load(tiny_dir / 'x_num.npy')
    x_bin = np.load(tiny_dir / 'x_bin.npy')
    x_cat = np.load(tiny_dir / 'x_cat.npy')
    y = np.load(tiny_dir / 'y.npy')
    idx = {p: np.load(tiny_dir / 'splits' / 'default' / f'{p}.npy') for p in PARTS}

    assert ds.task.type_ == TaskType.BINCLASS
    assert ds.task.n_classes == 2
    assert [ds.size(p) for p in PARTS] == [24, 8, 8]
    # Binary column extracted, constant column dropped.
    assert ds.n_num_features == 2
    # [x_cat, extracted binary, x_bin]
    assert ds.cat_cardinalities == [3, 2, 2]
    assert ds.x_num is not None and ds.x_cat is not None
    for part in PARTS:
        assert ds.x_num[part].dtype == torch.float32
        assert torch.isfinite(ds.x_num[part]).all()
        assert ds.x_cat[part].dtype == torch.int64
        assert ds.y[part].dtype == torch.int64
        np.testing.assert_array_equal(ds.y[part].numpy(), y[idx[part]])
        codes = ds.x_cat[part].numpy()
        # Ordinal codes follow sorted train categories; unseen -> train_max + 1.
        expected = np.searchsorted(['a', 'b', 'c'], x_cat[idx[part], 0])
        expected[x_cat[idx[part], 0] == 'unseen'] = 3
        np.testing.assert_array_equal(codes[:, 0], expected)
        np.testing.assert_array_equal(codes[:, 1], x_num[idx[part], 1] == 7.0)
        np.testing.assert_array_equal(codes[:, 2], x_bin[idx[part], 0])
    assert ds.x_cat['test'][-2, 0] == 3
    assert ds.x_num['test'][-1, 1] == 0.0  # NaN -> 0
    # The noisy-quantile transform is monotone per column on the train part.
    for j, source in enumerate([0, 3]):
        order = np.argsort(x_num[idx['train'], source], kind='stable')
        assert np.all(np.diff(ds.x_num['train'][:, j].numpy()[order]) >= 0)


def test_real_pipeline_is_deterministic_and_seeded(tiny_dir: Path) -> None:
    a = build_dataset(DataConfig(path=str(tiny_dir)))
    b = build_dataset(DataConfig(path=str(tiny_dir)))
    c = build_dataset(DataConfig(path=str(tiny_dir), seed=1))
    assert a.x_num is not None and b.x_num is not None and c.x_num is not None
    for part in PARTS:
        assert torch.equal(a.x_num[part], b.x_num[part])
    assert not all(torch.equal(a.x_num[p], c.x_num[p]) for p in PARTS)


def test_real_pipeline_resolves_relative_path(
    tiny_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TABPACK_DATA_DIR', str(tiny_dir.parent))
    relative = build_dataset(DataConfig(path='tiny'))
    absolute = build_dataset(DataConfig(path=str(tiny_dir)))
    for a, b in zip(_tensors(relative), _tensors(absolute), strict=True):
        assert torch.equal(a, b)
    assert relative.cat_cardinalities == absolute.cat_cardinalities


def test_real_pipeline_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_dataset(DataConfig(path=str(tmp_path / 'missing')))


# ---------------------------------------------------------------------------
# Real Churn

CHURN_SIZES = {'train': 6400, 'val': 1600, 'test': 2000}


@pytest.fixture
def churn(churn_dir: Path, monkeypatch: pytest.MonkeyPatch) -> PreparedDataset:
    # The default config uses the relative path 'churn'.
    monkeypatch.setenv('TABPACK_DATA_DIR', str(churn_dir.parent))
    return build_dataset(DataConfig())


@pytest.mark.data
def test_churn_shapes_and_dtypes(churn: PreparedDataset) -> None:
    assert churn.task.type_ == TaskType.BINCLASS
    assert churn.task.n_classes == 2
    # 7 numerical features (none of them binary); 1 categorical + 3 binary.
    assert churn.n_num_features == 7
    assert churn.cat_cardinalities == [3, 2, 2, 2]
    assert churn.n_cat_features == 4
    assert churn.x_num is not None and churn.x_cat is not None
    for part, n in CHURN_SIZES.items():
        assert churn.size(part) == n
        x_num, x_cat, y = churn.x_num[part], churn.x_cat[part], churn.y[part]
        assert x_num.shape == (n, 7)
        assert x_num.dtype == torch.float32
        assert not torch.isnan(x_num).any()
        assert x_cat.shape == (n, 4)
        assert x_cat.dtype == torch.int64
        cards = torch.tensor(churn.cat_cardinalities)
        assert (x_cat >= 0).all()
        assert (x_cat < cards + 1).all()
        assert y.shape == (n,)
        assert y.dtype == torch.int64
        assert set(y.unique().tolist()) <= {0, 1}
        assert all(t.device.type == 'cpu' for t in (x_num, x_cat, y))
    # Train codes cover exactly 0..cardinality-1 (official compute_cat_cardinalities).
    train_cat = churn.x_cat['train']
    for j, card in enumerate(churn.cat_cardinalities):
        assert train_cat[:, j].unique().tolist() == list(range(card))
    # Normal-output quantile transform: roughly standardized train features.
    train_num = churn.x_num['train']
    assert train_num.mean(0).abs().max() < 0.5
    assert (train_num.abs() <= 6).all()


@pytest.mark.data
def test_churn_matches_absolute_path_and_to_cpu_round_trip(
    churn: PreparedDataset, churn_dir: Path
) -> None:
    absolute = build_dataset(DataConfig(path=str(churn_dir)))
    for a, b in zip(_tensors(churn), _tensors(absolute), strict=True):
        assert torch.equal(a, b)
    moved = churn.to('cpu')
    assert moved is not churn
    for a, b in zip(_tensors(moved), _tensors(churn), strict=True):
        assert a.device.type == 'cpu'
        assert a.dtype == b.dtype
        assert torch.equal(a, b)
    assert moved.cat_cardinalities == churn.cat_cardinalities
    assert moved.task == churn.task
