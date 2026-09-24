"""Tests for tabpack_repro.data.numerical (a05)."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.preprocessing

from tabpack_repro.data.numerical import (
    _n_quantiles,
    drop_constant_columns,
    noisy_quantile_transform,
    standard_transform,
)
from tabpack_repro.types import PARTS


def _make_parts(
    *,
    n_train: int = 600,
    n_val: int = 150,
    n_test: int = 150,
    dtype: type = np.float64,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Mixed features: continuous, skewed, integer-valued (ties) and wide-range."""
    rng = np.random.default_rng(seed)
    parts = {}
    for part, n in zip(PARTS, (n_train, n_val, n_test), strict=True):
        parts[part] = np.stack(
            [
                rng.normal(3.0, 2.0, n),
                rng.exponential(5.0, n),
                rng.integers(0, 6, n).astype(float),
                rng.uniform(-1e4, 1e4, n),
            ],
            axis=1,
        ).astype(dtype)
    return parts


def _copy(x_num: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in x_num.items()}


def _reference_noisy_quantile(
    x_num: dict[str, np.ndarray], seed: int, noise_std: float = 1e-5
) -> dict[str, np.ndarray]:
    """The documented recipe written out with plain sklearn/numpy calls."""
    x_train = x_num['train']
    qt = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1_000_000_000,
        random_state=seed,
    )
    noise = np.random.RandomState(seed).normal(0.0, noise_std, x_train.shape)
    qt.fit(x_train + noise.astype(x_train.dtype))
    out = {k: np.nan_to_num(qt.transform(v)) for k, v in x_num.items()}
    mask = np.array([len(np.unique(c)) > 1 for c in out['train'].T])
    return {k: v[:, mask].astype(np.float32) for k, v in out.items()}


def _reference_standard(x_num: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scaler = sklearn.preprocessing.StandardScaler().fit(x_num['train'])
    out = {k: np.nan_to_num(scaler.transform(v)) for k, v in x_num.items()}
    mask = np.array([len(np.unique(c)) > 1 for c in out['train'].T])
    return {k: v[:, mask].astype(np.float32) for k, v in out.items()}


def _assert_parts_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert list(a) == list(b)
    for part, value in a.items():
        assert value.dtype == b[part].dtype
        np.testing.assert_array_equal(value, b[part], err_msg=part)


# ---------------------------------------------------------------------------
# noisy_quantile_transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('n_train', 'expected'),
    [
        (0, 10),
        (100, 10),
        (329, 10),
        (330, 11),
        (3000, 100),
        (30_000, 1000),
        (10**6, 1000),
    ],
)
def test_n_quantiles_formula(n_train: int, expected: int) -> None:
    assert _n_quantiles(n_train) == expected


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('n_train', [120, 3000])
def test_noisy_quantile_matches_sklearn_recipe(dtype: type, n_train: int) -> None:
    x_num = _make_parts(n_train=n_train, dtype=dtype)
    x_num['train'][:, 1] = 7.0  # constant on train -> must be dropped
    x_num['val'][3, 0] = np.nan
    for seed in (0, 3):
        _assert_parts_equal(
            noisy_quantile_transform(x_num, seed=seed),
            _reference_noisy_quantile(x_num, seed),
        )


def test_noisy_quantile_noise_std_is_used() -> None:
    x_num = _make_parts()
    out = noisy_quantile_transform(x_num, seed=0, noise_std=0.5)
    _assert_parts_equal(out, _reference_noisy_quantile(x_num, 0, noise_std=0.5))
    assert not np.array_equal(
        out['train'], noisy_quantile_transform(x_num, seed=0)['train']
    )


def test_noisy_quantile_is_deterministic() -> None:
    x_num = _make_parts()
    _assert_parts_equal(
        noisy_quantile_transform(x_num, seed=11),
        noisy_quantile_transform(x_num, seed=11),
    )


def test_noisy_quantile_seed_changes_the_fit() -> None:
    x_num = _make_parts()
    a = noisy_quantile_transform(x_num, seed=0)
    b = noisy_quantile_transform(x_num, seed=1)
    for part in PARTS:
        assert a[part].shape == b[part].shape
        assert not np.array_equal(a[part], b[part])
        # ... but only by a small amount on the continuous feature (noise std 1e-5;
        # the extreme train values move the most).
        np.testing.assert_allclose(a[part][:, 0], b[part][:, 0], atol=1e-2)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_noisy_quantile_output_is_finite_float32(dtype: type) -> None:
    x_num = {k: v.astype(dtype) for k, v in _make_parts().items()}
    out = noisy_quantile_transform(x_num, seed=0)
    assert list(out) == list(x_num)
    for part in PARTS:
        assert out[part].dtype == np.float32
        assert out[part].shape == (x_num[part].shape[0], 4)
        assert np.isfinite(out[part]).all()


def test_noisy_quantile_train_marginals_are_standard_normal() -> None:
    x_num = _make_parts(n_train=3000)
    out = noisy_quantile_transform(x_num, seed=0)['train']
    # Continuous features (the integer one has ties and is only roughly Gaussian).
    for j in (0, 1, 3):
        column = out[:, j].astype(np.float64)
        assert abs(column.mean()) < 0.05, j
        assert abs(column.std() - 1.0) < 0.1, j
        # Quantiles line up with N(0, 1): 2.5%, 50%, 97.5%.
        q = np.quantile(column, [0.025, 0.5, 0.975])
        np.testing.assert_allclose(q, [-1.96, 0.0, 1.96], atol=0.1)
    # The transform is monotone per feature: ranks are preserved on train.
    for j in (0, 1, 3):
        order = np.argsort(x_num['train'][:, j], kind='stable')
        assert np.all(np.diff(out[order, j]) >= 0)


@pytest.mark.filterwarnings('ignore:All-NaN slice encountered:RuntimeWarning')
def test_noisy_quantile_drops_train_constant_columns_in_all_parts() -> None:
    x_num = _make_parts()
    x_num['train'][:, 2] = -1.0  # constant on train only; val/test vary
    x_num['train'][:, 0] = np.nan  # all-NaN on train -> NaN -> 0 -> constant
    out = noisy_quantile_transform(x_num, seed=0)
    for part in PARTS:
        assert out[part].shape == (x_num[part].shape[0], 2)
    # The surviving columns are the original columns 1 and 3 (monotone in them).
    for new_j, old_j in enumerate((1, 3)):
        order = np.argsort(x_num['val'][:, old_j], kind='stable')
        assert np.all(np.diff(out['val'][order, new_j]) >= 0)


def test_noisy_quantile_nan_becomes_zero() -> None:
    x_num = _make_parts()
    x_num['train'][[5, 17], 0] = np.nan
    x_num['val'][[0, 9], 1] = np.nan
    x_num['test'][4, 3] = np.nan
    out = noisy_quantile_transform(x_num, seed=0)
    assert out['train'][5, 0] == 0.0
    assert out['train'][17, 0] == 0.0
    assert out['val'][0, 1] == 0.0
    assert out['val'][9, 1] == 0.0
    assert out['test'][4, 3] == 0.0
    for part in PARTS:
        assert np.isfinite(out[part]).all()


def test_noisy_quantile_does_not_mutate_inputs() -> None:
    x_num = _make_parts(dtype=np.float32)
    x_num['val'][2, 1] = np.nan
    before = _copy(x_num)
    noisy_quantile_transform(x_num, seed=0)
    assert list(x_num) == list(before)
    for part in PARTS:
        assert x_num[part].dtype == before[part].dtype
        np.testing.assert_array_equal(x_num[part], before[part])


def test_noisy_quantile_transforms_clean_train_not_noisy_train() -> None:
    # Identical rows in train map to identical outputs, which would not hold if the
    # noisy copy (which breaks those ties) were transformed instead of the clean one.
    x_num = _make_parts()
    x_num['train'][1] = x_num['train'][0]
    out = noisy_quantile_transform(x_num, seed=0, noise_std=1e-2)
    np.testing.assert_array_equal(out['train'][0], out['train'][1])


def test_noisy_quantile_accepts_any_subset_of_parts() -> None:
    x_num = _make_parts()
    del x_num['val']
    out = noisy_quantile_transform(x_num, seed=0)
    assert list(out) == ['train', 'test']
    _assert_parts_equal(out, _reference_noisy_quantile(x_num, 0))


# ---------------------------------------------------------------------------
# drop_constant_columns
# ---------------------------------------------------------------------------


def test_drop_constant_columns_uses_train_only() -> None:
    x_num = {
        'train': np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 0.0], [3.0, 5.0, 1.0]]),
        'val': np.array([[4.0, 6.0, 7.0]]),
        'test': np.array([[0.0, 1.0, 2.0], [9.0, 9.0, 9.0]]),
    }
    out = drop_constant_columns(x_num)
    np.testing.assert_array_equal(out['train'], x_num['train'][:, [0, 2]])
    np.testing.assert_array_equal(out['val'], [[4.0, 7.0]])
    np.testing.assert_array_equal(out['test'], [[0.0, 2.0], [9.0, 9.0]])


def test_drop_constant_columns_preserves_dtype_and_inputs() -> None:
    x_num = {
        'train': np.array([[1, 2], [1, 3]], dtype=np.int64),
        'test': np.array([[4, 5]], dtype=np.int64),
    }
    before = _copy(x_num)
    out = drop_constant_columns(x_num)
    assert out['train'].dtype == np.int64
    np.testing.assert_array_equal(out['train'], [[2], [3]])
    np.testing.assert_array_equal(out['test'], [[5]])
    out['train'][0, 0] = 100  # the result is a copy, not a view of the input
    for part, value in x_num.items():
        np.testing.assert_array_equal(value, before[part])


def test_drop_constant_columns_all_constant_and_empty() -> None:
    x_num = {'train': np.ones((4, 3)), 'test': np.arange(6.0).reshape(2, 3)}
    out = drop_constant_columns(x_num)
    assert out['train'].shape == (4, 0)
    assert out['test'].shape == (2, 0)
    empty = {'train': np.empty((4, 0)), 'test': np.empty((2, 0))}
    out = drop_constant_columns(empty)
    assert out['train'].shape == (4, 0)
    assert out['test'].shape == (2, 0)


# ---------------------------------------------------------------------------
# standard_transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_standard_matches_sklearn_recipe(dtype: type) -> None:
    x_num = _make_parts(dtype=dtype)
    x_num['train'][:, 3] = 2.5  # constant on train -> dropped
    x_num['test'][1, 0] = np.nan
    _assert_parts_equal(standard_transform(x_num), _reference_standard(x_num))


def test_standard_train_is_standardized_and_consistent() -> None:
    x_num = _make_parts()
    x_num['train'][:, 1] = 0.0
    before = _copy(x_num)
    out = standard_transform(x_num)
    for part in PARTS:
        assert out[part].dtype == np.float32
        assert out[part].shape == (x_num[part].shape[0], 3)
        np.testing.assert_array_equal(x_num[part], before[part])
    np.testing.assert_allclose(out['train'].mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out['train'].std(0), 1.0, atol=1e-5)
    # Val uses the train statistics of the surviving columns 0, 2 and 3.
    kept = x_num['train'][:, [0, 2, 3]]
    expected = (x_num['val'][:, [0, 2, 3]] - kept.mean(0)) / kept.std(0)
    np.testing.assert_allclose(out['val'], expected, rtol=1e-5, atol=1e-5)


def test_standard_nan_becomes_zero() -> None:
    x_num = _make_parts()
    x_num['val'][3, 2] = np.nan
    x_num['train'][0, 0] = np.nan  # ignored by the fit, then mapped to 0
    out = standard_transform(x_num)
    assert out['val'][3, 2] == 0.0
    assert out['train'][0, 0] == 0.0
    for part in PARTS:
        assert np.isfinite(out[part]).all()
