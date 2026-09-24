"""Tests for compute_stop_idx and FinishedPool (a22)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpack_repro.training.stopping import FinishedPool, compute_stop_idx


def _stop(bad, steps, *, patience, epoch_size=10, max_epochs=-1):
    return compute_stop_idx(
        np.asarray(bad, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
        patience=patience,
        epoch_size=epoch_size,
        max_epochs=max_epochs,
    )


def _reference_stop(bad, steps, *, patience, epoch_size, max_epochs):
    """Straightforward per-member loop over the documented rule."""
    idx = [
        i
        for i, (b, s) in enumerate(zip(bad, steps, strict=True))
        if (patience >= 0 and b > patience)
        or (max_epochs >= 0 and s // epoch_size >= max_epochs)
    ]
    return np.asarray(idx, dtype=np.int64) if idx else None


# ----------------------------------------------------------------------------------
# compute_stop_idx
# ----------------------------------------------------------------------------------
def test_patience_is_strictly_greater():
    # patience=2: members with 3+ bad updates stop, 2 does not.
    result = _stop([0, 2, 3, 5, 1], [0] * 5, patience=2)
    np.testing.assert_array_equal(result, [2, 3])
    assert result.dtype == np.int64


def test_patience_zero_stops_on_first_bad_update():
    np.testing.assert_array_equal(_stop([0, 1, 0], [0, 0, 0], patience=0), [1])


def test_patience_equal_count_does_not_stop():
    assert _stop([4, 4, 4], [100, 100, 100], patience=4) is None


def test_negative_patience_disables_early_stopping():
    assert _stop([10**6, 50], [0, 0], patience=-1) is None


def test_negative_max_epochs_disables_epoch_budget():
    assert _stop([0, 0], [10**9, 10**9], patience=-1, max_epochs=-1) is None
    assert _stop([0, 0], [10**9, 10**9], patience=5, max_epochs=-1) is None


def test_max_epochs_uses_floor_division_by_epoch_size():
    # epoch_size=10, max_epochs=3: stop once steps // 10 >= 3, i.e. steps >= 30.
    steps = [0, 29, 30, 31, 59]
    result = _stop([0] * 5, steps, patience=-1, epoch_size=10, max_epochs=3)
    np.testing.assert_array_equal(result, [2, 3, 4])


def test_max_epochs_non_divisible_epoch_size():
    # epoch_size=7, max_epochs=2: threshold at steps >= 14.
    result = _stop([0] * 4, [13, 14, 20, 6], patience=-1, epoch_size=7, max_epochs=2)
    np.testing.assert_array_equal(result, [1, 2])


def test_max_epochs_zero_stops_everyone():
    result = _stop([0, 0, 0], [0, 1, 2], patience=-1, epoch_size=5, max_epochs=0)
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_both_criteria_are_or_combined():
    bad = [0, 3, 0, 3, 1]
    steps = [0, 0, 40, 40, 10]
    result = _stop(bad, steps, patience=2, epoch_size=10, max_epochs=4)
    # member 1: patience; member 2: epochs; member 3: both.
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_none_when_nothing_stops():
    assert _stop([0, 1, 2], [1, 2, 3], patience=2, epoch_size=10, max_epochs=5) is None


def test_empty_pack_returns_none():
    assert _stop([], [], patience=0, epoch_size=1, max_epochs=0) is None


def test_matches_reference_rule_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = int(rng.integers(0, 12))
        bad = rng.integers(0, 8, k)
        steps = rng.integers(0, 200, k)
        patience = int(rng.integers(-1, 6))
        epoch_size = int(rng.integers(1, 30))
        max_epochs = int(rng.integers(-1, 10))
        kwargs = {
            'patience': patience,
            'epoch_size': epoch_size,
            'max_epochs': max_epochs,
        }
        expected = _reference_stop(bad, steps, **kwargs)
        result = compute_stop_idx(bad, steps, **kwargs)
        if expected is None:
            assert result is None
        else:
            np.testing.assert_array_equal(result, expected)
            assert result.dtype == np.int64


def test_does_not_mutate_inputs():
    bad = np.array([0, 5, 1], dtype=np.int64)
    steps = np.array([3, 3, 3], dtype=np.int64)
    bad0, steps0 = bad.copy(), steps.copy()
    compute_stop_idx(bad, steps, patience=1, epoch_size=1, max_epochs=2)
    np.testing.assert_array_equal(bad, bad0)
    np.testing.assert_array_equal(steps, steps0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _stop([0, 1], [0], patience=0)


def test_non_positive_epoch_size_with_budget_raises():
    with pytest.raises(ValueError):
        _stop([0], [0], patience=-1, epoch_size=0, max_epochs=1)


# ----------------------------------------------------------------------------------
# FinishedPool
# ----------------------------------------------------------------------------------
def _preds(values, n=4, *, with_test=True):
    """Binclass-like predictions: member m's row is filled with values[m]."""
    rows = torch.as_tensor(values, dtype=torch.float32)[:, None].expand(-1, n)
    out = {'train': rows.clone(), 'val': rows.clone() + 100}
    if with_test:
        out['test'] = rows.clone() + 200
    return out


def test_empty_pool_defaults():
    pool = FinishedPool()
    assert len(pool) == 0
    assert pool.ids.dtype == np.int64 and pool.ids.shape == (0,)
    assert pool.steps.dtype == np.int64 and pool.steps.shape == (0,)
    assert pool.predictions == {}


def test_default_pools_do_not_share_state():
    a, b = FinishedPool(), FinishedPool()
    a.extend(np.array([1]), np.array([2]), _preds([1.0]))
    assert len(b) == 0 and b.predictions == {}


def test_extend_concatenates_in_finishing_order():
    pool = FinishedPool()
    pool.extend(np.array([3, 0]), np.array([30, 5]), _preds([3.0, 0.0]))
    pool.extend(np.array([7]), np.array([12]), _preds([7.0]))
    pool.extend(np.array([1, 2]), np.array([40, 41]), _preds([1.0, 2.0]))

    assert len(pool) == 5
    np.testing.assert_array_equal(pool.ids, [3, 0, 7, 1, 2])
    np.testing.assert_array_equal(pool.steps, [30, 5, 12, 40, 41])
    assert set(pool.predictions) == {'train', 'val', 'test'}
    for offset, part in [(0, 'train'), (100, 'val'), (200, 'test')]:
        value = pool.predictions[part]
        assert value.shape == (5, 4)
        assert value.dtype == torch.float32
        expected = torch.tensor([3.0, 0.0, 7.0, 1.0, 2.0])[:, None] + offset
        torch.testing.assert_close(value, expected.expand(-1, 4))


def test_extend_casts_ids_and_steps_to_int64():
    pool = FinishedPool()
    pool.extend(
        np.array([1, 2], dtype=np.int32),
        np.array([10, 20], dtype=np.uint16),
        _preds([1.0, 2.0]),
    )
    pool.extend(np.array([5], dtype=np.int64), np.array([7]), _preds([5.0]))
    assert pool.ids.dtype == np.int64
    assert pool.steps.dtype == np.int64
    np.testing.assert_array_equal(pool.ids, [1, 2, 5])
    np.testing.assert_array_equal(pool.steps, [10, 20, 7])


def test_extend_multiclass_predictions():
    pool = FinishedPool()
    p1 = {'val': torch.rand(2, 6, 3), 'test': torch.rand(2, 5, 3)}
    p2 = {'val': torch.rand(1, 6, 3), 'test': torch.rand(1, 5, 3)}
    pool.extend(np.array([0, 1]), np.array([1, 1]), p1)
    pool.extend(np.array([2]), np.array([2]), p2)
    torch.testing.assert_close(
        pool.predictions['val'], torch.cat([p1['val'], p2['val']])
    )
    torch.testing.assert_close(
        pool.predictions['test'], torch.cat([p1['test'], p2['test']])
    )


def test_extend_with_zero_members_is_a_noop_on_contents():
    pool = FinishedPool()
    pool.extend(np.array([4]), np.array([9]), _preds([4.0]))
    pool.extend(np.zeros(0, np.int64), np.zeros(0, np.int64), _preds([]))
    assert len(pool) == 1
    assert pool.predictions['val'].shape == (1, 4)


def test_mismatched_parts_raise():
    pool = FinishedPool()
    pool.extend(np.array([0]), np.array([1]), _preds([0.0]))
    with pytest.raises(ValueError, match='same parts'):
        pool.extend(np.array([1]), np.array([1]), _preds([1.0], with_test=False))
    extra = _preds([1.0])
    extra_parts = {**extra, 'other': extra['val']}
    with pytest.raises(ValueError, match='same parts'):
        pool.extend(np.array([1]), np.array([1]), extra_parts)
    # A failed extend leaves the pool untouched.
    assert len(pool) == 1
    assert set(pool.predictions) == {'train', 'val', 'test'}


def test_row_count_mismatch_raises():
    pool = FinishedPool()
    with pytest.raises(ValueError):
        pool.extend(np.array([0, 1]), np.array([1, 1]), _preds([0.0]))
    with pytest.raises(ValueError):
        pool.extend(np.array([0, 1]), np.array([1]), _preds([0.0, 1.0]))
    assert len(pool) == 0 and pool.predictions == {}


def test_first_extend_keeps_tensors_and_device():
    pool = FinishedPool()
    preds = _preds([1.0, 2.0])
    pool.extend(np.array([0, 1]), np.array([3, 4]), preds)
    assert pool.predictions is not preds  # the caller's dict is not aliased
    for part, value in preds.items():
        assert pool.predictions[part] is value
        assert pool.predictions[part].device == value.device


@pytest.mark.gpu
def test_predictions_stay_on_cuda(cuda_device):
    pool = FinishedPool()
    device = cuda_device
    pool.extend(
        np.array([0]),
        np.array([1]),
        {k: v.to(device) for k, v in _preds([0.0]).items()},
    )
    pool.extend(
        np.array([1]),
        np.array([2]),
        {k: v.to(device) for k, v in _preds([1.0]).items()},
    )
    for value in pool.predictions.values():
        assert value.device.type == 'cuda'
        assert value.shape == (2, 4)
