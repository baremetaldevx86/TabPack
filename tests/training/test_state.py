"""Tests for PackState (a20): official StatePack semantics, host/device layout."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import Tensor

from tabpack_repro.training.state import PackState

K = 4
N = 6
C = 3


def _predictions(fill: Tensor, device: torch.device | str = 'cpu') -> dict:
    """Per-member constant predictions: member i is filled with fill[i]."""
    fill = fill.to(device=device, dtype=torch.float32)
    k = len(fill)
    return {
        'val': fill[:, None].expand(k, N).clone(),
        'test': fill[:, None, None].expand(k, N, C).clone(),
    }


def _model_state(fill: Tensor, device: torch.device | str = 'cpu') -> dict:
    fill = fill.to(device=device, dtype=torch.float32)
    k = len(fill)
    return {
        'w': fill[:, None, None].expand(k, 2, 5).clone(),
        'b': fill[:, None].expand(k, 5).clone(),
        'count': fill.to(torch.int64).clone(),
    }


def _update(
    state: PackState, scores: list[float], fill: list[float], device: str = 'cpu'
) -> Tensor:
    fill_t = torch.tensor(fill)
    return state.update(
        torch.tensor(scores, dtype=torch.float32, device=device),
        _predictions(fill_t, device),
        _model_state(fill_t, device),
    )


def _member_fill(state: PackState) -> np.ndarray:
    """The fill value of each member's best snapshot (checks all fields agree)."""
    fill = state.best_model_state['b'][:, 0].cpu().numpy()
    for x in (*state.best_predictions.values(), *state.best_model_state.values()):
        x = x.cpu().float().reshape(state.pack_size, -1).numpy()
        np.testing.assert_array_equal(x, np.broadcast_to(fill[:, None], x.shape))
    return fill


def test_init() -> None:
    state = PackState(K)
    assert state.pack_size == K
    np.testing.assert_array_equal(state.ids, np.arange(K))
    assert state.ids.dtype == np.int64
    np.testing.assert_array_equal(state.steps, 0)
    np.testing.assert_array_equal(state.n_bad_updates, 0)
    assert np.all(state.best_val_score == -np.inf)
    np.testing.assert_array_equal(state.best_step, -1)
    assert state.best_predictions == {}
    assert state.best_model_state == {}
    state.validate()
    PackState(0).validate()


def test_step_increments_all_members() -> None:
    state = PackState(K)
    for _ in range(3):
        state.step()
    np.testing.assert_array_equal(state.steps, 3)
    assert state.steps.dtype == np.int64
    state.validate()


def test_first_update_improves_all() -> None:
    state = PackState(K)
    state.step()
    # Even -inf and NaN scores count as an improvement at the first evaluation.
    improved = _update(state, [0.5, -np.inf, np.nan, 0.1], [1, 2, 3, 4])
    assert improved.dtype == torch.bool
    assert improved.device.type == 'cpu'
    assert improved.tolist() == [True] * K
    np.testing.assert_array_equal(state.best_step, 1)
    np.testing.assert_array_equal(state.n_bad_updates, 0)
    np.testing.assert_array_equal(
        state.best_val_score, np.array([0.5, -np.inf, np.nan, 0.1], dtype=np.float32)
    )
    np.testing.assert_array_equal(_member_fill(state), [1, 2, 3, 4])
    assert state.best_predictions['val'].shape == (K, N)
    assert state.best_predictions['test'].shape == (K, N, C)
    assert state.best_model_state['count'].dtype == torch.int64
    state.validate()


def test_first_update_clones_inputs() -> None:
    state = PackState(K)
    state.step()
    fill = torch.arange(K, dtype=torch.float32)
    predictions = _predictions(fill)
    model_state = _model_state(fill)
    state.update(torch.zeros(K), predictions, model_state)
    for x in (*predictions.values(), *model_state.values()):
        x.add_(100)
    np.testing.assert_array_equal(_member_fill(state), [0, 1, 2, 3])


def test_improvement_is_strict() -> None:
    state = PackState(3)
    state.step()
    _update(state, [0.5, 0.5, 0.5], [1, 1, 1])
    state.step()
    improved = _update(state, [0.5, 0.6, 0.4], [2, 2, 2])
    assert improved.tolist() == [False, True, False]
    np.testing.assert_array_equal(state.best_val_score, np.float32([0.5, 0.6, 0.5]))
    np.testing.assert_array_equal(state.best_step, [1, 2, 1])
    np.testing.assert_array_equal(state.n_bad_updates, [1, 0, 1])
    np.testing.assert_array_equal(_member_fill(state), [1, 2, 1])
    state.validate()


def test_nan_best_is_never_improved() -> None:
    # Official semantics: `score > nan` is False, so the member only accumulates
    # bad updates (and is eventually stopped by patience).
    state = PackState(2)
    state.step()
    _update(state, [np.nan, 0.0], [1, 1])
    for _ in range(3):
        state.step()
        improved = _update(state, [1.0, np.nan], [2, 2])
        assert improved.tolist() == [False, False]
    np.testing.assert_array_equal(state.n_bad_updates, [3, 3])
    np.testing.assert_array_equal(state.best_step, [1, 1])


def test_n_bad_updates_count_and_reset() -> None:
    state = PackState(2)
    history = [
        # scores, expected n_bad_updates, expected best_step
        ([0.1, 0.1], [0, 0], [1, 1]),
        ([0.0, 0.2], [1, 0], [1, 2]),
        ([0.1, 0.2], [2, 1], [1, 2]),
        ([0.05, 0.1], [3, 2], [1, 2]),
        ([0.3, 0.1], [0, 3], [5, 2]),
        ([0.3, 0.25], [1, 0], [5, 6]),
    ]
    for i, (scores, n_bad, best_step) in enumerate(history):
        state.step()
        improved = _update(state, scores, [i, i])
        np.testing.assert_array_equal(improved.numpy(), np.array(n_bad) == 0)
        np.testing.assert_array_equal(state.n_bad_updates, n_bad)
        np.testing.assert_array_equal(state.best_step, best_step)
        state.validate()
    # Each member's snapshot comes from its last improving update.
    np.testing.assert_array_equal(_member_fill(state), [4, 5])
    np.testing.assert_array_equal(state.best_val_score, np.float32([0.3, 0.25]))


def test_best_slices_follow_improvements_only() -> None:
    rng = np.random.default_rng(0)
    state = PackState(K)
    expected_fill = np.full(K, np.nan)
    expected_score = np.full(K, -np.inf)
    storage = None
    for update in range(20):
        state.step()
        scores = rng.integers(0, 5, size=K).astype(np.float32)
        fill = np.full(K, float(update + 1))
        improved = _update(state, scores.tolist(), fill.tolist()).numpy()
        expected_improved = (scores > expected_score) | (update == 0)
        np.testing.assert_array_equal(improved, expected_improved)
        expected_fill[expected_improved] = fill[expected_improved]
        expected_score[expected_improved] = scores[expected_improved]
        np.testing.assert_array_equal(_member_fill(state), expected_fill)
        np.testing.assert_array_equal(state.best_val_score, expected_score)
        # After the first update, the best tensors are updated in place.
        ptrs = {
            name: x.data_ptr()
            for name, x in (
                *state.best_predictions.items(),
                *state.best_model_state.items(),
            )
        }
        if storage is not None:
            assert ptrs == storage
        storage = ptrs
        state.validate()


def test_update_does_not_alias_later_inputs() -> None:
    state = PackState(2)
    state.step()
    _update(state, [0.0, 0.0], [1, 1])
    state.step()
    fill = torch.tensor([2.0, 2.0])
    predictions = _predictions(fill)
    model_state = _model_state(fill)
    state.update(torch.tensor([1.0, 1.0]), predictions, model_state)
    for x in (*predictions.values(), *model_state.values()):
        x.add_(100)
    np.testing.assert_array_equal(_member_fill(state), [2, 2])


def test_update_checks_shapes_and_keys() -> None:
    state = PackState(K)
    state.step()
    with pytest.raises(AssertionError):
        _update(state, [0.0] * (K - 1), [1] * K)
    _update(state, [0.0] * K, [1] * K)
    state.step()
    fill = torch.ones(K)
    predictions = _predictions(fill)
    del predictions['test']
    with pytest.raises(AssertionError):
        state.update(
            torch.tensor([0.0, 1.0, 0.0, 0.0]), predictions, _model_state(fill)
        )
    model_state = _model_state(fill)
    model_state['b'] = model_state['b'][:, :2]
    with pytest.raises(AssertionError):
        state.update(
            torch.tensor([0.0, 1.0, 0.0, 0.0]), _predictions(fill), model_state
        )
    # A rejected update leaves the state untouched.
    np.testing.assert_array_equal(state.n_bad_updates, 0)
    np.testing.assert_array_equal(state.best_step, 1)
    np.testing.assert_array_equal(state.best_val_score, 0.0)
    state.validate()


def _fill_state() -> PackState:
    """K members with distinct values in every per-member field."""
    state = PackState(K)
    state.step()
    _update(state, [0, 0, 0, 0], [0, 1, 2, 3])
    state.step()
    _update(state, [1, 0, 1, 0], [10, 11, 12, 13])
    state.step()
    _update(state, [2, 0, 0, 0], [20, 21, 22, 23])
    state.steps += np.arange(K)  # distinct steps too
    state.validate()
    np.testing.assert_array_equal(state.n_bad_updates, [0, 2, 1, 2])
    np.testing.assert_array_equal(state.best_step, [3, 1, 2, 1])
    np.testing.assert_array_equal(_member_fill(state), [20, 1, 12, 3])
    return state


@pytest.mark.parametrize(
    'keep', [[0, 1, 2, 3], [2, 0], [3], [1, 3, 2], []], ids=lambda x: str(x)
)
@pytest.mark.parametrize('kind', ['tensor', 'numpy'])
def test_select_keeps_order_and_all_fields(keep: list[int], kind: str) -> None:
    state = _fill_state()
    before = {
        'ids': state.ids.copy(),
        'steps': state.steps.copy(),
        'n_bad_updates': state.n_bad_updates.copy(),
        'best_val_score': state.best_val_score.copy(),
        'best_step': state.best_step.copy(),
    }
    before_tensors = {
        **{f'p.{k}': v.clone() for k, v in state.best_predictions.items()},
        **{f's.{k}': v.clone() for k, v in state.best_model_state.items()},
    }
    keep_idx = (
        torch.tensor(keep, dtype=torch.int64)
        if kind == 'tensor'
        else np.array(keep, dtype=np.int64)
    )
    state.select_(keep_idx)
    state.validate()
    assert state.pack_size == len(keep)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name), value[keep])
    for key, value in before_tensors.items():
        group = state.best_predictions if key[0] == 'p' else state.best_model_state
        torch.testing.assert_close(group[key[2:]], value[keep], rtol=0, atol=0)
    # The kept members keep their ids and continue to be tracked correctly.
    state.step()
    fill = [100.0 + i for i in range(len(keep))]
    scores = [1.0] * len(keep)
    improved = _update(state, scores, fill).numpy()
    np.testing.assert_array_equal(improved, before['best_val_score'][keep] < 1.0)
    state.validate()


def test_select_before_first_update() -> None:
    state = PackState(K)
    state.select_(torch.tensor([3, 1]))
    np.testing.assert_array_equal(state.ids, [3, 1])
    state.validate()
    state.step()
    improved = _update(state, [0.0, 0.0], [7, 8])
    assert improved.tolist() == [True, True]
    np.testing.assert_array_equal(_member_fill(state), [7, 8])
    state.validate()


def test_select_rejects_invalid_indices() -> None:
    for bad in ([0, 0], [K], [-1], [0.0]):
        state = PackState(K)
        with pytest.raises(AssertionError):
            state.select_(torch.tensor(bad))


def test_validate_detects_inconsistency() -> None:
    state = _fill_state()
    state.best_predictions['val'] = state.best_predictions['val'][:2]
    with pytest.raises(AssertionError):
        state.validate()

    state = _fill_state()
    state.steps = state.steps[:2]
    with pytest.raises(AssertionError):
        state.validate()

    state = _fill_state()
    state.ids[1] = state.ids[0]
    with pytest.raises(AssertionError):
        state.validate()

    state = _fill_state()
    state.best_step[0] = state.steps[0] + 1
    with pytest.raises(AssertionError):
        state.validate()


def test_empty_pack() -> None:
    state = PackState(K)
    state.step()
    _update(state, [0.0] * K, [1] * K)
    state.select_(torch.zeros(0, dtype=torch.int64))
    state.validate()
    assert state.pack_size == 0
    improved = state.update(
        torch.zeros(0),
        _predictions(torch.zeros(0)),
        _model_state(torch.zeros(0)),
    )
    assert improved.shape == (0,)
    state.validate()


@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:Synchronization debug mode is a prototype')
def test_cuda(cuda_device: torch.device) -> None:
    state = PackState(K)
    state.step()
    improved = _update(state, [0, 0, 0, 0], [0, 1, 2, 3], device='cuda')
    assert improved.device.type == 'cpu'
    for x in (*state.best_predictions.values(), *state.best_model_state.values()):
        assert x.device.type == 'cuda'
    state.step()
    improved = _update(state, [1, 0, 1, 0], [10, 11, 12, 13], device='cuda')
    assert improved.tolist() == [True, False, True, False]
    np.testing.assert_array_equal(_member_fill(state), [10, 1, 12, 3])

    # Exactly one synchronizing device -> host transfer per partial update.
    state.step()
    scores = torch.tensor([2.0, 0.0, 0.0, 1.0], device=cuda_device)
    fill = torch.tensor([20.0, 21.0, 22.0, 23.0])
    predictions = _predictions(fill, cuda_device)
    model_state = _model_state(fill, cuda_device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('warn')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            improved = state.update(scores, predictions, model_state)
    finally:
        torch.cuda.set_sync_debug_mode('default')
    syncs = [w for w in caught if 'synchroniz' in str(w.message).lower()]
    assert len(syncs) == 1, [str(w.message) for w in syncs]
    assert improved.tolist() == [True, False, False, True]
    np.testing.assert_array_equal(_member_fill(state), [20, 1, 12, 23])
    state.validate()

    # select_ with a device index tensor.
    state.select_(torch.tensor([3, 0], device=cuda_device))
    state.validate()
    np.testing.assert_array_equal(state.ids, [3, 0])
    np.testing.assert_array_equal(_member_fill(state), [23, 20])
    for x in (*state.best_predictions.values(), *state.best_model_state.values()):
        assert x.device.type == 'cuda'
