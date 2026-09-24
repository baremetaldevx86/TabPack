"""Tests for OnlineGreedyEnsemble (a26).

Most tests use a transparent local score: minus the mean squared distance of a
prediction to a fixed target vector (higher is better), so the greedy choice and the
acceptance decision can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.metrics import compute_metrics, make_score_fn
from tabpack_repro.types import TaskType

TASK = TaskInfo(type_=TaskType.REGRESSION, score='rmse')
TARGET = torch.zeros(4)
# Individually as good as each other (MSE 0.25), perfect when averaged.
PLUS = torch.tensor([0.5, -0.5, 0.5, -0.5])
MINUS = -PLUS


def neg_mse(predictions: torch.Tensor) -> torch.Tensor:
    """(M, N) -> (M,): minus the MSE to TARGET."""
    return -(predictions - TARGET).square().mean(dim=1)


def ids(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def preds(*rows: torch.Tensor, test_shift: float = 10.0) -> dict[str, torch.Tensor]:
    """val = the given rows; test = the same rows shifted (to tell the parts apart)."""
    val = torch.stack(rows) if rows else torch.zeros(0, 4)
    return {'val': val, 'test': val + test_shift}


NO_FINISHED = {
    'finished_ids': ids(),
    'finished_steps': ids(),
    'finished_predictions': {},
}


def make_ensemble(*, patience: int = 2, max_ensemble_size: int | None = None):
    return OnlineGreedyEnsemble(
        score_fn=neg_mse,
        task=TASK,
        max_ensemble_size=max_ensemble_size,
        patience=patience,
    )


def update_running(ensemble, running_ids, running_steps, *rows, **finished):
    kwargs = {**NO_FINISHED, **finished}
    return ensemble.update(
        running_ids=running_ids,
        running_steps=running_steps,
        running_predictions=preds(*rows),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Acceptance and patience
# ---------------------------------------------------------------------------


def test_initial_state():
    ensemble = make_ensemble(patience=3)
    assert ensemble.is_running
    assert ensemble.score is None
    assert ensemble.ids.dtype == np.int64 and len(ensemble.ids) == 0
    assert ensemble.steps.dtype == np.int64 and len(ensemble.steps) == 0
    assert ensemble.predictions() == {}


def test_first_update_is_accepted_even_if_bad():
    ensemble = make_ensemble()
    assert update_running(ensemble, ids(3), ids(7), torch.full((4,), 5.0))
    np.testing.assert_array_equal(ensemble.ids, [3])
    np.testing.assert_array_equal(ensemble.steps, [7])
    assert ensemble.score == pytest.approx(-25.0)
    assert ensemble.is_running


def test_first_update_selects_greedy_ensemble_and_averages():
    ensemble = make_ensemble()
    bad = torch.full((4,), 3.0)
    assert update_running(ensemble, ids(0, 1, 2), ids(5, 5, 5), PLUS, bad, MINUS)
    np.testing.assert_array_equal(ensemble.ids, [0, 2])
    np.testing.assert_array_equal(ensemble.steps, [5, 5])
    assert ensemble.score == 0.0
    predictions = ensemble.predictions()
    assert set(predictions) == {'val', 'test'}
    torch.testing.assert_close(predictions['val'], torch.zeros(4))
    torch.testing.assert_close(predictions['test'], torch.full((4,), 10.0))


def test_non_improving_updates_keep_the_ensemble_and_decrement_patience():
    ensemble = make_ensemble(patience=2)
    assert update_running(ensemble, ids(0), ids(1), PLUS)
    snapshot = ensemble.predictions()

    # The same predictions again: equal score is not a strict improvement.
    assert not update_running(ensemble, ids(0), ids(2), PLUS)
    assert ensemble._remaining_patience == 1
    # Worse predictions: the current ensemble entry (id 0, step 1) is kept.
    assert not update_running(ensemble, ids(0), ids(3), PLUS * 2)
    assert ensemble._remaining_patience == 0
    assert ensemble.is_running

    np.testing.assert_array_equal(ensemble.ids, [0])
    np.testing.assert_array_equal(ensemble.steps, [1])
    assert ensemble.score == pytest.approx(-0.25)
    for part, value in ensemble.predictions().items():
        torch.testing.assert_close(value, snapshot[part])


@pytest.mark.parametrize('patience', [0, 1, 3])
def test_is_running_flips_after_patience_plus_one_non_improvements(patience):
    ensemble = make_ensemble(patience=patience)
    assert update_running(ensemble, ids(0), ids(1), PLUS)
    for _ in range(patience):
        assert not update_running(ensemble, ids(0), ids(2), PLUS * 3)
        assert ensemble.is_running
    assert not update_running(ensemble, ids(0), ids(2), PLUS * 3)
    assert not ensemble.is_running
    with pytest.raises(RuntimeError, match='patience'):
        update_running(ensemble, ids(0), ids(3), PLUS)


def test_improvement_resets_patience():
    ensemble = make_ensemble(patience=1)
    assert update_running(ensemble, ids(0), ids(1), PLUS * 2)
    assert not update_running(ensemble, ids(0), ids(2), PLUS * 3)
    assert ensemble._remaining_patience == 0
    assert update_running(ensemble, ids(0), ids(3), PLUS)
    assert ensemble._remaining_patience == 1
    np.testing.assert_array_equal(ensemble.steps, [3])
    assert ensemble.score == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# The pool: current ensemble + finished + running (latest)
# ---------------------------------------------------------------------------


def test_snapshot_of_running_member_persists_after_its_predictions_change():
    ensemble = make_ensemble()
    running = preds(PLUS * 1.2, torch.full((4,), 3.0))
    assert ensemble.update(
        running_ids=ids(0, 1),
        running_steps=ids(1, 1),
        running_predictions=running,
        **NO_FINISHED,
    )
    np.testing.assert_array_equal(ensemble.ids, [0])

    # The caller reuses its buffers: in-place changes must not leak in.
    running['val'].fill_(100.0)
    running['test'].fill_(100.0)
    torch.testing.assert_close(ensemble.predictions()['val'], PLUS * 1.2)

    # Member 0 got worse at step 2, member 1 now complements the step-1 snapshot of
    # member 0, which is only available through the current ensemble.
    assert update_running(ensemble, ids(0, 1), ids(2, 2), PLUS * 3, MINUS * 1.2)
    np.testing.assert_array_equal(ensemble.ids, [0, 1])
    np.testing.assert_array_equal(ensemble.steps, [1, 2])
    assert ensemble.score == 0.0
    torch.testing.assert_close(ensemble.predictions()['test'], torch.full((4,), 10.0))


def test_same_id_with_different_steps_can_coexist():
    ensemble = make_ensemble()
    assert update_running(ensemble, ids(4), ids(1), PLUS)
    # Individually worse (MSE 0.36), but averaging it with the step-1 snapshot of the
    # same member is better than the snapshot alone.
    assert update_running(ensemble, ids(4), ids(2), MINUS * 1.2)
    np.testing.assert_array_equal(ensemble.ids, [4, 4])
    np.testing.assert_array_equal(ensemble.steps, [1, 2])
    report = ensemble.report({'val': np.zeros(4), 'test': np.zeros(4)})
    assert report['size'] == 2
    assert report['n_unique'] == 1


def test_finished_members_participate():
    ensemble = make_ensemble()
    assert update_running(ensemble, ids(0), ids(1), PLUS * 2)
    assert update_running(
        ensemble,
        ids(0),
        ids(2),
        PLUS * 3,
        finished_ids=ids(9),
        finished_steps=ids(6),
        finished_predictions=preds(MINUS * 2),
    )
    np.testing.assert_array_equal(ensemble.ids, [0, 9])
    np.testing.assert_array_equal(ensemble.steps, [1, 6])
    assert ensemble.score == 0.0


@pytest.mark.parametrize(
    'running_predictions', [{}, preds()], ids=['no-parts', 'zero-rows']
)
def test_pool_of_finished_members_only(running_predictions):
    ensemble = make_ensemble()
    assert ensemble.update(
        running_ids=ids(),
        running_steps=ids(),
        running_predictions=running_predictions,
        finished_ids=ids(1, 2),
        finished_steps=ids(3, 4),
        finished_predictions=preds(PLUS, MINUS),
    )
    np.testing.assert_array_equal(ensemble.ids, [1, 2])
    np.testing.assert_array_equal(ensemble.steps, [3, 4])


def test_pool_order_breaks_ties_current_then_finished_then_running():
    # Equal individual scores: greedy starts from the first entry of the pool.
    ensemble = make_ensemble()
    assert update_running(
        ensemble,
        ids(7),
        ids(1),
        PLUS,
        finished_ids=ids(5),
        finished_steps=ids(1),
        finished_predictions=preds(PLUS),
    )
    # Finished before running.
    np.testing.assert_array_equal(ensemble.ids, [5])

    # The current ensemble comes first: its (5, 1) entry wins the tie against the
    # identical finished entry (6, 2), and the running MINUS completes it.
    assert update_running(
        ensemble,
        ids(7),
        ids(2),
        MINUS,
        finished_ids=ids(6),
        finished_steps=ids(2),
        finished_predictions=preds(PLUS),
    )
    np.testing.assert_array_equal(ensemble.ids, [5, 7])
    np.testing.assert_array_equal(ensemble.steps, [1, 2])


def test_max_ensemble_size_is_respected():
    ensemble = make_ensemble(max_ensemble_size=1)
    assert update_running(ensemble, ids(0, 1), ids(1, 1), PLUS * 1.1, MINUS)
    # Size 2 would be perfect, but the limit is 1: the best individual is kept.
    np.testing.assert_array_equal(ensemble.ids, [1])


def test_parts_are_shared_by_all_sources_and_all_are_stored():
    ensemble = make_ensemble()
    finished = preds(MINUS)
    finished['train'] = torch.ones(1, 3)
    running = preds(PLUS)
    assert ensemble.update(
        running_ids=ids(0),
        running_steps=ids(1),
        running_predictions=running,
        finished_ids=ids(1),
        finished_steps=ids(1),
        finished_predictions=finished,
    )
    np.testing.assert_array_equal(ensemble.ids, [1, 0])
    assert set(ensemble.predictions()) == {'val', 'test'}
    assert set(ensemble._predictions) == {'val', 'test'}
    torch.testing.assert_close(
        ensemble._predictions['test'], torch.stack([MINUS, PLUS]) + 10
    )


def test_returned_arrays_are_copies():
    ensemble = make_ensemble()
    assert update_running(ensemble, ids(0), ids(1), PLUS)
    ensemble.ids[0] = 99
    ensemble.steps[0] = 99
    np.testing.assert_array_equal(ensemble.ids, [0])
    np.testing.assert_array_equal(ensemble.steps, [1])


def test_ids_and_steps_accept_tensors_and_lists():
    ensemble = make_ensemble()
    assert ensemble.update(
        running_ids=torch.tensor([0, 1]),
        running_steps=[2, 2],
        running_predictions=preds(PLUS, MINUS),
        finished_ids=[],
        finished_steps=[],
        finished_predictions={},
    )
    assert ensemble.ids.dtype == np.int64
    np.testing.assert_array_equal(ensemble.ids, [0, 1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'patience': -1}, 'patience'),
        ({'max_ensemble_size': 0}, 'max_ensemble_size'),
    ],
)
def test_constructor_validation(kwargs, error):
    with pytest.raises(ValueError, match=error):
        OnlineGreedyEnsemble(
            score_fn=neg_mse,
            task=TASK,
            **{'patience': 1, 'max_ensemble_size': None, **kwargs},
        )


def test_empty_pool_raises():
    with pytest.raises(ValueError, match='empty'):
        update_running(make_ensemble(), ids(), ids())


def test_mismatched_lengths_raise():
    ensemble = make_ensemble()
    with pytest.raises(ValueError, match='steps'):
        update_running(ensemble, ids(0, 1), ids(1), PLUS, MINUS)
    with pytest.raises(ValueError, match='rows'):
        update_running(ensemble, ids(0, 1), ids(1, 1), PLUS)
    assert ensemble.score is None
    assert ensemble._remaining_patience == 2


def test_missing_val_part_raises():
    ensemble = make_ensemble()
    with pytest.raises(ValueError, match="'val'"):
        ensemble.update(
            running_ids=ids(0),
            running_steps=ids(1),
            running_predictions={'test': PLUS[None]},
            **NO_FINISHED,
        )


# ---------------------------------------------------------------------------
# report() and real metrics
# ---------------------------------------------------------------------------


def test_report_before_first_update():
    report = make_ensemble().report({'val': np.zeros(4), 'test': np.zeros(4)})
    assert report == {
        'ids': [],
        'steps': [],
        'size': 0,
        'n_unique': 0,
        'score_val': None,
        'metrics': {},
    }


def test_report_binclass_with_real_score_fn():
    rng = np.random.default_rng(0)
    task = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)
    y = {part: rng.integers(0, 2, 64) for part in ('val', 'test')}
    ensemble = OnlineGreedyEnsemble(
        score_fn=make_score_fn(torch.as_tensor(y['val']), task),
        task=task,
        max_ensemble_size=4,
        patience=1,
    )
    probs = {
        part: torch.as_tensor(rng.random((6, 64)), dtype=torch.float32)
        for part in ('val', 'test')
    }
    assert ensemble.update(
        running_ids=ids(0, 1, 2),
        running_steps=ids(3, 3, 3),
        running_predictions={k: v[:3] for k, v in probs.items()},
        finished_ids=ids(3, 4, 5),
        finished_steps=ids(1, 2, 3),
        finished_predictions={k: v[3:] for k, v in probs.items()},
    )
    report = ensemble.report(y)
    assert set(report) == {'ids', 'steps', 'size', 'n_unique', 'score_val', 'metrics'}
    assert isinstance(report['ids'], list) and isinstance(report['steps'], list)
    assert report['ids'] == ensemble.ids.tolist()
    assert report['steps'] == ensemble.steps.tolist()
    assert 1 <= report['size'] == len(report['ids']) <= 4
    assert report['n_unique'] == len(set(report['ids']))
    assert report['score_val'] == ensemble.score
    assert set(report['metrics']) == {'val', 'test'}

    # Member i is row i of `probs` (running 0-2, finished 3-5).
    rows = report['ids']
    for part in ('val', 'test'):
        expected = compute_metrics(y[part], probs[part].numpy()[rows].mean(0), task)
        assert report['metrics'][part] == pytest.approx(expected)
    assert report['metrics']['val']['score'] == pytest.approx(report['score_val'])


def test_report_requires_labels_of_every_stored_part():
    ensemble = make_ensemble()
    assert update_running(ensemble, ids(0), ids(1), PLUS)
    with pytest.raises(KeyError, match='test'):
        ensemble.report({'val': np.zeros(4)})


def test_multiclass_predictions():
    rng = np.random.default_rng(1)
    task = TaskInfo(type_=TaskType.MULTICLASS, score='accuracy', n_classes=3)
    y = {part: rng.integers(0, 3, 32) for part in ('val', 'test')}
    probs = {
        part: torch.softmax(
            torch.as_tensor(rng.standard_normal((5, 32, 3))), -1
        ).float()
        for part in ('val', 'test')
    }
    ensemble = OnlineGreedyEnsemble(
        score_fn=make_score_fn(torch.as_tensor(y['val']), task),
        task=task,
        max_ensemble_size=None,
        patience=0,
    )
    assert ensemble.update(
        running_ids=ids(0, 1, 2, 3, 4),
        running_steps=ids(1, 1, 1, 1, 1),
        running_predictions=probs,
        **NO_FINISHED,
    )
    assert ensemble.predictions()['test'].shape == (32, 3)
    report = ensemble.report(y)
    assert report['metrics']['val']['score'] == pytest.approx(ensemble.score)


def test_random_training_run_invariants():
    """A long simulated run: the val score never decreases, the stored snapshots
    always reproduce the stored score, and patience bookkeeping matches."""
    rng = np.random.default_rng(2)
    task = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)
    y_val = rng.integers(0, 2, 50)
    score_fn = make_score_fn(torch.as_tensor(y_val), task)
    patience = 3
    ensemble = OnlineGreedyEnsemble(
        score_fn=score_fn, task=task, max_ensemble_size=8, patience=patience
    )
    running_ids = np.arange(6)
    finished = {'ids': [], 'steps': [], 'val': [], 'test': []}
    expected_remaining = patience
    previous_score = -np.inf
    for step in range(1, 40):
        if not ensemble.is_running:
            break
        latest = {
            part: torch.as_tensor(
                rng.random((len(running_ids), 50)), dtype=torch.float32
            )
            for part in ('val', 'test')
        }
        if step % 5 == 0 and len(running_ids) > 1:
            finished['ids'].append(int(running_ids[0]))
            finished['steps'].append(step)
            for part in ('val', 'test'):
                finished[part].append(latest[part][0])
                latest[part] = latest[part][1:]
            running_ids = running_ids[1:]
        improved = ensemble.update(
            running_ids=running_ids,
            running_steps=np.full(len(running_ids), step),
            running_predictions=latest,
            finished_ids=np.array(finished['ids'], dtype=np.int64),
            finished_steps=np.array(finished['steps'], dtype=np.int64),
            finished_predictions=(
                {part: torch.stack(finished[part]) for part in ('val', 'test')}
                if finished['ids']
                else {}
            ),
        )
        if improved:
            assert ensemble.score > previous_score
            previous_score = ensemble.score
            expected_remaining = patience
        else:
            assert ensemble.score == previous_score
            expected_remaining -= 1
        assert ensemble._remaining_patience == expected_remaining
        assert ensemble.is_running == (expected_remaining >= 0)
        assert len(ensemble.ids) == len(ensemble.steps) <= 8
        torch.testing.assert_close(
            ensemble.predictions()['val'], ensemble._predictions['val'].mean(0)
        )
        assert score_fn(ensemble.predictions()['val'][None]).item() == ensemble.score


def test_stored_predictions_are_detached():
    ensemble = make_ensemble()
    running = preds(PLUS.clone().requires_grad_(), MINUS)
    assert ensemble.update(
        running_ids=ids(0, 1),
        running_steps=ids(1, 1),
        running_predictions=running,
        **NO_FINISHED,
    )
    assert not ensemble._predictions['val'].requires_grad
    assert not ensemble.predictions()['val'].requires_grad


@pytest.mark.gpu
def test_pool_moves_to_the_running_device(cuda_device):
    ensemble = OnlineGreedyEnsemble(
        # TARGET is zero: a device-agnostic neg_mse.
        score_fn=lambda p: -p.square().mean(dim=1),
        task=TASK,
        max_ensemble_size=None,
        patience=1,
    )
    running = {k: v.to(cuda_device) for k, v in preds(PLUS).items()}
    assert ensemble.update(
        running_ids=ids(0),
        running_steps=ids(2),
        running_predictions=running,
        finished_ids=ids(1),
        finished_steps=ids(1),
        finished_predictions=preds(MINUS),  # on the CPU
    )
    np.testing.assert_array_equal(ensemble.ids, [1, 0])
    for value in ensemble.predictions().values():
        assert value.device.type == 'cuda'
    report = ensemble.report({'val': np.zeros(4), 'test': np.zeros(4)})
    assert report['metrics']['val']['rmse'] == 0.0
