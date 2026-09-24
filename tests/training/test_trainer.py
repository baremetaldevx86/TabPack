"""Tests of the pack training loop (a23): train_pack on small synthetic data (CPU)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from _helpers import make_synthetic_dataset

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.metrics import compute_metrics, make_score_fn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups
from tabpack_repro.training import trainer as trainer_module
from tabpack_repro.training.batches import epoch_size
from tabpack_repro.training.trainer import PackTrainResult, train_pack
from tabpack_repro.types import PARTS, TaskType

BATCH_SIZE = 64
N_TRAIN = 256
EPOCH_SIZE = epoch_size(N_TRAIN, BATCH_SIZE)
HISTORY_KEYS = {
    'epoch',
    'step',
    'time',
    'n_running',
    'n_finished',
    'train_loss',
    'ensemble_val',
    'ensemble_test',
}


def _dataset(**kwargs: Any) -> PreparedDataset:
    return make_synthetic_dataset(n_train=N_TRAIN, n_val=96, n_test=96, **kwargs)


def _setup(
    dataset: PreparedDataset,
    *,
    pack_size: int,
    lr: float | list[float] = 3e-3,
    n_blocks: int | list[int] = 1,
    dropout: float | list[float] = 0.0,
    init_seed: int = 0,
    device: torch.device | None = None,
) -> tuple[ModelPack, AdamWPack, list[int]]:
    """Model + optimizer; also records the pack size of every model call."""
    torch.manual_seed(init_seed)
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=dataset.task.n_classes,
        pack_size=pack_size,
        d_block=16,
        n_blocks=n_blocks,
        dropout=dropout,
    ).to(device)
    optimizer = AdamWPack(model.parameters(), lr=lr, pack_size=pack_size)
    call_pack_sizes: list[int] = []
    model.register_forward_pre_hook(
        lambda module, args: call_pack_sizes.append(module.pack_size)
    )
    return model, optimizer, call_pack_sizes


def _train(
    dataset: PreparedDataset,
    *,
    pack_size: int = 3,
    patience: int = 2,
    max_epochs: int = 20,
    seed: int = 0,
    setup_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[PackTrainResult, list[int]]:
    model, optimizer, call_pack_sizes = _setup(
        dataset, pack_size=pack_size, **(setup_kwargs or {})
    )
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        patience=patience,
        max_epochs=max_epochs,
        seed=seed,
        **kwargs,
    )
    return result, call_pack_sizes


def _make_ensemble(
    dataset: PreparedDataset, *, patience: int, max_ensemble_size: int | None = 4
) -> OnlineGreedyEnsemble:
    return OnlineGreedyEnsemble(
        score_fn=make_score_fn(dataset.y['val'], dataset.task),
        task=dataset.task,
        max_ensemble_size=max_ensemble_size,
        patience=patience,
    )


def _check_result(
    result: PackTrainResult, dataset: PreparedDataset, *, n_finished: int
) -> None:
    """Structural invariants of a PackTrainResult."""
    assert result.ids.dtype == np.int64 and result.ids.shape == (n_finished,)
    assert result.best_steps.dtype == np.int64
    assert result.best_steps.shape == (n_finished,)
    assert len(set(result.ids.tolist())) == n_finished
    # Members are evaluated after every epoch, so a best step is a whole epoch.
    assert np.all(result.best_steps >= EPOCH_SIZE)
    assert np.all(result.best_steps % EPOCH_SIZE == 0)
    assert np.all(result.best_steps <= result.n_steps)
    assert result.n_steps == result.n_epochs * EPOCH_SIZE
    assert result.time_sec > 0

    assert set(result.predictions) == set(PARTS)
    for part in PARTS:
        predictions = result.predictions[part]
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (n_finished, dataset.size(part))
        assert np.all((predictions >= 0) & (predictions <= 1))

    assert len(result.members) == n_finished
    y_true = {part: dataset.y[part].numpy() for part in PARTS}
    for i, member in enumerate(result.members):
        assert member['id'] == result.ids[i]
        assert member['best_step'] == result.best_steps[i]
        assert set(member['metrics']) == set(PARTS)
        for part in PARTS:
            expected = compute_metrics(
                y_true[part], result.predictions[part][i], dataset.task
            )
            assert member['metrics'][part] == pytest.approx(expected)

    assert len(result.history) == result.n_epochs
    for epoch, record in enumerate(result.history, 1):
        assert set(record) == HISTORY_KEYS
        assert record['epoch'] == epoch
        assert record['step'] == epoch * EPOCH_SIZE
        assert np.isfinite(record['train_loss'])
    n_finished_history = [r['n_finished'] for r in result.history]
    assert n_finished_history == sorted(n_finished_history)
    assert n_finished_history[-1] == n_finished


class _EvaluationRecorder:
    """Wraps trainer.evaluate_pack to record val predictions/scores per epoch."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.val_scores: list[np.ndarray] = []
        self.val_predictions: list[np.ndarray] = []
        evaluate_pack = trainer_module.evaluate_pack

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            scores, predictions = evaluate_pack(*args, **kwargs)
            self.val_scores.append(scores['val'].cpu().numpy().copy())
            self.val_predictions.append(predictions['val'].cpu().numpy().copy())
            return scores, predictions

        monkeypatch.setattr(trainer_module, 'evaluate_pack', wrapper)


def _scripted_stopping(
    monkeypatch: pytest.MonkeyPatch, schedule: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Replace compute_stop_idx: after epoch e, stop the members with ids
    schedule[e]. Returns {epoch: running ids before the stop}."""
    running = {'ids': None}
    epoch = {'value': 0}
    running_by_epoch: dict[int, list[int]] = {}

    def fake_compute_stop_idx(
        n_bad_updates: np.ndarray, steps: np.ndarray, **kwargs: Any
    ) -> np.ndarray | None:
        del kwargs
        if running['ids'] is None:
            running['ids'] = list(range(len(steps)))
        assert len(running['ids']) == len(steps)
        epoch['value'] += 1
        ids = running['ids']
        running_by_epoch[epoch['value']] = list(ids)
        to_stop = schedule.get(epoch['value'], [])
        running['ids'] = [i for i in ids if i not in to_stop]
        idx = [ids.index(i) for i in to_stop]
        return np.array(sorted(idx), dtype=np.int64) if idx else None

    monkeypatch.setattr(trainer_module, 'compute_stop_idx', fake_compute_stop_idx)
    return running_by_epoch


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Training behaviour
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def test_loss_decreases() -> None:
    dataset = _dataset()
    result, _ = _train(
        dataset, pack_size=3, patience=-1, max_epochs=8, setup_kwargs={'lr': 1e-2}
    )
    losses = [record['train_loss'] for record in result.history]
    assert len(losses) == 8
    assert losses[-1] < 0.8 * losses[0]
    # The members learned something: better than the majority class on val.
    y_val = dataset.y['val'].numpy()
    majority = max(y_val.mean(), 1 - y_val.mean())
    accuracies = [member['metrics']['val']['accuracy'] for member in result.members]
    assert max(accuracies) > majority


def test_all_members_finish_with_small_patience() -> None:
    dataset = _dataset()
    result, call_pack_sizes = _train(
        dataset, pack_size=4, patience=1, max_epochs=100, setup_kwargs={'lr': 1e-2}
    )
    # Early stopping (not the epoch limit) ended the run.
    assert result.n_epochs < 100
    _check_result(result, dataset, n_finished=4)
    assert sorted(result.ids.tolist()) == [0, 1, 2, 3]
    assert result.history[-1]['n_running'] == 0
    assert result.online_ensemble is None
    assert all(r['ensemble_val'] is None for r in result.history)
    assert all(r['ensemble_test'] is None for r in result.history)
    assert min(call_pack_sizes) > 0


def test_max_epochs_is_respected() -> None:
    dataset = _dataset()
    result, _ = _train(dataset, pack_size=3, patience=-1, max_epochs=3)
    assert result.n_epochs == 3
    assert result.n_steps == 3 * EPOCH_SIZE
    _check_result(result, dataset, n_finished=3)
    # Everyone stops after the last epoch, in pack order.
    assert result.ids.tolist() == [0, 1, 2]
    assert [r['n_running'] for r in result.history] == [3, 3, 0]


def test_finished_members_are_restored_to_their_best_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dropout makes the val score fluctuate, so best epoch != last epoch is likely;
    # with patience=-1 every member runs to max_epochs, so pack order == ids.
    dataset = _dataset()
    recorder = _EvaluationRecorder(monkeypatch)
    result, _ = _train(
        dataset,
        pack_size=4,
        patience=-1,
        max_epochs=6,
        setup_kwargs={'dropout': 0.3, 'lr': 2e-2},
    )
    _check_result(result, dataset, n_finished=4)
    val_scores = np.stack(recorder.val_scores)  # (epochs, K)
    val_predictions = np.stack(recorder.val_predictions)  # (epochs, K, N)
    for i, member_id in enumerate(result.ids):
        # Strict improvement => the FIRST epoch with the best score is kept.
        best_epoch = int(np.argmax(val_scores[:, member_id]))
        assert result.best_steps[i] == (best_epoch + 1) * EPOCH_SIZE
        np.testing.assert_allclose(
            result.predictions['val'][i],
            val_predictions[best_epoch, member_id],
            rtol=1e-5,
            atol=1e-6,
        )
        assert result.members[i]['metrics']['val']['score'] == pytest.approx(
            val_scores[best_epoch, member_id]
        )
    # The test does not depend on luck: some member's best epoch is not the last.
    assert np.any(result.best_steps < result.n_steps)


def test_members_stop_at_different_epochs() -> None:
    # Members with lr=0 never improve after their first evaluation, so with
    # patience=1 they stop after epoch 3 (bad updates at epochs 2 and 3), while the
    # others keep training in a smaller pack.
    dataset = _dataset()
    lr = [0.0, 1e-2, 0.0, 1e-2, 1e-2]
    result, call_pack_sizes = _train(
        dataset,
        pack_size=5,
        patience=1,
        max_epochs=12,
        setup_kwargs={'lr': lr, 'n_blocks': [1, 2, 1, 2, 1]},
    )
    _check_result(result, dataset, n_finished=5)
    assert sorted(result.ids.tolist()) == [0, 1, 2, 3, 4]
    frozen = {0, 2}
    for member_id, best_step in zip(result.ids, result.best_steps, strict=True):
        if member_id in frozen:
            assert best_step == EPOCH_SIZE
    # Members 0 and 2 are the first to finish, after epoch 3.
    first_finished = result.history[2]['n_finished']
    assert set(result.ids[:first_finished].tolist()) >= frozen
    assert result.history[1]['n_finished'] == 0
    # The model was called with a smaller pack after the removal, never with 0.
    assert min(call_pack_sizes) > 0
    if result.n_epochs > 3:
        assert result.history[2]['n_running'] < 5
        assert min(call_pack_sizes) < 5


def test_scripted_member_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove members at scripted epochs; the finished ones must be exactly their
    best-epoch snapshots and the survivors must keep training correctly."""
    dataset = _dataset()
    recorder = _EvaluationRecorder(monkeypatch)
    schedule = {2: [1], 3: [0, 4], 5: [3, 5, 2]}
    running_by_epoch = _scripted_stopping(monkeypatch, schedule)
    result, call_pack_sizes = _train(
        dataset,
        pack_size=6,
        patience=-1,
        max_epochs=100,  # Ignored by the scripted stopping.
        setup_kwargs={
            'dropout': [0.0, 0.2, 0.1, 0.0, 0.3, 0.1],
            'n_blocks': [1, 2, 3, 1, 2, 1],
            'lr': [1e-2, 3e-3, 2e-2, 5e-3, 1e-2, 1e-2],
        },
    )
    assert result.n_epochs == 5
    _check_result(result, dataset, n_finished=6)
    assert result.ids.tolist() == [1, 0, 4, 2, 3, 5]
    assert [r['n_running'] for r in result.history] == [6, 5, 3, 3, 0]
    # Training/evaluation packs of 6, 5 and 3 members; the finished members are
    # predicted in packs of their own (1, 2 and 3 members).
    assert sorted(set(call_pack_sizes)) == [1, 2, 3, 5, 6]

    for i, member_id in enumerate(result.ids.tolist()):
        # Pack position of the member at every epoch it was evaluated.
        epochs = [e for e, ids in running_by_epoch.items() if member_id in ids]
        scores = [
            recorder.val_scores[e - 1][running_by_epoch[e].index(member_id)]
            for e in epochs
        ]
        best = int(np.argmax(scores))
        best_epoch = epochs[best]
        assert result.best_steps[i] == best_epoch * EPOCH_SIZE
        position = running_by_epoch[best_epoch].index(member_id)
        np.testing.assert_allclose(
            result.predictions['val'][i],
            recorder.val_predictions[best_epoch - 1][position],
            rtol=1e-5,
            atol=1e-6,
        )


def test_removal_does_not_disturb_the_other_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Members are independent: removing one must not change the others.

    With full-batch training (one batch = the whole training set) and no dropout,
    a member's trajectory does not depend on the pack it is trained in (only the
    row order of its batch changes, which affects the mean up to rounding). So a
    run where member 1 is removed after epoch 1 must match a run without removal
    for the members 0 and 2 (parameters, Adam moments and state stay aligned).
    """
    dataset = _dataset()
    recorder = _EvaluationRecorder(monkeypatch)
    setup_kwargs = {'lr': [1e-2, 3e-2, 5e-3], 'n_blocks': [2, 1, 3]}

    def run() -> PackTrainResult:
        model, optimizer, _ = _setup(dataset, pack_size=3, **setup_kwargs)
        return train_pack(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            batch_size=N_TRAIN,  # Full batch.
            patience=-1,
            max_epochs=4,
            seed=0,
        )

    reference = run()
    assert reference.ids.tolist() == [0, 1, 2]
    reference_val = recorder.val_predictions[3]  # Epoch 4, members 0, 1, 2.

    _scripted_stopping(monkeypatch, {1: [1], 4: [0, 2]})
    result = run()
    assert result.ids.tolist() == [1, 0, 2]
    assert [r['n_running'] for r in result.history] == [2, 2, 2, 0]
    val = recorder.val_predictions[7]  # Epoch 4 of the second run: members 0, 2.
    assert val.shape[0] == 2
    np.testing.assert_allclose(val, reference_val[[0, 2]], rtol=1e-4, atol=1e-5)
    # The member removed after epoch 1 is identical in both runs up to then.
    np.testing.assert_allclose(
        result.predictions['val'][0],
        recorder.val_predictions[4][1],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        recorder.val_predictions[4], recorder.val_predictions[0], rtol=0, atol=0
    )


def test_muon_optimizer_with_heterogeneous_members() -> None:
    """The TabPack setup: MuonAdamWPack with per-member hyperparameters and
    per-member depths; members finish at different epochs."""
    dataset = _dataset()
    pack_size = 5
    torch.manual_seed(0)
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=dataset.task.n_classes,
        pack_size=pack_size,
        d_block=16,
        n_blocks=[1, 2, 3, 4, 2],
        dropout=[0.0, 0.1, 0.2, 0.0, 0.3],
    )
    optimizer = MuonAdamWPack(
        make_param_groups(model, muon=True),
        lr=[1e-3, 3e-3, 0.0, 2e-3, 1e-3],
        weight_decay=[1e-3, 1e-2, 0.0, 1e-1, 1e-2],
        muon_lr=[1e-2, 3e-2, 0.0, 2e-2, 5e-2],
        pack_size=pack_size,
    )
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        patience=1,
        max_epochs=12,
        seed=0,
        online_ensemble=_make_ensemble(dataset, patience=100),
    )
    _check_result(result, dataset, n_finished=pack_size)
    # Member 2 has zero learning rates: it never improves after epoch 1.
    assert result.best_steps[result.ids.tolist().index(2)] == EPOCH_SIZE
    assert result.history[-1]['train_loss'] < result.history[0]['train_loss']
    assert model.pack_size == 0


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Online ensemble
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def test_online_ensemble_stops_the_run() -> None:
    # Members never stop by themselves (patience=-1, max_epochs=200), so only the
    # online ensemble's patience can end the run; unfinished members are dropped.
    dataset = _dataset()
    ensemble = _make_ensemble(dataset, patience=1)
    result, _ = _train(
        dataset,
        pack_size=4,
        patience=-1,
        max_epochs=200,
        online_ensemble=ensemble,
    )
    assert result.n_epochs < 200
    assert not ensemble.is_running
    _check_result(result, dataset, n_finished=0)
    assert result.history[-1]['n_running'] == 4

    report = result.online_ensemble
    assert report is not None
    assert set(report['ids']) <= {0, 1, 2, 3}
    assert 1 <= report['size'] <= 4
    ensemble_val = [r['ensemble_val'] for r in result.history]
    ensemble_test = [r['ensemble_test'] for r in result.history]
    assert all(x is not None for x in ensemble_val + ensemble_test)
    # Only strict improvements are accepted.
    assert ensemble_val == sorted(ensemble_val)
    assert report['score_val'] == pytest.approx(ensemble_val[-1])
    assert report['metrics']['test']['score'] == pytest.approx(ensemble_test[-1])


def test_online_ensemble_with_finishing_members() -> None:
    dataset = _dataset()
    ensemble = _make_ensemble(dataset, patience=100, max_ensemble_size=None)
    result, _ = _train(
        dataset,
        pack_size=4,
        patience=1,
        max_epochs=30,
        online_ensemble=ensemble,
        setup_kwargs={'lr': [1e-2, 0.0, 3e-3, 1e-2]},
    )
    # The ensemble outlives the members, so the run ends when all have finished.
    assert result.history[-1]['n_running'] == 0
    _check_result(result, dataset, n_finished=4)
    report = result.online_ensemble
    assert report is not None
    assert set(report['ids']) <= set(result.ids.tolist())
    # Every ensemble entry is (id, step) of a snapshot evaluated during training.
    assert all(step % EPOCH_SIZE == 0 and step > 0 for step in report['steps'])
    assert report['score_val'] == pytest.approx(result.history[-1]['ensemble_val'])


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Determinism, options and input validation
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _fingerprint(result: PackTrainResult) -> tuple:
    return (
        result.ids.tolist(),
        result.best_steps.tolist(),
        [r['train_loss'] for r in result.history],
        [r['ensemble_val'] for r in result.history],
        {part: x.tobytes() for part, x in result.predictions.items()},
    )


def test_deterministic_for_a_fixed_seed() -> None:
    dataset = _dataset()

    def run(seed: int) -> PackTrainResult:
        result, _ = _train(
            dataset,
            pack_size=4,
            patience=1,
            max_epochs=10,
            seed=seed,
            online_ensemble=_make_ensemble(dataset, patience=3),
            setup_kwargs={'dropout': 0.2, 'n_blocks': [1, 2, 1, 2]},
        )
        return result

    first = run(0)
    assert _fingerprint(first) == _fingerprint(run(0))
    # The seed drives the batch order.
    assert _fingerprint(first) != _fingerprint(run(1))


def test_multiclass() -> None:
    base = _dataset()
    x = base.x_num
    assert x is not None
    # Three classes: 0, 1 or 2 of two conditions hold.
    y = {
        part: (x[part][:, 0] > 0).long() + (x[part][:, 1] > 0.5).long()
        for part in PARTS
    }
    dataset = PreparedDataset(
        x_num=base.x_num,
        x_cat=base.x_cat,
        y=y,
        task=TaskInfo(type_=TaskType.MULTICLASS, score='accuracy', n_classes=3),
        cat_cardinalities=base.cat_cardinalities,
    )
    result, _ = _train(
        dataset,
        pack_size=2,
        patience=-1,
        max_epochs=2,
        online_ensemble=_make_ensemble(dataset, patience=2),
    )
    for part in PARTS:
        predictions = result.predictions[part]
        assert predictions.shape == (2, dataset.size(part), 3)
        np.testing.assert_allclose(predictions.sum(-1), 1.0, rtol=1e-5)
    assert set(result.members[0]['metrics']['test']) == {
        'accuracy',
        'log-loss',
        'score',
    }


def test_numerical_features_only_and_autocast_and_progress() -> None:
    dataset = _dataset(cat_cardinalities=())
    result, _ = _train(
        dataset,
        pack_size=2,
        patience=-1,
        max_epochs=2,
        autocast=torch.autocast('cpu', dtype=torch.bfloat16),
        progress=True,
    )
    _check_result(result, dataset, n_finished=2)


def test_a_single_batch_per_epoch() -> None:
    dataset = _dataset()
    model, optimizer, _ = _setup(dataset, pack_size=2)
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=10 * N_TRAIN,
        patience=-1,
        max_epochs=3,
        seed=0,
    )
    assert result.n_epochs == result.n_steps == 3
    assert result.ids.tolist() == [0, 1]
    assert set(result.best_steps.tolist()) <= {1, 2, 3}


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'batch_size': 0}, 'batch_size'),
        ({'max_epochs': 0}, 'max_epochs'),
        ({'patience': -1, 'max_epochs': -1}, 'never stop'),
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    dataset = _dataset()
    model, optimizer, call_pack_sizes = _setup(dataset, pack_size=2)
    arguments: dict[str, Any] = {
        'model': model,
        'optimizer': optimizer,
        'dataset': dataset,
        'batch_size': BATCH_SIZE,
        'patience': 1,
        'max_epochs': 5,
        'seed': 0,
    }
    with pytest.raises(ValueError, match=match):
        train_pack(**(arguments | kwargs))
    assert call_pack_sizes == []


def test_on_return_the_model_holds_only_running_members() -> None:
    dataset = _dataset()
    model, optimizer, _ = _setup(dataset, pack_size=3)
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        patience=0,
        max_epochs=4,
        seed=0,
    )
    assert len(result.ids) == 3
    assert model.pack_size == 0
    for group in optimizer.param_groups:
        for p in group['params']:
            assert p.shape[0] == 0


@pytest.mark.gpu
def test_cuda_with_bf16_autocast(cuda_device: torch.device) -> None:
    dataset = _dataset().to(cuda_device)
    model, optimizer, call_pack_sizes = _setup(
        dataset, pack_size=4, n_blocks=[1, 2, 3, 1], dropout=0.1, device=cuda_device
    )
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        patience=1,
        max_epochs=30,
        seed=0,
        autocast=torch.autocast('cuda', dtype=torch.bfloat16),
        online_ensemble=_make_ensemble(dataset, patience=100),
    )
    assert result.history[-1]['n_running'] == 0
    assert min(call_pack_sizes) > 0
    cpu_dataset = _dataset()
    _check_result(result, cpu_dataset, n_finished=4)
    assert result.online_ensemble is not None
    assert set(result.online_ensemble['ids']) <= set(result.ids.tolist())
