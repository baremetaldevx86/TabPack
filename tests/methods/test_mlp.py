"""Tests for the ordinary MLP baseline (a30): methods/mlp.py.

Every run uses a tiny synthetic dataset written to disk in the official format, so
the whole path (load -> preprocess -> train -> report files) is exercised on CPU.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tabpack_repro.config import (
    DataConfig,
    MLPMethodConfig,
    MLPModelConfig,
    TrainingConfig,
    config_to_dict,
    load_config,
)
from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.methods import mlp
from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.types import TaskType

N_TRAIN, N_VAL, N_TEST = 400, 150, 150
BATCH_SIZE = 64
EPOCH_SIZE = math.ceil(N_TRAIN / BATCH_SIZE)
PATIENCE = 2

REPORT_KEYS = {
    'schema_version',
    'method',
    'dataset',
    'seed',
    'config',
    'metrics',
    'members',
    'ensemble',
    'best_member',
    'n_epochs',
    'n_steps',
    'time_sec',
    'env',
    'history',
}
HISTORY_KEYS = {'epoch', 'step', 'time', 'train_loss', 'val_score', 'test_score'}
BINCLASS_METRICS = {'accuracy', 'roc-auc', 'log-loss', 'score'}
BINCLASS_TASK = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_dataset(
    path: Path, *, task_type: str = 'binclass', n_classes: int = 2, seed: int = 0
) -> Path:
    """A learnable dataset in the official on-disk format (string categories)."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    x_num = rng.standard_normal((n, 3)).astype(np.float32)
    # A two-valued numerical column: moved to the categorical features by the
    # pipeline (extract_bin_from_num + convert-to-cat), as in Churn.
    binary = rng.integers(0, 2, n).astype(np.float32)
    x_num = np.concatenate([x_num, binary[:, None]], axis=1)
    cat_a = rng.choice(np.array(['red', 'green', 'blue']), n)
    cat_b = rng.choice(np.array(['x', 'y']), n)
    x_cat = np.stack([cat_a, cat_b], axis=1).astype(str)

    logit = 1.5 * x_num[:, 0] - x_num[:, 1] + (cat_a == 'red') - 0.5 * binary
    logit = logit + 0.3 * rng.standard_normal(n)
    if task_type == 'binclass':
        y = (logit > 0).astype(np.int64)
    else:
        edges = np.quantile(logit, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(logit, edges).astype(np.int64)

    path.mkdir(parents=True)
    (path / 'info.json').write_text(
        json.dumps({'task': {'type': task_type, 'score': 'accuracy'}})
    )
    np.save(path / 'x_num.npy', x_num)
    np.save(path / 'x_cat.npy', x_cat)
    np.save(path / 'y.npy', y)
    split_dir = path / 'splits' / 'default'
    split_dir.mkdir(parents=True)
    index = rng.permutation(n).astype(np.int64)
    np.save(split_dir / 'train.npy', index[:N_TRAIN])
    np.save(split_dir / 'val.npy', index[N_TRAIN : N_TRAIN + N_VAL])
    np.save(split_dir / 'test.npy', index[N_TRAIN + N_VAL :])
    return path


def make_config(data_path: Path, **training: object) -> MLPMethodConfig:
    return MLPMethodConfig(
        seed=0,
        data=DataConfig(path=str(data_path.resolve())),
        model=MLPModelConfig(n_blocks=1, d_block=16, dropout=0.1),
        training=TrainingConfig(
            **{
                'batch_size': BATCH_SIZE,
                'patience': PATIENCE,
                'max_epochs': 100,
                'device': 'cpu',
                **training,
            }
        ),
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def without_time(history: list[dict]) -> list[dict]:
    return [{k: v for k, v in record.items() if k != 'time'} for record in history]


@pytest.fixture(scope='module')
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_dataset(tmp_path_factory.mktemp('data') / 'toy')


@pytest.fixture(scope='module')
def finished_run(dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    config = make_config(dataset_dir)
    output_dir = tmp_path_factory.mktemp('run') / 'mlp'
    report = mlp.run(config, output_dir)
    return config, output_dir, report


# ---------------------------------------------------------------------------
# The run report and its files
# ---------------------------------------------------------------------------


def test_run_writes_the_three_files(finished_run) -> None:
    _, output_dir, report = finished_run
    assert {p.name for p in output_dir.iterdir()} == {
        'report.json',
        'predictions.npz',
        'config.toml',
    }
    on_disk = json.loads((output_dir / 'report.json').read_text())
    assert on_disk == json.loads(json.dumps(report))


def test_report_common_fields(finished_run, dataset_dir: Path) -> None:
    config, _, report = finished_run
    assert set(report) == REPORT_KEYS
    assert report['schema_version'] == RUN_REPORT_SCHEMA_VERSION
    assert report['method'] == 'mlp'
    assert report['dataset'] == dataset_dir.name
    assert report['seed'] == 0
    assert report['config'] == config_to_dict(config)
    assert report['ensemble'] is None
    assert report['env']['device'] == 'cpu'
    assert report['time_sec'] > 0


def test_report_member_and_metrics(finished_run) -> None:
    config, output_dir, report = finished_run
    (member,) = report['members']
    assert member['id'] == 0
    assert member['config'] == {
        'model': config_to_dict(config)['model'],
        'optimizer': config_to_dict(config)['optimizer'],
    }
    assert set(member['metrics']) == {'train', 'val', 'test'}
    for part_metrics in member['metrics'].values():
        assert set(part_metrics) == BINCLASS_METRICS
    # The single model is the final prediction and the best member.
    assert report['metrics'] == {
        'val': member['metrics']['val'],
        'test': member['metrics']['test'],
    }
    assert report['best_member'] == {'id': 0, 'metrics': member['metrics']}
    # The synthetic task is learnable.
    assert report['metrics']['test']['accuracy'] > 0.6
    assert report['metrics']['val']['accuracy'] > 0.6

    # Metrics are recomputable from the stored predictions.
    predictions = load_npz(output_dir / 'predictions.npz')
    assert set(predictions) == {'val', 'test'}
    labels = {'val': N_VAL, 'test': N_TEST}
    for part, size in labels.items():
        pred = predictions[part]
        assert pred.shape == (size,)
        assert pred.dtype == np.float32
        assert ((pred >= 0) & (pred <= 1)).all()
    y = np.load(Path(config.data.path) / 'y.npy')
    split = Path(config.data.path) / 'splits' / 'default'
    for part in labels:
        y_part = y[np.load(split / f'{part}.npy')]
        expected = compute_metrics(y_part, predictions[part], BINCLASS_TASK)
        assert report['metrics'][part] == pytest.approx(expected)


def test_config_toml_round_trips(finished_run) -> None:
    config, output_dir, _ = finished_run
    assert load_config(output_dir / 'config.toml') == config


def test_history_and_early_stopping(finished_run) -> None:
    _, _, report = finished_run
    history = report['history']
    n_epochs = report['n_epochs']
    assert len(history) == n_epochs >= 1
    assert report['n_steps'] == n_epochs * EPOCH_SIZE
    for epoch, record in enumerate(history, start=1):
        assert set(record) == HISTORY_KEYS
        assert record['epoch'] == epoch
        assert record['step'] == epoch * EPOCH_SIZE
        assert math.isfinite(record['train_loss']) and record['train_loss'] > 0
    times = [record['time'] for record in history]
    assert times == sorted(times)
    assert report['time_sec'] >= times[-1]

    # Replay the stopping rule on the recorded val scores: strict improvement,
    # stop when the consecutive non-improving evaluations exceed the patience.
    val_scores = [record['val_score'] for record in history]
    best, n_bad, best_epoch = -math.inf, 0, 0
    for epoch, score in enumerate(val_scores, start=1):
        assert n_bad <= PATIENCE  # the run did not continue past the stop
        if score > best:
            best, n_bad, best_epoch = score, 0, epoch
        else:
            n_bad += 1
    assert n_bad == PATIENCE + 1  # it stopped as soon as the rule said so
    assert n_epochs == best_epoch + PATIENCE + 1
    # The best epoch is the FIRST epoch with the maximal val score.
    assert best_epoch == int(np.argmax(val_scores)) + 1

    # Final metrics are those of the best epoch.
    (member,) = report['members']
    assert member['best_step'] == best_epoch * EPOCH_SIZE
    best_record = history[best_epoch - 1]
    assert report['metrics']['val']['score'] == pytest.approx(best_record['val_score'])
    assert report['metrics']['test']['score'] == pytest.approx(
        best_record['test_score']
    )


def test_max_epochs_limits_training(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config(dataset_dir, patience=-1, max_epochs=3)
    report = mlp.run(config, tmp_path / 'run')
    assert report['n_epochs'] == 3
    assert [r['epoch'] for r in report['history']] == [1, 2, 3]
    assert report['n_steps'] == 3 * EPOCH_SIZE


def test_patience_zero_stops_after_first_bad_epoch(
    dataset_dir: Path, tmp_path: Path
) -> None:
    config = make_config(dataset_dir, patience=0, max_epochs=50)
    report = mlp.run(config, tmp_path / 'run')
    scores = [r['val_score'] for r in report['history']]
    # Every epoch but the last improved strictly; the last one did not.
    assert all(b > a for a, b in itertools.pairwise(scores[:-1]))
    if len(scores) < 50:
        assert scores[-1] <= max(scores[:-1])


def test_stopping_disabled_is_rejected(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config(dataset_dir, patience=-1, max_epochs=-1)
    with pytest.raises(ValueError, match='never stop'):
        mlp.run(config, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_for_a_fixed_seed(finished_run, tmp_path: Path) -> None:
    config, output_dir, report = finished_run
    again = mlp.run(config, tmp_path / 'again')
    assert without_time(again['history']) == without_time(report['history'])
    assert again['members'] == report['members']
    assert again['metrics'] == report['metrics']
    first, second = (
        load_npz(output_dir / 'predictions.npz'),
        load_npz(tmp_path / 'again' / 'predictions.npz'),
    )
    for part in ('val', 'test'):
        np.testing.assert_array_equal(first[part], second[part])


def test_another_seed_gives_another_model(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config(dataset_dir, max_epochs=2)
    config.seed = 1
    mlp.run(config, tmp_path / 'seed1')
    config.seed = 0
    mlp.run(config, tmp_path / 'seed0')
    seed0 = load_npz(tmp_path / 'seed0' / 'predictions.npz')
    seed1 = load_npz(tmp_path / 'seed1' / 'predictions.npz')
    assert not np.array_equal(seed0['test'], seed1['test'])


# ---------------------------------------------------------------------------
# Other tasks and the autocast hook
# ---------------------------------------------------------------------------


def test_multiclass(tmp_path: Path) -> None:
    data = write_dataset(tmp_path / 'toy3', task_type='multiclass', n_classes=3)
    report = mlp.run(make_config(data, max_epochs=5), tmp_path / 'run')
    assert set(report['metrics']['test']) == {'accuracy', 'log-loss', 'score'}
    assert report['metrics']['test']['accuracy'] > 0.5
    predictions = load_npz(tmp_path / 'run' / 'predictions.npz')
    assert predictions['test'].shape == (N_TEST, 3)
    np.testing.assert_allclose(predictions['test'].sum(axis=1), 1.0, atol=1e-5)


class _CountingAutocast(contextlib.AbstractContextManager):
    def __init__(self) -> None:
        self.n_enter = 0

    def __enter__(self):
        self.n_enter += 1
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_uses_the_autocast_of_the_run_context(
    dataset_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    autocast = _CountingAutocast()
    real_setup_run = mlp.setup_run

    def setup_run(*args, **kwargs):
        ctx = real_setup_run(*args, **kwargs)
        ctx.autocast = autocast
        return ctx

    monkeypatch.setattr(mlp, 'setup_run', setup_run)
    report = mlp.run(make_config(dataset_dir, max_epochs=2), tmp_path / 'run')
    # One per training step, one per evaluation batch (val + test every epoch,
    # train once at the end; every part fits in one eval batch).
    assert autocast.n_enter == report['n_steps'] + 2 * report['n_epochs'] + 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_one_hot_maps_unknown_codes_to_zeros() -> None:
    codes = torch.tensor([[0, 1], [2, 0], [3, 5]])
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    result = mlp._one_hot(codes, [3, 2])
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_model_structure_and_optimizer_groups() -> None:
    config = MLPMethodConfig(
        model=MLPModelConfig(n_blocks=2, d_block=8, dropout=0.2, activation='GELU')
    )
    config.optimizer.weight_decay = 0.5
    model = mlp._make_model(5, 1, config.model)
    kinds = [type(m).__name__ for m in model]
    assert kinds == ['Linear', 'GELU', 'Dropout'] * 2 + ['Linear']
    assert [m.p for m in model if isinstance(m, torch.nn.Dropout)] == [0.2, 0.2]
    assert model(torch.randn(4, 5)).shape == (4, 1)

    optimizer = mlp._make_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    decay, no_decay = optimizer.param_groups
    weights = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
    biases = [m.bias for m in model if isinstance(m, torch.nn.Linear)]
    assert [id(p) for p in decay['params']] == [id(p) for p in weights]
    assert [id(p) for p in no_decay['params']] == [id(p) for p in biases]
    assert decay['weight_decay'] == 0.5
    assert no_decay['weight_decay'] == 0.0
    assert decay['lr'] == no_decay['lr'] == config.optimizer.lr


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match='activation'):
        mlp._make_model(3, 1, MLPModelConfig(activation='Tanh'))


@pytest.mark.data
def test_real_churn_smoke(churn_dir: Path, tmp_path: Path) -> None:
    config = make_config(churn_dir, max_epochs=2, batch_size=256)
    report = mlp.run(config, tmp_path / 'run')
    assert report['dataset'] == 'churn'
    assert report['n_epochs'] == 2
    assert report['metrics']['test']['accuracy'] > 0.75
