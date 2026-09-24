"""Tests for tabpack_repro.methods.homogeneous (a31).

End-to-end on CPU with a tiny on-disk dataset in the official format, so that the
whole chain (setup_run -> ModelPack -> AdamWPack -> train_pack -> averaging ->
write_run) runs for real.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tabpack_repro.config import (
    AdamWConfig,
    DataConfig,
    HomogeneousEnsembleConfig,
    MLPModelConfig,
    TrainingConfig,
)
from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.methods import homogeneous
from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.types import PARTS, TaskType

N_MODELS = 3

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_dataset(
    path: Path,
    *,
    n_classes: int = 2,
    sizes: tuple[int, int, int] = (320, 160, 160),
    seed: int = 0,
) -> Path:
    """A small learnable classification dataset in the official on-disk format."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    x_num = rng.standard_normal((n, 4)).astype(np.float32)
    categories = np.array(['red', 'green', 'blue'])
    cat_codes = rng.integers(0, len(categories), n)
    x_cat = categories[cat_codes][:, None]  # (N, 1) fixed-width unicode strings
    logit = 1.5 * x_num[:, 0] - x_num[:, 1] + (cat_codes == 1) + 0.3 * x_num[:, 2]
    noisy = logit + 0.5 * rng.standard_normal(n)
    if n_classes == 2:
        y = (noisy > 0).astype(np.int64)
        task = {'type': 'binclass', 'score': 'accuracy'}
    else:
        y = np.digitize(noisy, np.quantile(noisy, [1 / 3, 2 / 3])).astype(np.int64)
        task = {'type': 'multiclass', 'score': 'accuracy'}

    path.mkdir(parents=True)
    (path / 'info.json').write_text(json.dumps({'task': task}))
    np.save(path / 'x_num.npy', x_num)
    np.save(path / 'x_cat.npy', x_cat)
    np.save(path / 'y.npy', y)
    split_dir = path / 'splits' / 'default'
    split_dir.mkdir(parents=True)
    perm = rng.permutation(n)
    bounds = np.cumsum((0, *sizes))
    for part, lo, hi in zip(PARTS, bounds[:-1], bounds[1:], strict=True):
        np.save(split_dir / f'{part}.npy', np.sort(perm[lo:hi]).astype(np.int64))
    return path


def _config(data_path: Path, **overrides: Any) -> HomogeneousEnsembleConfig:
    kwargs: dict[str, Any] = {
        'seed': 0,
        'n_models': N_MODELS,
        'data': DataConfig(path=str(data_path)),
        'model': MLPModelConfig(n_blocks=1, d_block=16, dropout=0.1),
        'optimizer': AdamWConfig(lr=3e-3, weight_decay=1e-4),
        'training': TrainingConfig(
            batch_size=64,
            patience=2,
            max_epochs=60,
            eval_batch_size=128,
            amp_dtype=None,
            device='cpu',
        ),
    }
    kwargs.update(overrides)
    return HomogeneousEnsembleConfig(**kwargs)


class _TrainPackSpy:
    """Wraps homogeneous.train_pack: records the optimizer/model setup it receives
    and the PackTrainResult it returns."""

    def __init__(self, train_pack):
        self._train_pack = train_pack
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        model = kwargs['model']
        call = {
            'kwargs': dict(kwargs),
            'pack_size': model.pack_size,
            'param_groups': [
                {k: v for k, v in group.items() if k != 'params'}
                | {'n_params': len(group['params'])}
                for group in kwargs['optimizer'].param_groups
            ],
            'n_model_params': len(list(model.parameters())),
            'n_blocks': model.backbone.n_blocks.tolist(),
            'dropout': [block.dropout.p.tolist() for block in model.backbone.blocks],
            'd_block': model.head.in_features,
            'optimizer_type': type(kwargs['optimizer']),
        }
        call['result'] = self._train_pack(**kwargs)
        self.calls.append(call)
        return call['result']


@pytest.fixture
def spy(monkeypatch) -> _TrainPackSpy:
    spy = _TrainPackSpy(homogeneous.train_pack)
    monkeypatch.setattr(homogeneous, 'train_pack', spy)
    return spy


@pytest.fixture(scope='module')
def binclass_dir(tmp_path_factory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp('data') / 'toy-binclass')


@pytest.fixture(scope='module')
def binclass_run(binclass_dir, tmp_path_factory) -> dict[str, Any]:
    """One real run, shared by the read-only report checks."""
    output_dir = tmp_path_factory.mktemp('run') / 'homogeneous'
    config = _config(binclass_dir)
    spy = _TrainPackSpy(homogeneous.train_pack)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(homogeneous, 'train_pack', spy)
        report = homogeneous.run(config, output_dir)
    return {
        'report': report,
        'output_dir': output_dir,
        'config': config,
        'call': spy.calls[0],
    }


def _strip_volatile(report: dict[str, Any]) -> dict[str, Any]:
    """The report without wall-clock fields."""
    report = copy.deepcopy(report)
    report.pop('time_sec')
    for row in report['history']:
        row.pop('time')
    return report


def _y(dataset_dir: Path, part: str) -> np.ndarray:
    idx = np.load(dataset_dir / 'splits' / 'default' / f'{part}.npy')
    return np.load(dataset_dir / 'y.npy')[idx]


# ---------------------------------------------------------------------------
# report and files
# ---------------------------------------------------------------------------


def test_report_top_level_fields(binclass_run, binclass_dir):
    report = binclass_run['report']
    assert set(report) == {
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
    assert report['schema_version'] == RUN_REPORT_SCHEMA_VERSION
    assert report['method'] == 'homogeneous'
    assert report['dataset'] == binclass_dir.name
    assert report['seed'] == 0
    assert report['config']['method'] == 'homogeneous'
    assert report['config']['n_models'] == N_MODELS
    assert report['env']['device'] == 'cpu'
    assert set(report['metrics']) == {'val', 'test'}
    for part in ('val', 'test'):
        assert {'accuracy', 'roc-auc', 'log-loss', 'score'} <= set(
            report['metrics'][part]
        )
        assert report['metrics'][part]['score'] == report['metrics'][part]['accuracy']
    assert report['time_sec'] > 0


def test_members_ensemble_and_best_member(binclass_run):
    report = binclass_run['report']
    members = report['members']
    assert [m['id'] for m in members] == list(range(N_MODELS))
    for member in members:
        assert set(member) == {'id', 'best_step', 'config', 'metrics'}
        assert member['config'] is None
        assert set(member['metrics']) == set(PARTS)
        assert 0 < member['best_step'] <= report['n_steps']
        assert 0.0 <= member['metrics']['val']['accuracy'] <= 1.0

    assert report['ensemble'] == {
        'ids': list(range(N_MODELS)),
        'steps': [m['best_step'] for m in members],
        'size': N_MODELS,
        'n_unique': N_MODELS,
    }

    # max() returns the first of equal maxima, like best_member (first on ties).
    best = max(members, key=lambda m: m['metrics']['val']['score'])
    assert report['best_member'] == {'id': best['id'], 'metrics': best['metrics']}


def test_history_and_counters(binclass_run):
    report = binclass_run['report']
    history = report['history']
    assert len(history) == report['n_epochs'] >= 1
    assert [row['epoch'] for row in history] == list(range(1, report['n_epochs'] + 1))
    assert history[-1]['step'] == report['n_steps']
    assert history[-1]['n_running'] == 0
    assert history[-1]['n_finished'] == N_MODELS
    # No online ensemble in this method.
    assert all(row['ensemble_val'] is None for row in history)
    # The last member stops `patience + 1` epochs after its best epoch (unless the
    # epoch limit stopped it first).
    training = binclass_run['config'].training
    if report['n_epochs'] < training.max_epochs:
        steps_per_epoch = report['n_steps'] // report['n_epochs']
        last_best_step = max(m['best_step'] for m in report['members'])
        assert report['n_steps'] % report['n_epochs'] == 0
        assert last_best_step % steps_per_epoch == 0
        last_best_epoch = last_best_step // steps_per_epoch
        assert report['n_epochs'] == last_best_epoch + training.patience + 1


def test_files_written(binclass_run, binclass_dir):
    output_dir: Path = binclass_run['output_dir']
    report = binclass_run['report']
    assert (output_dir / 'config.toml').is_file()
    on_disk = json.loads((output_dir / 'report.json').read_text())
    assert on_disk == json.loads(json.dumps(report))

    with np.load(output_dir / 'predictions.npz') as npz:
        assert set(npz.files) == {'val', 'test'}
        for part in ('val', 'test'):
            y_pred = npz[part]
            y_true = _y(binclass_dir, part)
            assert y_pred.dtype == np.float32
            assert y_pred.shape == y_true.shape
            assert np.all((y_pred >= 0) & (y_pred <= 1))
            task = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)
            assert compute_metrics(y_true, y_pred, task) == report['metrics'][part]


# ---------------------------------------------------------------------------
# the method itself
# ---------------------------------------------------------------------------


def test_setup_passed_to_train_pack(binclass_run):
    call = binclass_run['call']
    config = binclass_run['config']
    kwargs = call['kwargs']
    assert call['pack_size'] == N_MODELS
    assert call['optimizer_type'] is homogeneous.AdamWPack
    assert kwargs['online_ensemble'] is None
    assert kwargs['batch_size'] == config.training.batch_size
    assert kwargs['patience'] == config.training.patience
    assert kwargs['max_epochs'] == config.training.max_epochs
    assert kwargs['eval_batch_size'] == config.training.eval_batch_size
    assert kwargs['seed'] == config.seed

    # Identical architecture for every member.
    assert call['n_blocks'] == [config.model.n_blocks] * N_MODELS
    assert (
        call['dropout']
        == [[pytest.approx(config.model.dropout)] * N_MODELS] * config.model.n_blocks
    )
    assert call['d_block'] == config.model.d_block

    # Scalar (shared) hyperparameters; every parameter in exactly one group.
    groups = call['param_groups']
    assert sum(g['n_params'] for g in groups) == call['n_model_params']
    assert not any(g.get('muon', False) for g in groups)
    for group in groups:
        assert group['lr'] == pytest.approx(config.optimizer.lr)
        assert isinstance(group['lr'], float)
        assert isinstance(group['weight_decay'], float)
    assert {g['weight_decay'] for g in groups} <= {0.0, config.optimizer.weight_decay}
    assert config.optimizer.weight_decay in {g['weight_decay'] for g in groups}


def test_final_prediction_is_uniform_average_of_all_members(binclass_run):
    result = binclass_run['call']['result']
    order = np.argsort(result.ids)
    with np.load(binclass_run['output_dir'] / 'predictions.npz') as npz:
        for part in ('val', 'test'):
            members = result.predictions[part][order]
            assert members.shape[0] == N_MODELS
            np.testing.assert_allclose(npz[part], members.mean(0), rtol=0, atol=1e-6)


def test_members_have_different_predictions(binclass_run):
    result = binclass_run['call']['result']
    predictions = result.predictions['test']
    for i, j in itertools.combinations(range(N_MODELS), 2):
        assert not np.allclose(predictions[i], predictions[j], atol=1e-4)


def test_ensemble_at_least_as_good_as_mean_member(binclass_run):
    report = binclass_run['report']
    tolerance = 0.025  # 4 val / test rows out of 160
    for part in ('val', 'test'):
        member_metrics = [m['metrics'][part] for m in report['members']]
        ensemble = report['metrics'][part]
        mean_accuracy = np.mean([m['accuracy'] for m in member_metrics])
        assert ensemble['accuracy'] >= mean_accuracy - tolerance
        # Exact: the log-loss is convex in the probabilities (Jensen), so the
        # log-loss of the averaged prediction never exceeds the members' mean.
        mean_log_loss = np.mean([m['log-loss'] for m in member_metrics])
        assert ensemble['log-loss'] <= mean_log_loss + 1e-6
    # The task is learnable: the ensemble is clearly better than chance.
    assert report['metrics']['test']['accuracy'] > 0.75


def test_deterministic_for_fixed_seed(binclass_run, binclass_dir, tmp_path):
    report = homogeneous.run(binclass_run['config'], tmp_path / 'again')
    assert _strip_volatile(report) == _strip_volatile(binclass_run['report'])
    with (
        np.load(binclass_run['output_dir'] / 'predictions.npz') as first,
        np.load(tmp_path / 'again' / 'predictions.npz') as second,
    ):
        for part in ('val', 'test'):
            np.testing.assert_array_equal(first[part], second[part])


def test_different_seed_changes_the_run(binclass_run, binclass_dir, tmp_path):
    report = homogeneous.run(_config(binclass_dir, seed=1), tmp_path / 'seed-1')
    assert report['seed'] == 1
    with (
        np.load(binclass_run['output_dir'] / 'predictions.npz') as first,
        np.load(tmp_path / 'seed-1' / 'predictions.npz') as second,
    ):
        assert not np.allclose(first['test'], second['test'], atol=1e-4)


def test_multiclass(tmp_path, spy):
    data_dir = _write_dataset(tmp_path / 'toy-multiclass', n_classes=3)
    report = homogeneous.run(_config(data_dir, n_models=2), tmp_path / 'run')
    assert report['ensemble']['size'] == 2
    assert [m['id'] for m in report['members']] == [0, 1]
    result = spy.calls[0]['result']
    order = np.argsort(result.ids)
    with np.load(tmp_path / 'run' / 'predictions.npz') as npz:
        for part in ('val', 'test'):
            assert npz[part].shape == (len(_y(data_dir, part)), 3)
            np.testing.assert_allclose(npz[part].sum(1), 1.0, atol=1e-5)
            np.testing.assert_allclose(
                npz[part], result.predictions[part][order].mean(0), atol=1e-6
            )
    assert {'accuracy', 'log-loss', 'score'} == set(report['metrics']['test'])


def test_single_member(tmp_path, binclass_dir):
    report = homogeneous.run(_config(binclass_dir, n_models=1), tmp_path / 'run')
    assert report['ensemble'] == {
        'ids': [0],
        'steps': [report['members'][0]['best_step']],
        'size': 1,
        'n_unique': 1,
    }
    # With one member, the "ensemble" is that member.
    for part in ('val', 'test'):
        assert report['metrics'][part] == report['members'][0]['metrics'][part]


@pytest.mark.parametrize('n_models', [0, -1, 2.0, True])
def test_bad_n_models(tmp_path, binclass_dir, n_models):
    with pytest.raises(ValueError, match='n_models'):
        homogeneous.run(_config(binclass_dir, n_models=n_models), tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
