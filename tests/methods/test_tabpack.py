"""Tests for the reduced heterogeneous TabPack method (a32): methods/tabpack.py."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tabpack_repro.config import (
    DataConfig,
    OnlineEnsembleConfig,
    TabPackConfig,
    TrainingConfig,
)
from tabpack_repro.data.pipeline import build_dataset
from tabpack_repro.methods import tabpack
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.sampler import sample_configs
from tabpack_repro.utils.io import to_jsonable

N_MODELS = 4
MAX_ENSEMBLE_SIZE = 3
SPACE: dict[str, Any] = {
    'model': {
        'n_blocks': ['_tune_', 'int', 1, 2],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
    },
    'optimizer': {
        'lr': ['_tune_', 'loguniform', 0.001, 0.01],
        'weight_decay': ['_tune_', 'loguniform', 0.001, 0.1],
        'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
    },
}
SCHEMA_KEYS = {
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


# ---------------------------------------------------------------------------
# Fixtures


def _write_dataset(path: Path, *, n: int = 300, seed: int = 0) -> Path:
    """A small learnable binclass dataset in the official on-disk format."""
    rng = np.random.default_rng(seed)
    x_num = rng.standard_normal((n, 3)).astype(np.float32)
    x_cat = rng.choice(np.array(['a', 'b', 'c']), (n, 1))
    logit = 2.0 * x_num[:, 0] - x_num[:, 1] + 1.0 * (x_cat[:, 0] == 'b')
    y = (logit + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    order = rng.permutation(n)
    splits = {'train': order[:180], 'val': order[180:240], 'test': order[240:]}

    (path / 'splits' / 'default').mkdir(parents=True)
    (path / 'info.json').write_text(
        json.dumps({'task': {'type': 'binclass', 'score': 'accuracy'}})
    )
    np.save(path / 'x_num.npy', x_num)
    np.save(path / 'x_cat.npy', x_cat)
    np.save(path / 'y.npy', y)
    for part, idx in splits.items():
        np.save(path / 'splits' / 'default' / f'{part}.npy', idx.astype(np.int64))
    return path


def _config(data_dir: Path | str, **kwargs: Any) -> TabPackConfig:
    defaults: dict[str, Any] = {
        'seed': 0,
        'n_models': N_MODELS,
        'data': DataConfig(path=str(data_dir)),
        'd_block': 16,
        'training': TrainingConfig(
            batch_size=32,
            patience=2,
            max_epochs=8,
            eval_batch_size=1024,
            amp_dtype=None,
            device='cpu',
        ),
        'online_ensemble': OnlineEnsembleConfig(
            patience=2, max_ensemble_size=MAX_ENSEMBLE_SIZE
        ),
        'space': copy.deepcopy(SPACE),
    }
    defaults.update(kwargs)
    return TabPackConfig(**defaults)


@pytest.fixture(scope='module')
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp('data') / 'tiny')


def _spy(calls: dict[str, list[Any]], name: str, fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.setdefault(name, []).append((args, kwargs))
        return fn(*args, **kwargs)

    return wrapper


@pytest.fixture(scope='module')
def main_run(
    data_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], Path, dict[str, list[Any]]]:
    """One TabPack run with spies on the components (real implementations)."""
    output_dir = tmp_path_factory.mktemp('run') / 'tabpack'
    calls: dict[str, list[Any]] = {}
    with pytest.MonkeyPatch.context() as mp:
        for name in (
            'ModelPack',
            'MuonAdamWPack',
            'OnlineGreedyEnsemble',
            'train_pack',
        ):
            mp.setattr(tabpack, name, _spy(calls, name, getattr(tabpack, name)))
        report = tabpack.run(_config(data_dir), output_dir)
    return report, output_dir, calls


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# End-to-end run


def test_files_and_report_fields(main_run) -> None:
    report, output_dir, _ = main_run
    for name in ('report.json', 'predictions.npz', 'config.toml'):
        assert (output_dir / name).is_file(), name

    assert SCHEMA_KEYS <= set(report)
    assert report['schema_version'] == 1
    assert report['method'] == 'tabpack'
    assert report['dataset'] == 'tiny'
    assert report['seed'] == 0
    assert report['config']['n_models'] == N_MODELS
    assert report['n_models'] == N_MODELS
    assert report['n_finished'] == len(report['members'])
    assert 1 <= len(report['members']) <= N_MODELS
    assert report['n_epochs'] >= 1
    assert report['n_steps'] >= report['n_epochs']
    assert report['time_sec'] >= 0.0
    assert len(report['history']) == report['n_epochs']
    assert set(report['env']) >= {'device', 'git_commit'}
    for part in ('val', 'test'):
        assert 'score' in report['metrics'][part]
        assert 'accuracy' in report['metrics'][part]

    # report.json holds exactly the returned report.
    assert _load_json(output_dir / 'report.json') == to_jsonable(report)


def test_predictions_file_matches_metrics(main_run, data_dir: Path) -> None:
    report, output_dir, _ = main_run
    dataset = build_dataset(DataConfig(path=str(data_dir)))
    with np.load(output_dir / 'predictions.npz') as npz:
        assert set(npz.files) == {'val', 'test'}
        for part in ('val', 'test'):
            y_pred = npz[part]
            assert y_pred.dtype == np.float32
            assert y_pred.shape == (dataset.size(part),)
            assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
            expected = compute_metrics(dataset.y[part].numpy(), y_pred, dataset.task)
            assert report['metrics'][part] == pytest.approx(expected)


def test_member_configs_recorded_and_differ(main_run) -> None:
    report, _, _ = main_run
    member_configs = report['member_configs']
    # Sampled from the space with the run seed, in member-id order.
    assert member_configs == sample_configs(SPACE, N_MODELS, seed=0)
    assert len({json.dumps(c, sort_keys=True) for c in member_configs}) == N_MODELS
    for c in member_configs:
        assert set(c) == {'model', 'optimizer'}
        assert set(c['model']) == {'n_blocks', 'dropout'}
        assert set(c['optimizer']) == {'lr', 'weight_decay', 'muon_lr'}
        assert c['model']['n_blocks'] in (1, 2)

    ids = [m['id'] for m in report['members']]
    assert len(set(ids)) == len(ids)
    for member in report['members']:
        assert 0 <= member['id'] < N_MODELS
        assert member['config'] == member_configs[member['id']]
        assert member['best_step'] >= 1
        assert set(member['metrics']) == {'train', 'val', 'test'}


def test_hyperparameters_reach_the_components(main_run) -> None:
    report, _, calls = main_run
    configs = report['member_configs']
    config = _config('unused')

    ((_, model_kwargs),) = calls['ModelPack']
    assert model_kwargs['pack_size'] == N_MODELS
    assert model_kwargs['d_block'] == 16
    assert model_kwargs['activation'] == 'ReLU'
    assert list(model_kwargs['n_blocks']) == [c['model']['n_blocks'] for c in configs]
    assert list(model_kwargs['dropout']) == [c['model']['dropout'] for c in configs]

    ((opt_args, opt_kwargs),) = calls['MuonAdamWPack']
    for key in ('lr', 'weight_decay', 'muon_lr'):
        assert list(opt_kwargs[key]) == [c['optimizer'][key] for c in configs]
    assert opt_kwargs['pack_size'] == N_MODELS
    for key in (
        'muon_momentum',
        'muon_nesterov',
        'muon_ns_steps',
        'beta1',
        'beta2',
        'eps',
        'shared_step',
    ):
        assert opt_kwargs[key] == getattr(config.optimizer, key), key
    # make_param_groups(muon=True): one Muon group per backbone block.
    groups = opt_args[0]
    n_muon = sum(bool(g.get('muon')) for g in groups)
    assert n_muon == max(c['model']['n_blocks'] for c in configs)

    ((_, ens_kwargs),) = calls['OnlineGreedyEnsemble']
    assert ens_kwargs['max_ensemble_size'] == MAX_ENSEMBLE_SIZE
    assert ens_kwargs['patience'] == 2

    ((_, train_kwargs),) = calls['train_pack']
    assert train_kwargs['seed'] == 0
    assert train_kwargs['batch_size'] == 32
    assert train_kwargs['patience'] == 2
    assert train_kwargs['max_epochs'] == 8
    assert train_kwargs['eval_batch_size'] == 1024
    assert train_kwargs['online_ensemble'] is not None


def test_ensemble_fields(main_run) -> None:
    report, _, _ = main_run
    ensemble = report['ensemble']
    assert set(ensemble) == {'ids', 'steps', 'size', 'n_unique'}
    assert 1 <= ensemble['size'] <= MAX_ENSEMBLE_SIZE
    assert ensemble['size'] == len(ensemble['ids']) == len(ensemble['steps'])
    assert ensemble['n_unique'] == len(set(ensemble['ids']))
    assert set(ensemble['ids']) <= set(range(N_MODELS))
    assert all(step >= 1 for step in ensemble['steps'])
    assert ensemble['steps'] and max(ensemble['steps']) <= report['n_steps']

    online = report['online_ensemble']
    assert online['ids'] == ensemble['ids']
    assert online['steps'] == ensemble['steps']
    assert online['size'] == ensemble['size']
    assert online['n_unique'] == ensemble['n_unique']
    assert online['score_val'] == pytest.approx(report['metrics']['val']['score'])


def test_ensemble_is_at_least_as_good_as_best_member_on_val(main_run) -> None:
    report, _, _ = main_run
    best = report['best_member']
    assert best is not None
    val_scores = {m['id']: m['metrics']['val']['score'] for m in report['members']}
    assert best['metrics']['val']['score'] == max(val_scores.values())
    assert best['config'] == report['member_configs'][best['id']]
    # The greedy ensemble starts from the best single snapshot of its pool, and the
    # best member's best snapshot was in the pool at its best epoch.
    assert report['metrics']['val']['score'] >= best['metrics']['val']['score'] - 1e-12


def test_deterministic(main_run, data_dir: Path, tmp_path: Path) -> None:
    report, output_dir, _ = main_run
    again = tabpack.run(_config(data_dir), tmp_path / 'again')

    for key in ('metrics', 'members', 'ensemble', 'best_member', 'member_configs'):
        assert again[key] == report[key], key
    assert again['n_epochs'] == report['n_epochs']
    assert again['n_steps'] == report['n_steps']

    def strip_time(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in h.items() if k != 'time'} for h in history]

    assert strip_time(again['history']) == strip_time(report['history'])
    with (
        np.load(output_dir / 'predictions.npz') as first,
        np.load(tmp_path / 'again' / 'predictions.npz') as second,
    ):
        for part in ('val', 'test'):
            np.testing.assert_array_equal(first[part], second[part])


# ---------------------------------------------------------------------------
# Explicit member configs


def test_explicit_configs(data_dir: Path, tmp_path: Path) -> None:
    configs = [
        {
            'model': {'n_blocks': 2, 'dropout': 0.1},
            'optimizer': {'lr': 0.003, 'weight_decay': 0.01, 'muon_lr': 0.02},
        },
        {
            'model': {'n_blocks': 1, 'dropout': 0.0},
            'optimizer': {'lr': 0.001, 'weight_decay': 0.1, 'muon_lr': 0.005},
        },
    ]
    config = _config(data_dir, n_models=2, configs=copy.deepcopy(configs))
    report = tabpack.run(config, tmp_path / 'explicit')

    assert report['member_configs'] == configs
    assert report['n_models'] == 2
    assert report['config']['configs'] == configs
    for member in report['members']:
        assert member['config'] == configs[member['id']]
    assert set(report['ensemble']['ids']) <= {0, 1}
    # The input config is not modified.
    assert config.configs == configs
    assert (tmp_path / 'explicit' / 'report.json').is_file()


def test_explicit_configs_n_models_mismatch_raises(tmp_path: Path) -> None:
    configs = [
        {
            'model': {'n_blocks': 1, 'dropout': 0.0},
            'optimizer': {'lr': 0.001, 'weight_decay': 0.01, 'muon_lr': 0.01},
        }
    ] * 2
    # The dataset does not exist: the error must come before any data is loaded.
    config = _config(tmp_path / 'missing', n_models=3, configs=configs)
    with pytest.raises(ValueError, match='n_models=3'):
        tabpack.run(config, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize(
    ('mutate', 'match'),
    [
        (lambda c: c['model'].update(d_block=8), r'not used: model\.d_block'),
        (lambda c: c.update(extra={'x': 1}), 'not used: extra'),
        (lambda c: c['optimizer'].pop('muon_lr'), r'optimizer\.muon_lr'),
        (lambda c: c.pop('model'), r'model\.n_blocks'),
    ],
)
def test_explicit_configs_with_unused_or_missing_keys_raise(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    configs = [
        {
            'model': {'n_blocks': 1, 'dropout': 0.0},
            'optimizer': {'lr': 0.001, 'weight_decay': 0.01, 'muon_lr': 0.01},
        }
        for _ in range(2)
    ]
    for c in configs:
        mutate(c)
    config = _config(tmp_path / 'missing', n_models=2, configs=configs)
    with pytest.raises(ValueError, match=match):
        tabpack.run(config, tmp_path / 'out')


def test_space_with_unused_section_raises(tmp_path: Path) -> None:
    space = copy.deepcopy(SPACE)
    space['training'] = {'batch_size': ['_tune_', 'int', 16, 64]}
    config = _config(tmp_path / 'missing', space=space)
    with pytest.raises(ValueError, match='not used: training'):
        tabpack.run(config, tmp_path / 'out')


@pytest.mark.parametrize(
    'online_ensemble',
    [
        OnlineEnsembleConfig(include_current_ensemble_in_pool=False),
        OnlineEnsembleConfig(update_type='best'),  # type: ignore[arg-type]
        OnlineEnsembleConfig(type='uniform'),  # type: ignore[arg-type]
    ],
)
def test_unsupported_online_ensemble_raises(
    tmp_path: Path, online_ensemble: OnlineEnsembleConfig
) -> None:
    config = _config(tmp_path / 'missing', online_ensemble=online_ensemble)
    with pytest.raises(ValueError, match='online_ensemble'):
        tabpack.run(config, tmp_path / 'out')


def test_n_models_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='n_models'):
        tabpack.run(_config(tmp_path / 'missing', n_models=0), tmp_path / 'out')


# ---------------------------------------------------------------------------
# Config transposition


def test_split_member_configs_transposes() -> None:
    configs = sample_configs(SPACE, 3, seed=1)
    split = tabpack._split_member_configs(configs)
    assert split == {
        'model': {
            key: [c['model'][key] for c in configs] for key in ('n_blocks', 'dropout')
        },
        'optimizer': {
            key: [c['optimizer'][key] for c in configs]
            for key in ('lr', 'weight_decay', 'muon_lr')
        },
    }


def test_split_member_configs_requires_same_keys() -> None:
    configs = sample_configs(SPACE, 2, seed=0)
    configs[1]['model']['activation'] = 'ReLU'
    with pytest.raises(ValueError, match='same keys'):
        tabpack._split_member_configs(configs)


def test_split_member_configs_rejects_non_dicts() -> None:
    configs = sample_configs(SPACE, 2, seed=0)
    configs[1]['optimizer'] = [0.1]
    with pytest.raises(TypeError, match='must be a dict'):
        tabpack._split_member_configs(configs)


def test_sampled_configs_depend_on_the_seed() -> None:
    config = TabPackConfig(n_models=3, space=copy.deepcopy(SPACE))
    first = tabpack._member_configs(config)
    assert first == tabpack._member_configs(dataclasses.replace(config))
    assert first != tabpack._member_configs(dataclasses.replace(config, seed=1))
