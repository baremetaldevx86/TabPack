"""Tests for tabpack_repro.methods.common (a55)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from _helpers import make_synthetic_dataset

from tabpack_repro import config as config_module
from tabpack_repro.config import (
    ConservativeEvalConfig,
    DataConfig,
    HomogeneousEnsembleConfig,
    MLPMethodConfig,
    TabPackConfig,
    TrainingConfig,
)
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.methods import common
from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.metrics import score_pack
from tabpack_repro.types import PARTS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _RecordingDataset(PreparedDataset):
    """A PreparedDataset whose .to() records the device and returns itself."""

    moved_to: list[torch.device | str]

    def to(self, device):
        self.moved_to.append(device)
        return self


def _recording_dataset(**kwargs) -> _RecordingDataset:
    base = make_synthetic_dataset(**kwargs)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    dataset = _RecordingDataset(**fields)
    dataset.moved_to = []
    return dataset


@pytest.fixture
def fake_deps(monkeypatch):
    """Replace the (possibly not yet implemented) dependencies of setup_run and
    write_run with simple fakes that record their calls in order."""
    calls: list[tuple] = []
    dataset = _recording_dataset(n_train=40, n_val=16, n_test=24)
    sentinel_autocast = contextlib.nullcontext()

    def seed_everything(seed):
        calls.append(('seed_everything', seed))

    def resolve_device(spec):
        calls.append(('resolve_device', spec))
        return torch.device('cpu')

    def build_dataset(data):
        calls.append(('build_dataset', data))
        return dataset

    def make_autocast(amp_dtype, device):
        calls.append(('make_autocast', amp_dtype, device))
        return sentinel_autocast

    def describe_device(device):
        calls.append(('describe_device', device))
        return {'device': str(device), 'gpu': None, 'torch': '9.9', 'cuda': None}

    def git_commit():
        calls.append(('git_commit',))
        return 'abc123'

    def dump_json(path, obj):
        calls.append(('dump_json', Path(path)))
        Path(path).write_text(json.dumps(obj, indent=2))

    def dump_config(config, path):
        calls.append(('dump_config', Path(path)))
        Path(path).write_text(f'method = "{config.method}"\n')

    for name, fn in [
        ('seed_everything', seed_everything),
        ('resolve_device', resolve_device),
        ('build_dataset', build_dataset),
        ('make_autocast', make_autocast),
        ('describe_device', describe_device),
        ('git_commit', git_commit),
        ('dump_json', dump_json),
        ('dump_config', dump_config),
    ]:
        monkeypatch.setattr(common, name, fn)
    return {'calls': calls, 'dataset': dataset, 'autocast': sentinel_autocast}


def _member(id_: int, val_score: float, **extra) -> dict:
    return {
        'id': id_,
        'best_step': 10 * id_,
        'config': None,
        'metrics': {
            'train': {'score': 1.0},
            'val': {'score': val_score, 'accuracy': val_score},
            'test': {'score': 0.5},
        },
        **extra,
    }


def _predictions(n_val=16, n_test=24) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        'val': rng.random(n_val).astype(np.float64),
        'test': rng.random(n_test).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# setup_run
# ---------------------------------------------------------------------------


def test_setup_run_calls_dependencies_in_order(fake_deps):
    data = DataConfig(path='churn')
    training = TrainingConfig(device='cpu', amp_dtype=None)
    ctx = common.setup_run(7, data, training)

    names = [c[0] for c in fake_deps['calls']]
    assert names[:4] == [
        'seed_everything',
        'resolve_device',
        'build_dataset',
        'make_autocast',
    ]
    assert fake_deps['calls'][0] == ('seed_everything', 7)
    assert fake_deps['calls'][1] == ('resolve_device', 'cpu')
    assert fake_deps['calls'][2] == ('build_dataset', data)
    assert fake_deps['calls'][3] == ('make_autocast', None, torch.device('cpu'))
    assert set(names[4:]) == {'describe_device', 'git_commit'}

    assert ctx.device == torch.device('cpu')
    assert ctx.autocast is fake_deps['autocast']
    assert ctx.dataset is fake_deps['dataset']
    assert fake_deps['dataset'].moved_to == [torch.device('cpu')]


def test_setup_run_env_merges_device_info_and_commit(fake_deps):
    ctx = common.setup_run(0, DataConfig(), TrainingConfig(device='cpu'))
    assert ctx.env == {
        'device': 'cpu',
        'gpu': None,
        'torch': '9.9',
        'cuda': None,
        'git_commit': 'abc123',
    }


def test_setup_run_y_true_is_an_owned_numpy_copy(fake_deps):
    ctx = common.setup_run(0, DataConfig(), TrainingConfig(device='cpu'))
    dataset = fake_deps['dataset']
    assert set(ctx.y_true) == set(PARTS)
    for part in PARTS:
        y = ctx.y_true[part]
        assert isinstance(y, np.ndarray)
        np.testing.assert_array_equal(y, dataset.y[part].numpy())
        assert y.shape == (dataset.size(part),)
        # Mutating the report labels must not corrupt the training labels.
        assert not np.shares_memory(y, dataset.y[part].numpy())


def test_setup_run_score_fns_score_every_part(fake_deps):
    ctx = common.setup_run(0, DataConfig(), TrainingConfig(device='cpu'))
    dataset = fake_deps['dataset']
    assert set(ctx.score_fns) == set(PARTS)
    for part in PARTS:
        y = dataset.y[part]
        n = dataset.size(part)
        perfect = y.to(torch.float32)
        predictions = torch.stack([perfect, 1.0 - perfect, torch.full((n,), 0.5)])
        scores = ctx.score_fns[part](predictions)
        assert scores.shape == (3,)
        assert scores.dtype == torch.float32
        torch.testing.assert_close(scores, score_pack(y, predictions, dataset.task))
        assert scores[0].item() == 1.0
        assert scores[1].item() == 0.0


# ---------------------------------------------------------------------------
# base_report
# ---------------------------------------------------------------------------


def _fake_ctx(env=None) -> common.RunContext:
    dataset = make_synthetic_dataset(n_train=8, n_val=4, n_test=4)
    return common.RunContext(
        device=torch.device('cpu'),
        autocast=None,
        dataset=dataset,
        score_fns={},
        y_true={},
        env=env if env is not None else {'device': 'cpu', 'git_commit': None},
    )


@pytest.mark.parametrize(
    'config',
    [MLPMethodConfig(seed=3), HomogeneousEnsembleConfig(seed=3), TabPackConfig(seed=3)],
    ids=['mlp', 'homogeneous', 'tabpack'],
)
def test_base_report_common_fields(config):
    ctx = _fake_ctx()
    report = common.base_report(config.method, config, 3, ctx)
    assert list(report) == [
        'schema_version',
        'method',
        'dataset',
        'seed',
        'config',
        'env',
    ]
    assert report['schema_version'] == RUN_REPORT_SCHEMA_VERSION == 1
    assert report['method'] == config.method
    assert report['dataset'] == 'churn'
    assert report['seed'] == 3
    assert type(report['seed']) is int
    assert report['config'] == config_module.config_to_dict(config)
    assert report['env'] == ctx.env
    # The report owns its env: later edits of the report do not touch the context.
    report['env']['extra'] = 1
    assert 'extra' not in ctx.env


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('churn', 'churn'),
        ('churn/', 'churn'),
        ('/some/abs/data/churn', 'churn'),
        ('nested/dir/adult', 'adult'),
    ],
)
def test_base_report_dataset_is_basename_of_data_path(path, expected):
    config = MLPMethodConfig(data=DataConfig(path=path))
    report = common.base_report('mlp', config, 0, _fake_ctx())
    assert report['dataset'] == expected


def test_base_report_numpy_seed_becomes_int():
    report = common.base_report('mlp', MLPMethodConfig(), np.int64(5), _fake_ctx())
    assert report['seed'] == 5
    assert type(report['seed']) is int


def test_base_report_config_without_data_section():
    config = ConservativeEvalConfig(source_run='runs/x')
    report = common.base_report('tabpack-conservative', config, 0, _fake_ctx())
    assert report['dataset'] is None
    assert report['config']['source_run'] == 'runs/x'


def test_base_report_is_json_serializable():
    config = TabPackConfig()
    report = common.base_report('tabpack', config, 0, _fake_ctx())
    assert json.loads(json.dumps(report)) == report


# ---------------------------------------------------------------------------
# best_member
# ---------------------------------------------------------------------------


def test_best_member_empty_is_none():
    assert common.best_member([]) is None


def test_best_member_picks_highest_val_score():
    members = [_member(0, 0.80), _member(4, 0.86), _member(2, 0.85)]
    best = common.best_member(members)
    assert best == {'id': 4, 'metrics': members[1]['metrics']}
    assert list(best) == ['id', 'metrics']


def test_best_member_first_on_ties():
    members = [_member(5, 0.8), _member(1, 0.9), _member(3, 0.9), _member(0, 0.9)]
    assert common.best_member(members)['id'] == 1


def test_best_member_negative_scores():
    # Error metrics are negated (score = -rmse): the least negative wins.
    members = [_member(0, -3.0), _member(1, -0.5), _member(2, -math.inf)]
    assert common.best_member(members)['id'] == 1


def test_best_member_ignores_nan_scores():
    members = [_member(0, math.nan), _member(1, 0.1), _member(2, math.nan)]
    assert common.best_member(members)['id'] == 1
    assert common.best_member([_member(0, math.nan)]) is None


def test_best_member_accepts_numpy_and_tensor_scores():
    members = [
        _member(np.int64(0), np.float32(0.5)),
        _member(np.int64(1), torch.tensor(0.7)),
    ]
    best = common.best_member(members)
    assert best['id'] == 1
    assert type(best['id']) is int


def test_best_member_returns_a_copy():
    members = [_member(0, 0.5)]
    best = common.best_member(members)
    best['metrics']['val']['score'] = -1.0
    assert members[0]['metrics']['val']['score'] == 0.5


def test_best_member_missing_val_score_raises():
    member = _member(0, 0.5)
    del member['metrics']['val']['score']
    with pytest.raises(ValueError, match='val'):
        common.best_member([member])


# ---------------------------------------------------------------------------
# write_run
# ---------------------------------------------------------------------------


def test_write_run_creates_nested_dir_and_all_files(fake_deps, tmp_path):
    output_dir = tmp_path / 'a' / 'b' / 'run'
    config = MLPMethodConfig()
    report = {'method': 'mlp', 'metrics': {'val': {'score': 0.5}}}
    predictions = _predictions()
    common.write_run(output_dir, report, predictions, config)

    assert sorted(p.name for p in output_dir.iterdir()) == [
        'config.toml',
        'predictions.npz',
        'report.json',
    ]
    assert json.loads((output_dir / 'report.json').read_text()) == report
    assert (output_dir / 'config.toml').read_text() == 'method = "mlp"\n'
    with np.load(output_dir / 'predictions.npz') as npz:
        assert sorted(npz.files) == ['test', 'val']
        for part in ('val', 'test'):
            assert npz[part].dtype == np.float32
            np.testing.assert_array_equal(
                npz[part], predictions[part].astype(np.float32)
            )
    # The dependencies received the documented file names.
    written = {c[0]: c[1] for c in fake_deps['calls'] if c[0].startswith('dump_')}
    assert written == {
        'dump_json': output_dir / 'report.json',
        'dump_config': output_dir / 'config.toml',
    }


def test_write_run_accepts_str_path_and_overwrites(fake_deps, tmp_path):
    output_dir = tmp_path / 'run'
    common.write_run(str(output_dir), {'x': 1}, _predictions(), MLPMethodConfig())
    new = {'val': np.zeros(3), 'test': np.ones((2, 4))}
    common.write_run(str(output_dir), {'x': 2}, new, MLPMethodConfig())
    assert json.loads((output_dir / 'report.json').read_text()) == {'x': 2}
    with np.load(output_dir / 'predictions.npz') as npz:
        np.testing.assert_array_equal(npz['val'], np.zeros(3, np.float32))
        # Multiclass-shaped (N, C) predictions are stored as they are.
        np.testing.assert_array_equal(npz['test'], np.ones((2, 4), np.float32))
    # No temporary files are left behind.
    assert sorted(p.name for p in output_dir.iterdir()) == [
        'config.toml',
        'predictions.npz',
        'report.json',
    ]


def test_write_run_accepts_tensors(fake_deps, tmp_path):
    predictions = {
        'val': torch.tensor([0.25, 0.5], dtype=torch.bfloat16),
        'test': torch.tensor([0.125, 1.0], dtype=torch.float64, requires_grad=True),
    }
    common.write_run(tmp_path, {}, predictions, MLPMethodConfig())
    with np.load(tmp_path / 'predictions.npz') as npz:
        assert npz['val'].dtype == npz['test'].dtype == np.float32
        np.testing.assert_array_equal(npz['val'], [0.25, 0.5])
        np.testing.assert_array_equal(npz['test'], [0.125, 1.0])


@pytest.mark.parametrize(
    'keys',
    [('val',), ('test',), ('val', 'test', 'train'), ('valid', 'test'), ()],
)
def test_write_run_rejects_wrong_prediction_keys(fake_deps, tmp_path, keys):
    predictions = {k: np.zeros(2) for k in keys}
    output_dir = tmp_path / 'run'
    with pytest.raises(ValueError, match='val'):
        common.write_run(output_dir, {}, predictions, MLPMethodConfig())
    # Validation happens before anything is written.
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# real dependencies (a07 pipeline, a28 config, a29 utils) and real Churn data
# ---------------------------------------------------------------------------

_CHURN_SIZES = {'train': 6400, 'val': 1600, 'test': 2000}


@pytest.fixture
def churn_ctx(churn_dir) -> common.RunContext:
    training = TrainingConfig(device='cpu', amp_dtype=None)
    return common.setup_run(0, DataConfig(path=str(churn_dir)), training)


@pytest.mark.data
def test_setup_run_real_churn_dataset(churn_ctx):
    ctx = churn_ctx
    assert ctx.device == torch.device('cpu')
    assert ctx.autocast is None
    dataset = ctx.dataset
    assert {part: dataset.size(part) for part in PARTS} == _CHURN_SIZES
    assert dataset.n_num_features == 7
    assert dataset.cat_cardinalities == [3, 2, 2, 2]
    assert dataset.task.type_ == 'binclass'
    assert dataset.task.score == 'accuracy'
    for part in PARTS:
        assert dataset.y[part].device.type == 'cpu'
        assert dataset.x_num[part].dtype == torch.float32
        assert dataset.x_cat[part].dtype == torch.int64
        y = ctx.y_true[part]
        assert isinstance(y, np.ndarray)
        assert y.shape == (_CHURN_SIZES[part],)
        assert set(np.unique(y)) == {0, 1}
        np.testing.assert_array_equal(y, dataset.y[part].numpy())


@pytest.mark.data
def test_setup_run_real_churn_score_fns(churn_ctx):
    from tabpack_repro.metrics import compute_metrics

    ctx = churn_ctx
    generator = torch.Generator().manual_seed(0)
    for part in PARTS:
        y = ctx.dataset.y[part]
        n = _CHURN_SIZES[part]
        random = torch.rand(n, generator=generator)
        predictions = torch.stack([y.float(), torch.zeros(n), random])
        scores = ctx.score_fns[part](predictions)
        assert scores.shape == (3,)
        assert scores[0].item() == 1.0
        # Predicting "no churn" everywhere scores the majority-class rate.
        assert scores[1].item() == pytest.approx(1.0 - ctx.y_true[part].mean())
        expected = compute_metrics(ctx.y_true[part], random.numpy(), ctx.dataset.task)
        assert scores[2].item() == pytest.approx(expected['score'])


@pytest.mark.data
def test_setup_run_real_env(churn_ctx):
    env = churn_ctx.env
    assert {'device', 'gpu', 'torch', 'git_commit'} <= set(env)
    assert env['device'] == 'cpu'
    assert env['gpu'] is None
    assert env['torch'] == torch.__version__
    # The tests run inside a git checkout.
    assert isinstance(env['git_commit'], str)
    assert len(env['git_commit']) == 40


@pytest.mark.data
def test_setup_run_real_seeds_global_rngs(churn_dir):
    data = DataConfig(path=str(churn_dir))
    training = TrainingConfig(device='cpu', amp_dtype=None)
    draws = []
    for _ in range(2):
        common.setup_run(123, data, training)
        draws.append((torch.rand(4), np.random.rand(4)))
    torch.testing.assert_close(draws[0][0], draws[1][0], rtol=0, atol=0)
    np.testing.assert_array_equal(draws[0][1], draws[1][1])
