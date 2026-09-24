"""Parity of `tabpack_repro.sampler` with the official hyperparameter sampler.

The official side is `project.tabpack.HyperparameterSampler(type='RandomSampler', ...)`
with `ask(i)` for i in range(n), i.e. `lib.tools.tune._sample_config` on the trials of
a seeded optuna RandomSampler study. Our side is `sample_configs(space, n, seed=s)`.
Configs must be identical: same keys in the same order, same Python types and
bitwise-equal floats. The recorded trials (labels, values, distributions) must match
too, and seed 0 must reproduce the member configs of the official Churn run.

`project.tabpack` imports `rtdl_num_embeddings` / `rtdl_revisiting_models` at module
level (only for `isinstance` checks in `lib.optim.utils`). They are in the parity
dependency group; if they are missing anyway (e.g. an install without that group),
placeholder modules are used while importing, since the sampler never touches them.
The clone itself is never modified.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import optuna
import pytest

from tabpack_repro.config import TabPackConfig
from tabpack_repro.sampler import sample_config, sample_configs

SEEDS = (0, 1, 2, 3, 4)
N_VALUES = (1, 32, 64)
STUDY_KWARGS = {'study_name': 'x', 'direction': 'maximize'}

# Placeholder classes for the optional modules that `lib.optim.utils` imports.
_STUBS = {
    'rtdl_num_embeddings': ('LinearEmbeddings', 'LinearReLUEmbeddings', '_Periodic'),
    'rtdl_revisiting_models': ('LinearEmbeddings',),
}

# Everything that both implementations support: constants of every kind (incl. None,
# bytes, empty containers), nested dicts, plain lists holding subspaces, int/uniform
# with step, loguniform, categorical, '?' variants (with and without step), $list.
RICH_SPACE: dict[str, Any] = {
    'seed_offset': 7,
    'flag': True,
    'name': 'mlp',
    'blob': b'\x00raw',
    'ratio': 0.25,
    'nothing': None,
    'empty_list': [],
    'empty_dict': {},
    'model': {
        'n_blocks': ['_tune_', 'int', 1, 4],
        'd_block': ['_tune_', 'int', 64, 1024, 16],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
        'activation': ['_tune_', 'categorical', ['ReLU', 'GELU', 'SiLU']],
        'embeddings': {
            'type': 'PeriodicEmbeddings',
            'n_frequencies': ['_tune_', 'int', 16, 96, 8],
            'frequency_init_scale': ['_tune_', '?loguniform', 0.01, 0.01, 10.0],
            'd_embedding': ['_tune_', '?int', 16, 8, 64, 8],
            'lite': ['_tune_', 'categorical', [False, True]],
        },
        'layers': [
            {'width': ['_tune_', 'int', 8, 64, 8], 'bias': True},
            {'width': 32, 'gain': ['_tune_', 'uniform', 0.5, 2.0, 0.25]},
            ['_tune_', 'uniform', -1.0, 1.0],
            'constant-item',
        ],
    },
    'optimizer': {
        'lr': ['_tune_', 'loguniform', 1e-5, 1e-2],
        'weight_decay': ['_tune_', '?loguniform', 0.0, 1e-6, 1e-3],
        'betas': [0.9, ['_tune_', 'uniform', 0.95, 0.999]],
        'momentum': ['_tune_', 'uniform', 0.8, 0.99],
    },
    'tags': ['x', ['_tune_', 'int', 0, 3]],
    'batch_size': ['_tune_', '?categorical', 256, [64, 128, 512]],
    'per_feature_scale': ['_tune_', '$list', 5, 'loguniform', 0.1, 10.0],
    'per_feature_bins': ['_tune_', '$list', 3, 'int', 2, 64, 2],
    'per_feature_kind': ['_tune_', '$list', 4, 'categorical', ['a', 'b', 'c']],
}

SPACES = {
    'churn': lambda: TabPackConfig().space,
    'rich': lambda: copy.deepcopy(RICH_SPACE),
}


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Helpers
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@contextlib.contextmanager
def _quiet_optuna() -> Iterator[None]:
    # The official code does not silence "A new study created in memory ...".
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(verbosity)


def _assert_identical(ours: Any, theirs: Any, path: str = '<root>') -> None:
    """Strict equality: same types, same dict key order, bitwise-equal floats."""
    assert type(ours) is type(theirs), (
        f'{path}: {type(ours).__name__} {ours!r} != {type(theirs).__name__} {theirs!r}'
    )
    if isinstance(ours, dict):
        assert list(ours) == list(theirs), (
            f'{path}: keys {list(ours)} != {list(theirs)}'
        )
        for key in ours:
            _assert_identical(ours[key], theirs[key], f'{path}.{key}')
    elif isinstance(ours, list):
        assert len(ours) == len(theirs), f'{path}: {ours!r} != {theirs!r}'
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            _assert_identical(a, b, f'{path}.{i}')
    elif isinstance(ours, float):
        assert ours.hex() == theirs.hex(), f'{path}: {ours!r} != {theirs!r}'
    else:
        assert ours == theirs, f'{path}: {ours!r} != {theirs!r}'


def _official_sampler(tabpack: ModuleType, space: dict[str, Any], seed: int) -> Any:
    with _quiet_optuna():
        return tabpack.HyperparameterSampler(
            type='RandomSampler',
            space=space,
            seed=seed,
            study_kwargs=dict(STUDY_KWARGS),
        )


def _official_configs(
    tabpack: ModuleType, space: dict[str, Any], n: int, seed: int
) -> list[dict[str, Any]]:
    sampler = _official_sampler(tabpack, space, seed)
    return [sampler.ask(i) for i in range(n)]


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Fixtures
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.fixture(scope='module')
def tabpack(official) -> Iterator[ModuleType]:
    """The official `project.tabpack`, imported with placeholders for missing deps."""
    added = []
    for name, class_names in _STUBS.items():
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        module = ModuleType(name)
        module.__doc__ = 'Placeholder installed by tests/parity/test_parity_sampler.py'
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (), {}))
        sys.modules[name] = module
        added.append(name)
    try:
        yield official('project.tabpack')
    finally:
        for name in added:
            sys.modules.pop(name, None)


@pytest.fixture(scope='module')
def churn_run(official) -> Path:
    """The official Churn TabPack run shipped with the clone."""
    root = Path(official('lib').__file__).resolve().parents[2]
    run = root / 'experiments' / 'tabpack' / 'churn' / 'main'
    if not (run / 'report.json').exists() or not (run / 'config.json').exists():
        pytest.skip(f'The official Churn run is missing at {run}')
    return run


@pytest.fixture(scope='module')
def recorded_configs(churn_run: Path) -> dict[int, dict[str, Any]]:
    """Member id -> recorded config of the official Churn run."""
    report = json.loads((churn_run / 'report.json').read_text())
    recorded = {e['report']['id']: e['config'] for e in report['experiments']}
    assert len(recorded) == len(report['experiments']), 'duplicate member ids'
    return recorded


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# sample_configs == HyperparameterSampler.ask
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.mark.parametrize('n', N_VALUES)
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('space_name', list(SPACES))
def test_sample_configs_matches_official_sampler(tabpack, space_name, seed, n):
    space = SPACES[space_name]()
    original = copy.deepcopy(space)

    theirs = _official_configs(tabpack, copy.deepcopy(space), n, seed)
    ours = sample_configs(space, n, seed=seed)

    assert len(ours) == len(theirs) == n
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        _assert_identical(a, b, f'member[{i}]')
    assert space == original, 'sample_configs mutated the space'


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('space_name', list(SPACES))
def test_recorded_trials_match_official_sampler(tabpack, space_name, seed):
    # Same optuna calls in the same order: labels (incl. the '?<label>' flags),
    # values and distributions of every trial coincide.
    n = 32
    space = SPACES[space_name]()

    official_sampler = _official_sampler(tabpack, copy.deepcopy(space), seed)
    for i in range(n):
        official_sampler.ask(i)
    theirs = official_sampler._study.get_trials(deepcopy=False)

    with _quiet_optuna():
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=seed), **STUDY_KWARGS
        )
    for _ in range(n):
        sample_config(study.ask(), space)
    ours = study.get_trials(deepcopy=False)

    assert len(ours) == len(theirs) == n
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        _assert_identical(a.params, b.params, f'trial[{i}].params')
        assert a.distributions == b.distributions, f'trial[{i}].distributions'


def test_rich_space_exercises_every_feature():
    # Guard against the rich space silently degenerating: across 64 members, every
    # '?' flag takes both values and every step/categorical/$list leaf is present.
    configs = sample_configs(copy.deepcopy(RICH_SPACE), 64, seed=0)
    model = [c['model'] for c in configs]
    assert {m['dropout'] == 0.0 for m in model} == {False, True}
    assert {m['embeddings']['d_embedding'] == 16 for m in model} == {False, True}
    assert {c['batch_size'] == 256 for c in configs} == {False, True}
    assert {c['optimizer']['weight_decay'] == 0.0 for c in configs} == {False, True}
    assert all(m['d_block'] % 16 == 0 for m in model)
    assert all((m['layers'][1]['gain'] - 0.5) % 0.25 == 0 for m in model)
    assert {m['activation'] for m in model} == {'ReLU', 'GELU', 'SiLU'}
    assert all(len(c['per_feature_scale']) == 5 for c in configs)
    assert all(b % 2 == 0 for c in configs for b in c['per_feature_bins'])
    assert all(c['nothing'] is None and c['blob'] == b'\x00raw' for c in configs)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Regression: the official Churn run
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def test_official_churn_run_uses_our_space(churn_run):
    config = json.loads((churn_run / 'config.json').read_text())
    assert config['seed'] == 0
    assert config['n_models'] == 64
    assert config['sampler']['type'] == 'RandomSampler'
    assert set(config['sampler']) == {'type', 'space'}
    _assert_identical(TabPackConfig().space, config['sampler']['space'])


def test_official_churn_report_coverage(recorded_configs):
    # 57 of the 64 members were recorded; pin the set so that the checks below
    # cannot pass vacuously if the report changes.
    missing = sorted(set(range(64)) - set(recorded_configs))
    assert missing == [10, 12, 24, 25, 28, 31, 44]
    assert len(recorded_configs) == 57


@pytest.mark.parametrize('n', N_VALUES)
def test_seed0_reproduces_official_churn_report(recorded_configs, n):
    ours = sample_configs(TabPackConfig().space, n, seed=0)
    checked = [i for i in range(n) if i in recorded_configs]
    for i in checked:
        _assert_identical(ours[i], recorded_configs[i], f'member[{i}]')
    assert len(checked) == sum(1 for i in recorded_configs if i < n)


def test_official_sampler_reproduces_its_churn_report(tabpack, recorded_configs):
    # Control: the official code under the installed optuna reproduces its own run,
    # so a failure of the test above is ours, not an optuna version drift.
    theirs = _official_configs(tabpack, TabPackConfig().space, 64, 0)
    for i, config in recorded_configs.items():
        _assert_identical(theirs[i], config, f'member[{i}]')
