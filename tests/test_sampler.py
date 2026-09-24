"""Tests for tabpack_repro.sampler (a24)."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import optuna
import pytest

from tabpack_repro.config import TabPackConfig
from tabpack_repro.sampler import sample_config, sample_configs


def official_space() -> dict[str, Any]:
    return TabPackConfig().space


class RecordingTrial:
    """A fake optuna trial that records every suggest_* call.

    Numeric suggestions return `low`; categorical ones return `choices[pick]`.
    """

    def __init__(self, pick: int = -1) -> None:
        self.pick = pick
        self.calls: list[tuple[str, str, tuple, dict]] = []

    def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
        self.calls.append(('int', name, args, kwargs))
        return args[0]

    def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        self.calls.append(('float', name, args, kwargs))
        return args[0]

    def suggest_categorical(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(('categorical', name, args, kwargs))
        return args[0][self.pick]


def _study_trial(seed: int = 0) -> optuna.trial.Trial:
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    finally:
        optuna.logging.set_verbosity(verbosity)
    return study.ask()


# ---------------------------------------------------------------------------
# Ranges and distributions
# ---------------------------------------------------------------------------
def test_official_space_ranges_and_types() -> None:
    configs = sample_configs(official_space(), 256, seed=0)
    assert len(configs) == 256
    for c in configs:
        assert set(c) == {'model', 'optimizer'}
        assert set(c['model']) == {'n_blocks', 'dropout'}
        assert set(c['optimizer']) == {'lr', 'weight_decay', 'muon_lr'}
        n_blocks = c['model']['n_blocks']
        assert type(n_blocks) is int and 1 <= n_blocks <= 4
        assert type(c['model']['dropout']) is float
        assert 0.0 <= c['model']['dropout'] <= 0.5
        for key, (low, high) in {
            'lr': (1e-4, 5e-3),
            'weight_decay': (1e-3, 1.0),
            'muon_lr': (1e-3, 0.1),
        }.items():
            value = c['optimizer'][key]
            assert type(value) is float and low <= value <= high, (key, value)
    # All four n_blocks values occur, and the int bounds are inclusive.
    assert {c['model']['n_blocks'] for c in configs} == {1, 2, 3, 4}


def test_loguniform_is_log_distributed() -> None:
    space = {'x': ['_tune_', 'loguniform', 1e-4, 1.0]}
    values = [c['x'] for c in sample_configs(space, 2000, seed=1)]
    logs = [math.log10(v) for v in values]
    # Uniform in log10-space over [-4, 0]: each decade gets ~25% of the draws,
    # whereas a linear uniform would put ~90% of them into [0.1, 1].
    for decade in range(-4, 0):
        share = sum(decade <= lg < decade + 1 for lg in logs) / len(logs)
        assert 0.2 < share < 0.3, (decade, share)


def test_uniform_is_linear() -> None:
    values = [
        c['x']
        for c in sample_configs({'x': ['_tune_', 'uniform', 0.0, 1.0]}, 2000, seed=2)
    ]
    assert all(0.0 <= v <= 1.0 for v in values)
    share_top = sum(v >= 0.5 for v in values) / len(values)
    assert 0.45 < share_top < 0.55


def test_step_argument() -> None:
    space = {
        'i': ['_tune_', 'int', 0, 10, 5],
        'f': ['_tune_', 'uniform', 0.0, 1.0, 0.25],
    }
    configs = sample_configs(space, 200, seed=0)
    assert {c['i'] for c in configs} == {0, 5, 10}
    assert {c['f'] for c in configs} <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert len({c['f'] for c in configs}) == 5


def test_categorical() -> None:
    choices = ['ReLU', 'GELU', 'SiLU']
    configs = sample_configs({'act': ['_tune_', 'categorical', choices]}, 100, seed=0)
    assert {c['act'] for c in configs} == set(choices)


# ---------------------------------------------------------------------------
# '?' (optional) distributions
# ---------------------------------------------------------------------------
def test_optional_uniform_default_and_sampled_members() -> None:
    configs = sample_configs(official_space(), 64, seed=0)
    dropouts = [c['model']['dropout'] for c in configs]
    n_default = sum(d == 0.0 for d in dropouts)
    # Coin flip per member: both branches occur in 64 draws.
    assert 0 < n_default < 64
    assert all(0.0 < d <= 0.5 for d in dropouts if d != 0.0)


def test_optional_default_is_exact_and_outside_the_sampled_range() -> None:
    space = {'p': ['_tune_', '?uniform', 0.25, 0.5, 1.0]}
    values = [c['p'] for c in sample_configs(space, 200, seed=3)]
    defaults = [v for v in values if v == 0.25]
    sampled = [v for v in values if v != 0.25]
    assert defaults and sampled
    assert all(0.5 <= v <= 1.0 for v in sampled)
    # The coin is fair.
    assert 0.35 < len(defaults) / len(values) < 0.65


def test_optional_default_not_sampled_branch_uses_default_of_any_type() -> None:
    default = {'nested': [1, 2]}
    trial = RecordingTrial(pick=0)  # suggest_categorical -> False
    value = sample_config(trial, ['_tune_', '?int', default, 1, 4], ['a'])
    assert value == default
    assert value is not default  # no aliasing between members
    assert trial.calls == [('categorical', '?a', ([False, True],), {})]


def test_optional_records_flip_and_value_in_optuna_trial() -> None:
    for seed in range(20):
        trial = _study_trial(seed)
        config = sample_config(
            trial, {'m': {'d': ['_tune_', '?uniform', 0.0, 0.0, 0.5]}}
        )
        assert '?m.d' in trial.params
        if trial.params['?m.d']:
            assert trial.params['m.d'] == config['m']['d']
        else:
            assert 'm.d' not in trial.params
            assert config['m']['d'] == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_seed_same_configs() -> None:
    a = sample_configs(official_space(), 32, seed=7)
    b = sample_configs(official_space(), 32, seed=7)
    assert a == b


def test_different_seeds_differ() -> None:
    a = sample_configs(official_space(), 32, seed=0)
    b = sample_configs(official_space(), 32, seed=1)
    assert a != b
    # Not just a permutation / partial overlap: the lr values are disjoint.
    assert {c['optimizer']['lr'] for c in a}.isdisjoint(
        {c['optimizer']['lr'] for c in b}
    )


def test_prefix_property() -> None:
    # Member i depends only on the seed and i (sequential study.ask()), so a
    # smaller n yields a prefix of a larger n.
    small = sample_configs(official_space(), 5, seed=4)
    large = sample_configs(official_space(), 40, seed=4)
    assert large[:5] == small


def test_members_differ_within_one_call() -> None:
    configs = sample_configs(official_space(), 32, seed=0)
    assert len({c['optimizer']['lr'] for c in configs}) == 32


def test_does_not_touch_global_numpy_or_python_rng() -> None:
    import random

    import numpy as np

    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.rand())
    random.seed(123)
    np.random.seed(123)
    sample_configs(official_space(), 16, seed=0)
    assert (random.random(), np.random.rand()) == expected


# ---------------------------------------------------------------------------
# Structure pass-through and labels
# ---------------------------------------------------------------------------
def test_constants_nested_dicts_and_lists_pass_through() -> None:
    space = {
        'int': 1,
        'float': 2.5,
        'str': 'ReLU',
        'bool': True,
        'none': None,
        'bytes': b'z',
        'list': [1, 'a', [2.0, {'deep': False}]],
        'empty_list': [],
        'empty_dict': {},
        'dict': {'a': {'b': {'c': 3}}},
    }
    (config,) = sample_configs(space, 1, seed=0)
    assert config == space


def test_tuned_values_inside_lists_and_nested_dicts() -> None:
    space = {
        'layers': [
            {'width': ['_tune_', 'int', 8, 16], 'act': 'ReLU'},
            {'width': 32},
        ],
        'a': {'b': {'c': ['_tune_', 'uniform', 0.0, 1.0]}},
    }
    for config in sample_configs(space, 20, seed=0):
        assert 8 <= config['layers'][0]['width'] <= 16
        assert config['layers'][0]['act'] == 'ReLU'
        assert config['layers'][1] == {'width': 32}
        assert 0.0 <= config['a']['b']['c'] <= 1.0


def test_members_are_independent_objects() -> None:
    space = {'const': [1, 2], 'nested': {'x': 1}}
    a, b = sample_configs(space, 2, seed=0)
    a['const'].append(3)
    a['nested']['x'] = 2
    assert b == space
    assert space == {'const': [1, 2], 'nested': {'x': 1}}


def test_official_space_call_sequence_and_labels() -> None:
    # The exact sequence of optuna calls determines the RNG stream, hence the
    # parity with the official configs.
    trial = RecordingTrial(pick=-1)  # the '?' coin -> True
    config = sample_config(trial, official_space())
    assert trial.calls == [
        ('int', 'model.n_blocks', (1, 4), {}),
        ('categorical', '?model.dropout', ([False, True],), {}),
        ('float', 'model.dropout', (0.0, 0.5), {}),
        ('float', 'optimizer.lr', (0.0001, 0.005), {'log': True}),
        ('float', 'optimizer.weight_decay', (0.001, 1.0), {'log': True}),
        ('float', 'optimizer.muon_lr', (0.001, 0.1), {'log': True}),
    ]
    assert config == {
        'model': {'n_blocks': 1, 'dropout': 0.0},
        'optimizer': {'lr': 0.0001, 'weight_decay': 0.001, 'muon_lr': 0.001},
    }


def test_labels_for_lists_label_parts_and_dollar_list() -> None:
    trial = RecordingTrial()
    space = {
        'layers': [
            {'w': ['_tune_', 'int', 1, 2]},
            ['_tune_', 'uniform', 0.0, 1.0, 0.5],
        ],
        'per_feature': ['_tune_', '$list', 3, 'loguniform', 0.1, 1.0],
    }
    sample_config(trial, space, ['root'])
    assert trial.calls == [
        ('int', 'root.layers.0.w', (1, 2), {}),
        ('float', 'root.layers.1', (0.0, 1.0), {'step': 0.5}),
        ('float', 'root.per_feature.0', (0.1, 1.0), {'log': True}),
        ('float', 'root.per_feature.1', (0.1, 1.0), {'log': True}),
        ('float', 'root.per_feature.2', (0.1, 1.0), {'log': True}),
    ]


def test_label_parts_default_and_non_str_parts() -> None:
    trial = RecordingTrial()
    sample_config(trial, ['_tune_', 'int', 1, 2])
    sample_config(trial, ['_tune_', 'int', 1, 2], ['a', 0, 'b'])  # type: ignore[list-item]
    assert [c[1] for c in trial.calls] == ['', 'a.0.b']


def test_optuna_trial_param_names() -> None:
    trial = _study_trial(0)
    sample_config(trial, official_space())
    assert set(trial.params) >= {
        'model.n_blocks',
        '?model.dropout',
        'optimizer.lr',
        'optimizer.weight_decay',
        'optimizer.muon_lr',
    }


# ---------------------------------------------------------------------------
# The default TabPack space
# ---------------------------------------------------------------------------
def test_default_tabpack_space_n32() -> None:
    config = TabPackConfig()
    configs = sample_configs(config.space, config.n_models, seed=config.seed)
    assert config.n_models == 32
    assert len(configs) == 32
    assert all(set(c) == {'model', 'optimizer'} for c in configs)
    # JSON round trip (configs are stored in run reports).
    assert json.loads(json.dumps(configs)) == configs
    # The space itself is not mutated.
    assert config.space == TabPackConfig().space


def test_n_zero() -> None:
    assert sample_configs(official_space(), 0, seed=0) == []


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'space',
    [
        {'x': ['_tune_', 'normal', 0.0, 1.0]},
        {'x': ['_tune_', '?normal', 0.0, 0.0, 1.0]},
        {'x': ['_tune_']},
        {'x': ['_tune_', '?uniform']},
        {'x': {'_tune_': '$custom', 'a': 0}},
    ],
)
def test_invalid_spaces_raise_value_error(space: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sample_configs(space, 1, seed=0)


def test_unknown_distribution_error_names_the_label() -> None:
    with pytest.raises(ValueError, match=r'model\.x'):
        sample_configs({'model': {'x': ['_tune_', 'normal', 0.0, 1.0]}}, 1, seed=0)


def test_unsupported_types_raise_type_error() -> None:
    with pytest.raises(TypeError, match='set'):
        sample_configs({'x': {1, 2}}, 1, seed=0)
    with pytest.raises(TypeError):
        sample_configs([1, 2], 1, seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize('n', [-1, 1.5, True])
def test_bad_n(n: Any) -> None:
    with pytest.raises(ValueError):
        sample_configs(official_space(), n, seed=0)


# ---------------------------------------------------------------------------
# optuna logging
# ---------------------------------------------------------------------------
class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def optuna_records():
    logger = logging.getLogger('optuna')
    handler = _Collect()
    logger.addHandler(handler)
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        optuna.logging.set_verbosity(verbosity)


def test_optuna_info_logs_are_silenced_and_verbosity_restored(optuna_records) -> None:
    # Control: creating a study at INFO verbosity does log.
    optuna.create_study()
    assert any(r.levelno == logging.INFO for r in optuna_records)
    optuna_records.clear()

    sample_configs(official_space(), 8, seed=0)
    assert not [r for r in optuna_records if r.levelno < logging.WARNING]
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


def test_verbosity_restored_on_error(optuna_records) -> None:
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    with pytest.raises(ValueError):
        sample_configs({'x': ['_tune_', 'normal', 0.0, 1.0]}, 1, seed=0)
    assert optuna.logging.get_verbosity() == optuna.logging.DEBUG


# ---------------------------------------------------------------------------
# The official Churn run (reference clone, data only)
# ---------------------------------------------------------------------------
def _official_churn_run() -> Path:
    reference = os.environ.get('TABPACK_REFERENCE_DIR')
    if not reference:
        pytest.skip('TABPACK_REFERENCE_DIR is not set')
    run = Path(reference) / 'experiments' / 'tabpack' / 'churn' / 'main'
    if not (run / 'report.json').exists() or not (run / 'config.json').exists():
        pytest.skip(f'The official Churn run is not available at {run}')
    return run


def test_default_space_equals_the_official_churn_space() -> None:
    run = _official_churn_run()
    official = json.loads((run / 'config.json').read_text())
    assert official['sampler']['type'] == 'RandomSampler'
    assert official['sampler']['space'] == official_space()


def test_matches_the_configs_of_the_official_churn_run() -> None:
    # The official run: HyperparameterSampler(type='RandomSampler', seed=0), 64
    # members, member i = ask(i). The report keeps the members that survived,
    # each with its config and member id.
    run = _official_churn_run()
    official = json.loads((run / 'config.json').read_text())
    report = json.loads((run / 'report.json').read_text())
    assert official['seed'] == 0 and official['n_models'] == 64

    configs = sample_configs(official['sampler']['space'], 64, seed=official['seed'])
    experiments = report['experiments']
    assert len(experiments) > 0
    mismatches = [
        (e['report']['id'], configs[e['report']['id']], e['config'])
        for e in experiments
        if configs[e['report']['id']] != e['config']
    ]
    # Exact (bitwise float) equality for every recorded member.
    assert not mismatches, mismatches[:3]
