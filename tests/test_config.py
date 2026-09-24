"""Tests for the config loaders (a28): config_from_dict, load_config, dump_config and
the TOML files in configs/churn/."""

from __future__ import annotations

import dataclasses
import math
import random
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tabpack_repro.config import (
    CONFIG_CLASSES,
    AdamWConfig,
    ConservativeEvalConfig,
    DataConfig,
    HomogeneousEnsembleConfig,
    MLPMethodConfig,
    MLPModelConfig,
    MuonAdamWConfig,
    OnlineEnsembleConfig,
    TabPackConfig,
    TrainingConfig,
    config_from_dict,
    config_to_dict,
    dump_config,
    load_config,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_DIR / 'configs' / 'churn'

MEMBER_CONFIGS = [
    {
        'model': {'n_blocks': 2, 'dropout': 0.0},
        'optimizer': {'lr': 0.0001, 'weight_decay': 0.005, 'muon_lr': 0.01},
    },
    {
        'model': {'n_blocks': 4, 'dropout': 0.3217},
        'optimizer': {'lr': 1e-4, 'weight_decay': 1.0, 'muon_lr': 5e-3},
    },
]


def strict(value: Any) -> Any:
    """A comparable form of `value` that also distinguishes types (1 vs 1.0 vs True,
    list vs tuple, 0.0 vs -0.0), so that equality below means an exact round trip."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple(
                (f.name, strict(getattr(value, f.name)))
                for f in dataclasses.fields(value)
            ),
        )
    if isinstance(value, dict):
        return ('dict', sorted((k, strict(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return (type(value).__name__, [strict(v) for v in value])
    if isinstance(value, float):
        return ('float', repr(value))
    return (type(value).__name__, value)


def roundtrip(config: Any, tmp_path: Path) -> tuple[Any, str]:
    path = tmp_path / 'config.toml'
    dump_config(config, path)
    return load_config(path), path.read_text(encoding='utf-8')


def assert_roundtrip(config: Any, tmp_path: Path) -> str:
    loaded, text = roundtrip(config, tmp_path)
    assert loaded == config
    assert strict(loaded) == strict(config)
    return text


def non_default_configs() -> list[Any]:
    data = DataConfig(path='/abs/path/to/churn', seed=3, extract_bin_from_num=False)
    training = TrainingConfig(
        batch_size=64, patience=0, max_epochs=5, amp_dtype='float16', device='cpu'
    )
    return [
        MLPMethodConfig(
            seed=7,
            data=data,
            model=MLPModelConfig(n_blocks=1, d_block=8, dropout=0.0, activation='GELU'),
            optimizer=AdamWConfig(lr=3e-4, weight_decay=0.0, beta2=0.95, eps=1e-6),
            training=training,
        ),
        HomogeneousEnsembleConfig(seed=4, n_models=3, data=data, training=training),
        TabPackConfig(
            seed=2,
            n_models=2,
            d_block=16,
            activation='SiLU',
            optimizer=MuonAdamWConfig(muon_nesterov=False, shared_step=False),
            online_ensemble=OnlineEnsembleConfig(
                include_current_ensemble_in_pool=False, max_ensemble_size=1
            ),
            configs=MEMBER_CONFIGS,
        ),
        ConservativeEvalConfig(source_run='runs/churn/tabpack/seed-0', n_seeds=2),
    ]


# --- round trips ------------------------------------------------------------------


@pytest.mark.parametrize('cls', list(CONFIG_CLASSES.values()), ids=list(CONFIG_CLASSES))
def test_default_configs_roundtrip(cls: type, tmp_path: Path) -> None:
    text = assert_roundtrip(cls(), tmp_path)
    assert f'method = "{cls().method}"' in text
    tomllib.loads(text)  # Plain TOML, readable without our loader.


@pytest.mark.parametrize(
    'config', non_default_configs(), ids=lambda c: type(c).__name__
)
def test_non_default_configs_roundtrip(config: Any, tmp_path: Path) -> None:
    assert_roundtrip(config, tmp_path)


def test_tabpack_configs_none_empty_and_list_roundtrip(tmp_path: Path) -> None:
    text = assert_roundtrip(TabPackConfig(configs=None), tmp_path)
    assert 'configs' not in tomllib.loads(text)

    text = assert_roundtrip(TabPackConfig(configs=[]), tmp_path)
    assert tomllib.loads(text)['configs'] == []

    config = TabPackConfig(n_models=2, configs=MEMBER_CONFIGS)
    text = assert_roundtrip(config, tmp_path)
    assert text.count('[[configs]]') == 2
    assert tomllib.loads(text)['configs'] == MEMBER_CONFIGS


def test_tabpack_member_configs_with_nested_tables_roundtrip(tmp_path: Path) -> None:
    configs = [
        {},
        {'model': {}},
        {'model': {'n_blocks': 1, 'extra': {'deep': {'x': [1, 2.5, 'a']}}}},
        {'tags': [{'a': 1}, {'b': [True, False]}], 'grid': [[1, 2], [3.0], []]},
    ]
    assert_roundtrip(TabPackConfig(n_models=4, configs=configs), tmp_path)


def test_config_dict_roundtrip() -> None:
    for config in [cls() for cls in CONFIG_CLASSES.values()] + non_default_configs():
        rebuilt = config_from_dict(config_to_dict(config))
        assert rebuilt == config
        assert strict(rebuilt) == strict(config)


def test_dump_then_load_is_idempotent_on_text(tmp_path: Path) -> None:
    config = non_default_configs()[2]
    first = tmp_path / 'a.toml'
    second = tmp_path / 'b.toml'
    dump_config(config, first)
    dump_config(load_config(first), second)
    assert first.read_text() == second.read_text()


# --- None handling ----------------------------------------------------------------


@pytest.mark.parametrize(
    ('section', 'key'),
    [
        ('data', 'num_policy'),
        ('data', 'bin_policy'),
        ('data', 'cat_policy'),
        ('training', 'amp_dtype'),
    ],
)
def test_explicit_none_in_optional_strings_roundtrips(
    section: str, key: str, tmp_path: Path
) -> None:
    for cls in (MLPMethodConfig, HomogeneousEnsembleConfig, TabPackConfig):
        base = cls()
        sub = dataclasses.replace(getattr(base, section), **{key: None})
        config = dataclasses.replace(base, **{section: sub})
        text = assert_roundtrip(config, tmp_path)
        assert tomllib.loads(text)[section][key] == 'none'
        assert getattr(
            getattr(load_config(tmp_path / 'config.toml'), section), key
        ) is (None)


def test_all_optional_strings_none_at_once(tmp_path: Path) -> None:
    config = TabPackConfig(
        data=DataConfig(num_policy=None, bin_policy=None, cat_policy=None),
        training=TrainingConfig(amp_dtype=None),
        configs=None,
    )
    assert_roundtrip(config, tmp_path)


def test_none_string_and_null_mean_none_in_config_from_dict() -> None:
    for value in ('none', None):
        config = config_from_dict(
            {
                'method': 'mlp',
                'data': {'num_policy': value, 'cat_policy': value},
                'training': {'amp_dtype': value},
            }
        )
        assert config.data.num_policy is None
        assert config.data.cat_policy is None
        assert config.data.bin_policy == 'convert-to-cat'
        assert config.training.amp_dtype is None
    # Only the exact lowercase string is the sentinel.
    config = config_from_dict({'method': 'mlp', 'training': {'amp_dtype': 'None'}})
    assert config.training.amp_dtype == 'None'
    # Non-optional string fields keep 'none' as a plain string.
    config = config_from_dict({'method': 'mlp', 'training': {'device': 'none'}})
    assert config.training.device == 'none'


def test_null_configs_in_dict_means_default() -> None:
    config = config_from_dict({'method': 'tabpack', 'configs': None})
    assert config.configs is None


def test_dump_rejects_the_reserved_none_string(tmp_path: Path) -> None:
    config = MLPMethodConfig(training=TrainingConfig(amp_dtype='none'))
    with pytest.raises(ValueError, match=r'training\.amp_dtype.*reserved'):
        dump_config(config, tmp_path / 'config.toml')


def test_dump_rejects_none_in_non_optional_field(tmp_path: Path) -> None:
    config = MLPMethodConfig(seed=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'seed'"):
        dump_config(config, tmp_path / 'config.toml')


def test_dump_rejects_none_inside_free_form_values(tmp_path: Path) -> None:
    config = TabPackConfig(space={'model': {'dropout': None}})
    with pytest.raises(ValueError, match=r'space\.model\.dropout'):
        dump_config(config, tmp_path / 'config.toml')
    config = TabPackConfig(n_models=1, configs=[{'optimizer': {'lr': [1, None]}}])
    with pytest.raises(ValueError, match=r'configs\.optimizer\.lr\[1\]'):
        dump_config(config, tmp_path / 'config.toml')


def test_dump_rejects_unsupported_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r'space\.x'):
        dump_config(TabPackConfig(space={'x': {1, 2}}), tmp_path / 'c.toml')
    with pytest.raises(TypeError, match='keys must be strings'):
        dump_config(TabPackConfig(space={'x': {1: 2}}), tmp_path / 'c.toml')
    with pytest.raises(TypeError):
        dump_config({'method': 'mlp'}, tmp_path / 'c.toml')  # type: ignore[arg-type]


def test_dump_rejects_method_of_another_class(tmp_path: Path) -> None:
    config = MLPMethodConfig(method='homogeneous')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='different config class'):
        dump_config(config, tmp_path / 'c.toml')


# --- config_from_dict: defaults, unknown keys, types ------------------------------


@pytest.mark.parametrize('method', list(CONFIG_CLASSES))
def test_missing_keys_take_defaults(method: str) -> None:
    assert config_from_dict({'method': method}) == CONFIG_CLASSES[method]()


def test_partial_nested_tables_take_defaults() -> None:
    config = config_from_dict(
        {'method': 'tabpack', 'training': {'patience': 3}, 'online_ensemble': {}}
    )
    assert config == TabPackConfig(training=TrainingConfig(patience=3))


@pytest.mark.parametrize(
    ('data', 'path'),
    [
        ({'method': 'mlp', 'sede': 1}, 'sede'),
        ({'method': 'mlp', 'training': {'patiance': 3}}, 'training.patiance'),
        ({'method': 'tabpack', 'model': {}}, 'model'),
        (
            {'method': 'tabpack', 'online_ensemble': {'max_size': 3}},
            'online_ensemble.max_size',
        ),
        ({'method': 'homogeneous', 'data': {'policy': 'x'}}, 'data.policy'),
        ({'method': 'tabpack-conservative', 'data': {}}, 'data'),
        (
            {'method': 'mlp', 'optimizer': {'lr': 0.1, 'muon_lr': 0.1}},
            'optimizer.muon_lr',
        ),
    ],
)
def test_unknown_keys_raise_with_full_path(data: dict, path: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f'unknown config key {path!r}')):
        config_from_dict(data)


def test_unknown_key_message_suggests_and_lists_valid_keys() -> None:
    with pytest.raises(ValueError) as info:
        config_from_dict({'method': 'mlp', 'training': {'patiance': 3}})
    message = str(info.value)
    assert "did you mean 'training.patience'" in message
    assert 'batch_size' in message and 'amp_dtype' in message


def test_free_form_tables_accept_any_keys() -> None:
    space = {'model': {'anything': ['_tune_', 'int', 1, 2]}, 'new': 1}
    config = config_from_dict({'method': 'tabpack', 'space': space})
    assert config.space == space  # Replaces the default space, no merge.


def test_missing_or_unknown_method() -> None:
    with pytest.raises(ValueError, match="no 'method' key"):
        config_from_dict({'seed': 0})
    with pytest.raises(ValueError, match="unknown method 'xgboost'"):
        config_from_dict({'method': 'xgboost'})
    with pytest.raises(ValueError, match='unknown method'):
        config_from_dict({'method': ['mlp']})
    with pytest.raises(TypeError):
        config_from_dict([('method', 'mlp')])  # type: ignore[arg-type]


def test_int_is_accepted_for_float_fields() -> None:
    config = config_from_dict(
        {
            'method': 'mlp',
            'optimizer': {'lr': 1, 'weight_decay': 0},
            'model': {'dropout': 0},
        }
    )
    assert config.optimizer.lr == 1.0 and type(config.optimizer.lr) is float
    assert type(config.optimizer.weight_decay) is float
    assert type(config.model.dropout) is float


@pytest.mark.parametrize(
    ('data', 'path'),
    [
        ({'method': 'mlp', 'seed': 1.5}, 'seed'),
        ({'method': 'mlp', 'seed': 1.0}, 'seed'),
        ({'method': 'mlp', 'seed': True}, 'seed'),
        ({'method': 'mlp', 'seed': '0'}, 'seed'),
        ({'method': 'mlp', 'optimizer': {'lr': '1e-3'}}, 'optimizer.lr'),
        ({'method': 'mlp', 'optimizer': {'lr': False}}, 'optimizer.lr'),
        (
            {'method': 'mlp', 'data': {'extract_bin_from_num': 1}},
            'data.extract_bin_from_num',
        ),
        ({'method': 'mlp', 'data': {'path': 3}}, 'data.path'),
        ({'method': 'mlp', 'data': {'num_policy': 1}}, 'data.num_policy'),
        ({'method': 'mlp', 'training': 16}, 'training'),
        ({'method': 'mlp', 'training': {'patience': None}}, 'training.patience'),
        ({'method': 'tabpack', 'space': [1, 2]}, 'space'),
        ({'method': 'tabpack', 'space': {1: 2}}, 'space'),
        ({'method': 'tabpack', 'configs': {'model': {}}}, 'configs'),
        ({'method': 'tabpack', 'configs': [{'model': {}}, 3]}, 'configs[1]'),
        (
            {'method': 'tabpack', 'optimizer': {'muon_ns_steps': 5.0}},
            'optimizer.muon_ns_steps',
        ),
        ({'method': 'tabpack-conservative', 'n_seeds': '5'}, 'n_seeds'),
    ],
)
def test_wrong_types_raise_naming_the_key(data: dict, path: str) -> None:
    with pytest.raises(TypeError, match=re.escape(f"'{path}'")) as info:
        config_from_dict(data)
    assert isinstance(info.value, ValueError)  # Catchable as either.


def test_literal_fields_are_checked() -> None:
    with pytest.raises(ValueError, match=r"'online_ensemble\.type'.*'greedy'"):
        config_from_dict({'method': 'tabpack', 'online_ensemble': {'type': 'random'}})
    with pytest.raises(ValueError, match=r"'online_ensemble\.update_type'"):
        config_from_dict(
            {'method': 'tabpack', 'online_ensemble': {'update_type': 'best'}}
        )


def test_tuples_become_lists_and_input_is_not_aliased() -> None:
    space = {'model': {'n_blocks': ('_tune_', 'int', 1, 4)}}
    configs = [{'model': {'n_blocks': 1}}]
    config = config_from_dict(
        {'method': 'tabpack', 'space': space, 'configs': (configs[0],)}
    )
    assert config.space == {'model': {'n_blocks': ['_tune_', 'int', 1, 4]}}
    assert type(config.space['model']['n_blocks']) is list
    assert type(config.configs) is list
    configs[0]['model']['n_blocks'] = 99
    space['model']['extra'] = 1
    assert config.configs == [{'model': {'n_blocks': 1}}]
    assert 'extra' not in config.space['model']


def test_numpy_scalars_are_accepted() -> None:
    np = pytest.importorskip('numpy')
    config = config_from_dict(
        {'method': 'mlp', 'seed': np.int64(3), 'optimizer': {'lr': np.float32(0.5)}}
    )
    assert config.seed == 3 and type(config.seed) is int
    assert config.optimizer.lr == 0.5 and type(config.optimizer.lr) is float


# --- load_config ------------------------------------------------------------------


def test_load_config_errors_mention_the_file(tmp_path: Path) -> None:
    path = tmp_path / 'bad.toml'
    path.write_text('method = "mlp"\n[training]\npatiance = 3\n')
    with pytest.raises(ValueError, match=r'training\.patiance') as info:
        load_config(path)
    assert any(str(path) in note for note in info.value.__notes__)

    path.write_text('method = "mlp"\nseed = \n')
    with pytest.raises(tomllib.TOMLDecodeError) as decode_info:
        load_config(str(path))
    assert any(str(path) in note for note in decode_info.value.__notes__)


def test_dump_config_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / 'a' / 'b' / 'config.toml'
    dump_config(MLPMethodConfig(), str(path))
    assert load_config(path) == MLPMethodConfig()


# --- the TOML serializer ----------------------------------------------------------

FLOATS = [
    1e-4,
    0.0001,
    5e-3,
    0.005,
    1e-8,
    0.1,
    0.3217,
    1.0,
    0.0,
    -0.0,
    -2.5,
    1 / 3,
    2 / 3,
    1e16,
    1.5e300,
    5e-324,
    2.2250738585072014e-308,
    123456789.123456789,
    float('inf'),
    float('-inf'),
]


def test_floats_roundtrip_exactly(tmp_path: Path) -> None:
    space = {'floats': FLOATS, 'lr': 1e-4, 'wd': 5e-3, 'x': 0.0001}
    loaded, text = roundtrip(TabPackConfig(space=space), tmp_path)
    assert strict(loaded.space) == strict(space)
    assert [math.copysign(1.0, v) for v in loaded.space['floats']] == [
        math.copysign(1.0, v) for v in FLOATS
    ]
    assert 'lr = 0.0001' in text
    assert 'wd = 0.005' in text


def test_nan_is_written_as_toml_nan(tmp_path: Path) -> None:
    loaded, text = roundtrip(TabPackConfig(space={'x': float('nan')}), tmp_path)
    assert 'x = nan' in text
    assert math.isnan(loaded.space['x'])


def test_mixed_type_arrays_keep_their_types(tmp_path: Path) -> None:
    config = TabPackConfig()
    loaded, text = roundtrip(config, tmp_path)
    assert 'n_blocks = ["_tune_", "int", 1, 4]' in text
    assert 'dropout = ["_tune_", "?uniform", 0.0, 0.0, 0.5]' in text
    n_blocks = loaded.space['model']['n_blocks']
    assert [type(v) for v in n_blocks] == [str, str, int, int]
    dropout = loaded.space['model']['dropout']
    assert [type(v) for v in dropout] == [str, str, float, float, float]


def test_ints_and_bools_keep_their_types(tmp_path: Path) -> None:
    space = {'i': [0, -1, 2**62, -(2**63)], 'b': [True, False], 'f': [1.0, 0.0]}
    loaded, _ = roundtrip(TabPackConfig(space=space), tmp_path)
    assert strict(loaded.space) == strict(space)


def test_strings_and_keys_are_escaped(tmp_path: Path) -> None:
    tricky = [
        '',
        'plain',
        'with space',
        'quote " and \' apostrophe',
        'back\\slash',
        'new\nline and tab\t and cr\r',
        'bell\x07 del\x7f nul\x00 esc\x1b',
        'unicode: é ü 中文 ✓ 🙂',
        '# not a comment',
        '[not.a.table]',
        '"""',
    ]
    space = {
        'values': tricky,
        'dotted.key': 1,
        'with space': 2,
        '': 3,
        'quote"key': {'a\\b': 4},
        'ünï': 5,
        '123': 6,
        'true': 7,
        'bare-key_OK': 8,
    }
    config = ConservativeEvalConfig(source_run='runs/with "quotes"\\and\nnewline')
    assert_roundtrip(config, tmp_path)
    assert_roundtrip(TabPackConfig(space=space), tmp_path)


def test_nested_free_form_structures_roundtrip(tmp_path: Path) -> None:
    space = {
        'empty': {},
        'empty_list': [],
        'nested_empty': {'a': {}, 'b': {'c': {}}},
        'only_tables': {'x': {'y': {'z': 1}}},
        'lists': [[1, [2, [3.5, 'x']]], [], [[]]],
        'inline_tables_in_list': [1, {'a': 1, 'b': [{'c': 'd'}]}, 'x'],
        'list_of_tables': [{'a': 1}, {}, {'b': {'c': [1, 2]}, 'd': [{'e': 1}]}],
        'nested_lists_of_tables': [[{'a': 1}], [{'b': 2}, {}]],
    }
    assert_roundtrip(TabPackConfig(space=space), tmp_path)


def random_value(rng: random.Random, depth: int) -> Any:
    kinds = ['int', 'float', 'str', 'bool']
    if depth < 3:
        kinds += ['list', 'dict', 'dict']
    kind = rng.choice(kinds)
    if kind == 'int':
        return rng.randint(-(10**12), 10**12)
    if kind == 'float':
        return rng.choice(
            [rng.uniform(-1, 1), 10 ** rng.uniform(-12, 12), float(rng.randint(-5, 5))]
        )
    if kind == 'str':
        alphabet = 'ab_-. "\\\n\t\x01é中'
        return ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
    if kind == 'bool':
        return rng.random() < 0.5
    if kind == 'list':
        if rng.random() < 0.3:  # A list of tables.
            return [random_table(rng, depth + 1) for _ in range(rng.randint(1, 3))]
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return random_table(rng, depth + 1)


def random_table(rng: random.Random, depth: int) -> dict[str, Any]:
    keys = ['a', 'b', 'c.d', 'e f', '', 'model', 'lr', 'x"y', '1']
    return {
        rng.choice(keys): random_value(rng, depth) for _ in range(rng.randint(0, 4))
    }


def test_random_free_form_structures_roundtrip(tmp_path: Path) -> None:
    rng = random.Random(0)
    for _ in range(300):
        space = random_table(rng, 0)
        configs = [random_table(rng, 1) for _ in range(rng.randint(0, 3))]
        config = TabPackConfig(space=space, n_models=len(configs), configs=configs)
        assert_roundtrip(config, tmp_path)


# --- configs/churn/*.toml ---------------------------------------------------------

EXPECTED_FILES = {
    'mlp.toml': MLPMethodConfig(),
    'homogeneous.toml': HomogeneousEnsembleConfig(),
    'tabpack.toml': TabPackConfig(),
    'tabpack-conservative.toml': ConservativeEvalConfig(
        source_run='runs/churn/tabpack/seed-0', n_seeds=5
    ),
}


def test_config_directory_has_exactly_the_expected_files() -> None:
    assert {p.name for p in CONFIGS_DIR.glob('*.toml')} == set(EXPECTED_FILES)


@pytest.mark.parametrize('name', sorted(EXPECTED_FILES))
def test_config_files_load_to_the_defaults(name: str, tmp_path: Path) -> None:
    path = CONFIGS_DIR / name
    expected = EXPECTED_FILES[name]
    loaded = load_config(path)
    assert loaded == expected
    assert strict(loaded) == strict(expected)
    # Every knob is written out explicitly (and documented), with the same types a
    # dump of the expected config has: the file is a complete, readable reference.
    with path.open('rb') as file:
        raw = tomllib.load(file)
    dumped = tmp_path / 'expected.toml'
    dump_config(expected, dumped)
    with dumped.open('rb') as file:
        assert strict(raw) == strict(tomllib.load(file))


@pytest.mark.parametrize('name', sorted(EXPECTED_FILES))
def test_config_files_point_to_the_experiment_doc(name: str) -> None:
    text = (CONFIGS_DIR / name).read_text(encoding='utf-8')
    assert 'docs/EXPERIMENT.md' in text
    assert f'method = "{EXPECTED_FILES[name].method}"' in text
