"""Experiment configuration (frozen contract).

The dataclasses below are the single source of truth for every tunable knob of the
three compared methods. Defaults reproduce the decisions recorded in
``docs/EXPERIMENT.md``:

* Ordinary MLP and homogeneous ensemble: fixed, untuned defaults, AdamW.
* Reduced TabPack: 32 sampled MLPs (official Churn uses 64), greedy online
  ensemble of at most 16 members (official: 32), Muon+AdamW, official search space.

``load_config`` / ``dump_config`` / ``config_to_dict`` / ``config_from_dict`` are
implemented by task a28 (TOML files live in ``configs/churn/``).
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import math
import numbers
import re
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MethodName = Literal['mlp', 'homogeneous', 'tabpack', 'tabpack-conservative']


@dataclass(kw_only=True)
class DataConfig:
    """Mirrors the official Churn data config (``lib.config.make_data_config``)."""

    # Relative paths are resolved against ``tabpack_repro.data.get_data_dir()``.
    path: str = 'churn'
    # Seed of the noisy-quantile transform. The official code always uses 0
    # (lib.data.build_dataset default), independent of the run seed; so do we.
    seed: int = 0
    num_policy: str | None = 'noisy-quantile'
    extract_bin_from_num: bool = True
    bin_policy: str | None = 'convert-to-cat'
    cat_policy: str | None = 'ordinal'


@dataclass(kw_only=True)
class TrainingConfig:
    batch_size: int = 256
    # Early stopping: a member stops after `patience + 1` consecutive epochs
    # without a strict improvement of its validation score (official semantics:
    # stop when n_consecutive_bad_updates > patience).
    patience: int = 16
    # -1 means "no epoch limit" (only early stopping), as in the official config.
    max_epochs: int = -1
    eval_batch_size: int = 32768
    # 'bfloat16' | 'float16' | None. Autocast is only enabled on CUDA.
    amp_dtype: str | None = 'bfloat16'
    # 'auto' picks CUDA when available.
    device: str = 'auto'


@dataclass(kw_only=True)
class MLPModelConfig:
    n_blocks: int = 3
    d_block: int = 384
    dropout: float = 0.1
    activation: str = 'ReLU'


@dataclass(kw_only=True)
class AdamWConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(kw_only=True)
class MuonAdamWConfig:
    """Fixed (non-sampled) Muon+AdamW settings; lr/weight_decay/muon_lr are sampled."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    shared_step: bool = True


@dataclass(kw_only=True)
class OnlineEnsembleConfig:
    type: Literal['greedy'] = 'greedy'
    update_type: Literal['latest'] = 'latest'
    include_current_ensemble_in_pool: bool = True
    patience: int = 32
    max_ensemble_size: int = 16


def _official_space() -> dict[str, Any]:
    # Identical to the official Churn search space.
    return {
        'model': {
            'n_blocks': ['_tune_', 'int', 1, 4],
            'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
        },
        'optimizer': {
            'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
            'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
            'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
        },
    }


@dataclass(kw_only=True)
class MLPMethodConfig:
    method: Literal['mlp'] = 'mlp'
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: MLPModelConfig = field(default_factory=MLPModelConfig)
    optimizer: AdamWConfig = field(default_factory=AdamWConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(kw_only=True)
class HomogeneousEnsembleConfig:
    method: Literal['homogeneous'] = 'homogeneous'
    seed: int = 0
    # K identical-hyperparameter members = TabPack's max ensemble size.
    n_models: int = 16
    data: DataConfig = field(default_factory=DataConfig)
    model: MLPModelConfig = field(default_factory=MLPModelConfig)
    optimizer: AdamWConfig = field(default_factory=AdamWConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(kw_only=True)
class TabPackConfig:
    method: Literal['tabpack'] = 'tabpack'
    seed: int = 0
    n_models: int = 32
    data: DataConfig = field(default_factory=DataConfig)
    d_block: int = 384
    activation: str = 'ReLU'
    optimizer: MuonAdamWConfig = field(default_factory=MuonAdamWConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    online_ensemble: OnlineEnsembleConfig = field(default_factory=OnlineEnsembleConfig)
    # Search space in the official "_tune_" format (see tabpack_repro.sampler).
    space: dict[str, Any] = field(default_factory=_official_space)
    # Explicit per-member configs ({'model': {...}, 'optimizer': {...}}).
    # When set, `space` is ignored and n_models must equal len(configs).
    # Used by the conservative evaluation protocol.
    configs: list[dict[str, Any]] | None = None


@dataclass(kw_only=True)
class ConservativeEvalConfig:
    """The paper's "conservative" protocol for TabPack.

    Take the ensemble selected by a finished TabPack main run, and retrain only the
    selected member configs from scratch with `n_seeds` fresh seeds (0..n_seeds-1);
    the reported metric is the online-ensemble test score of each retraining run.
    """

    method: Literal['tabpack-conservative'] = 'tabpack-conservative'
    # Directory of the finished TabPack main run (contains report.json).
    source_run: str = ''
    n_seeds: int = 5


AnyMethodConfig = (
    MLPMethodConfig | HomogeneousEnsembleConfig | TabPackConfig | ConservativeEvalConfig
)

CONFIG_CLASSES: dict[str, type] = {
    'mlp': MLPMethodConfig,
    'homogeneous': HomogeneousEnsembleConfig,
    'tabpack': TabPackConfig,
    'tabpack-conservative': ConservativeEvalConfig,
}


def config_to_dict(config: AnyMethodConfig) -> dict[str, Any]:
    """Convert a config to a JSON/TOML-serializable nested dict (a28)."""
    return dataclasses.asdict(config)


def config_from_dict(data: dict[str, Any]) -> AnyMethodConfig:
    """Build a config from a nested dict, dispatching on data['method'] (a28).

    Must reject unknown keys at every nesting level with a ValueError that names
    the offending key, and must fill omitted keys with the dataclass defaults.

    Details:

    * ``data['method']`` is required and selects the class in ``CONFIG_CLASSES``.
    * Nested dataclass fields are built recursively from nested mappings, guided by
      the type hints. Omitted keys, at any level, take the dataclass defaults.
    * An unknown key raises ``ValueError`` naming its full key path, e.g.
      ``'training.patiance'`` or ``'online_ensemble.max_size'``.
    * Values are type-checked: ``int`` fields reject bools and floats, ``float``
      fields also accept ints (converted to float), ``bool``, ``str`` and
      ``Literal`` fields accept only matching values, arrays may be lists or tuples.
      A wrong type raises an exception that is both a ``TypeError`` and a
      ``ValueError``; the message names the key path.
    * ``None`` is accepted by optional fields. In optional string fields
      (``str | None``: ``data.num_policy``, ``data.bin_policy``,
      ``data.cat_policy``, ``training.amp_dtype``) the string ``'none'`` also means
      None, because TOML has no null (see ``dump_config``).
    * Free-form values (``space``, the entries of ``configs``) are deep-copied,
      tuples become lists and mapping keys must be strings. A table given for
      ``space`` replaces the default search space; it is not merged into it.
    """
    if not isinstance(data, Mapping):
        raise _type_error('', 'a table (mapping)', data)
    if 'method' not in data:
        raise ValueError(
            f"config has no 'method' key; expected one of {_choices(CONFIG_CLASSES)}"
        )
    method = data['method']
    if not isinstance(method, str) or method not in CONFIG_CLASSES:
        raise ValueError(
            f"config key 'method': unknown method {method!r}; expected one of "
            f'{_choices(CONFIG_CLASSES)}'
        )
    return _build_dataclass(CONFIG_CLASSES[method], data, '')


def load_config(path: str | Path) -> AnyMethodConfig:
    """Load a TOML config file (a28). Uses the stdlib ``tomllib``.

    Equivalent to ``config_from_dict(tomllib.load(file))``; see ``config_from_dict``
    for the validation rules and ``dump_config`` for how None is written. Errors
    (``tomllib.TOMLDecodeError`` or the ones of ``config_from_dict``) carry a note
    with the file path.
    """
    path = Path(path)
    try:
        with path.open('rb') as file:
            data = tomllib.load(file)
        return config_from_dict(data)
    except (TypeError, ValueError) as error:
        error.add_note(f'while loading the config file {path}')
        raise


def dump_config(config: AnyMethodConfig, path: str | Path) -> None:
    """Write a config as TOML (a28), such that load_config(path) == config.

    The stdlib has no TOML writer, so this uses a small built-in TOML 1.0 writer.
    Fields are written in dataclass order (plain keys first, then the sub-tables),
    parent directories are created and the file is UTF-8.

    TOML has no null, so None is written as follows:

    * a field whose default is None (``TabPackConfig.configs``) is omitted when it
      is None. An empty list is written as ``configs = []`` and a non-empty one as
      ``[[configs]]`` tables, so None, ``[]`` and a list all round-trip;
    * an optional string field whose default is not None (``data.num_policy``,
      ``data.bin_policy``, ``data.cat_policy``, ``training.amp_dtype``) is written
      as the string ``"none"``. That string is therefore reserved in optional
      string fields: a config holding the literal ``'none'`` there raises
      ``ValueError`` instead of being written ambiguously;
    * None anywhere else (for example inside ``space`` or a member config) cannot be
      represented and raises ``ValueError``.

    Free-form values (``space``, ``configs``) may hold bools, ints, floats, strings,
    lists or tuples (mixed types allowed) and mappings with string keys; other types
    raise ``TypeError``. Tuples are written as arrays and load back as lists. Ints
    stay ints and floats stay floats: floats are written with ``repr`` (the shortest
    string that parses back to the same float), e.g. ``0.0001``, ``0.005``,
    ``1e-08``, and ``nan``/``inf``/``-inf``.
    """
    if not isinstance(config, tuple(CONFIG_CLASSES.values())):
        raise TypeError(
            f'expected one of {_choices(CONFIG_CLASSES.values())}, '
            f'got {type(config).__name__}'
        )
    if CONFIG_CLASSES.get(config.method) is not type(config):
        raise ValueError(
            f'{type(config).__name__}.method is {config.method!r}, which would load '
            f'back as a different config class'
        )
    text = _DUMP_HEADER + '\n' + _toml_dumps(_to_toml_data(config, ''))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


# ---------------------------------------------------------------------------------
# Private helpers of config_from_dict / load_config / dump_config (a28).
# ---------------------------------------------------------------------------------

# TOML has no null: in optional string fields (`str | None`, e.g. amp_dtype) this
# string stands for None. See dump_config.
_NONE_STRING = 'none'
_NONE_TYPE = type(None)
_BARE_KEY_RE = re.compile(r'[A-Za-z0-9_-]+')
_STRING_ESCAPES = {
    '"': '\\"',
    '\\': '\\\\',
    '\b': '\\b',
    '\t': '\\t',
    '\n': '\\n',
    '\f': '\\f',
    '\r': '\\r',
}
_DUMP_HEADER = (
    '# Written by tabpack_repro.config.dump_config; read it with load_config.\n'
    '# TOML has no null: a key whose default is None is omitted when it is None,\n'
    '# and "none" means None in optional string settings (e.g. training.amp_dtype).\n'
)


class _ConfigTypeError(TypeError, ValueError):
    """A config value has the wrong type (catchable as TypeError or ValueError)."""


def _choices(items: Any) -> str:
    names = [item if isinstance(item, str) else item.__name__ for item in items]
    return ', '.join(repr(name) if isinstance(name, str) else name for name in names)


def _join(path: str, key: object) -> str:
    return f'{path}.{key}' if path else str(key)


def _type_error(path: str, expected: str, value: Any) -> _ConfigTypeError:
    where = f'config key {path!r}' if path else 'config'
    return _ConfigTypeError(
        f'{where}: expected {expected}, got {value!r} ({type(value).__name__})'
    )


@functools.cache
def _type_hints(cls: type) -> dict[str, Any]:
    # The module uses `from __future__ import annotations`: resolve the strings.
    return typing.get_type_hints(cls)


def _is_optional_str(tp: Any) -> bool:
    if typing.get_origin(tp) not in (typing.Union, types.UnionType):
        return False
    args = typing.get_args(tp)
    return str in args and _NONE_TYPE in args


def _type_name(tp: Any) -> str:
    return tp.__name__ if isinstance(tp, type) else str(tp).replace('typing.', '')


# --- dict -> config ---------------------------------------------------------------


def _build_dataclass(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise _type_error(path, f'a table (mapping) for {cls.__name__}', data)
    names = [f.name for f in dataclasses.fields(cls) if f.init]
    for key in data:
        if key not in names:
            raise ValueError(_unknown_key_message(cls, path, key, names))
    hints = _type_hints(cls)
    kwargs = {
        key: _convert(value, hints[key], _join(path, key))
        for key, value in data.items()
    }
    return cls(**kwargs)


def _unknown_key_message(cls: type, path: str, key: Any, valid: list[str]) -> str:
    message = f'unknown config key {_join(path, key)!r}'
    if isinstance(key, str):
        close = difflib.get_close_matches(key, valid, n=1)
        if close:
            message += f' (did you mean {_join(path, close[0])!r}?)'
    where = f'in {path!r}' if path else f'at the top level of {cls.__name__}'
    return f'{message}; valid keys {where}: {", ".join(valid)}'


def _convert(value: Any, tp: Any, path: str) -> Any:
    """Check `value` against the type hint `tp` and return a fresh, converted copy."""
    if tp is Any:
        return _plain(value, path)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, value, path)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        return _convert_union(value, tp, path)
    if origin is Literal:
        for choice in args:
            if type(value) is type(choice) and value == choice:
                return choice
        raise ValueError(
            f'config key {path!r}: expected one of {_choices(args)}, got {value!r}'
        )
    if origin is list or tp is list:
        if not isinstance(value, list | tuple):
            raise _type_error(path, 'an array', value)
        item_tp = args[0] if args else Any
        return [_convert(v, item_tp, f'{path}[{i}]') for i, v in enumerate(value)]
    if origin is tuple or tp is tuple:
        return _convert_tuple(value, args, path)
    if origin is dict or tp is dict:
        if not isinstance(value, Mapping):
            raise _type_error(path, 'a table (mapping)', value)
        item_tp = args[1] if args else Any
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _type_error(path, 'string keys', key)
            result[key] = _convert(item, item_tp, _join(path, key))
        return result
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise _type_error(path, 'a boolean', value)
    if tp is int:
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        raise _type_error(path, 'an integer', value)
    if tp is float:
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return float(value)
        raise _type_error(path, 'a number', value)
    if tp is str:
        if isinstance(value, str):
            return str(value)
        raise _type_error(path, 'a string', value)
    if isinstance(tp, type) and not isinstance(value, tp):
        raise _type_error(path, _type_name(tp), value)
    return value


def _convert_union(value: Any, tp: Any, path: str) -> Any:
    args = typing.get_args(tp)
    options = [arg for arg in args if arg is not _NONE_TYPE]
    if value is None:
        if _NONE_TYPE in args:
            return None
        raise _type_error(path, _type_name(tp), value)
    if _is_optional_str(tp) and isinstance(value, str) and value == _NONE_STRING:
        return None
    errors: list[Exception] = []
    for option in options:
        try:
            return _convert(value, option, path)
        except (TypeError, ValueError) as error:
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    raise _type_error(path, _type_name(tp), value)


def _convert_tuple(value: Any, args: tuple[Any, ...], path: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise _type_error(path, 'an array', value)
    if len(args) == 2 and args[1] is Ellipsis:
        item_tps = [args[0]] * len(value)
    elif args:
        if len(args) != len(value):
            raise _type_error(path, f'an array of length {len(args)}', value)
        item_tps = list(args)
    else:
        item_tps = [Any] * len(value)
    return tuple(
        _convert(v, t, f'{path}[{i}]')
        for i, (v, t) in enumerate(zip(value, item_tps, strict=True))
    )


def _plain(value: Any, path: str) -> Any:
    """Deep copy of a free-form value: mappings -> dicts (string keys), tuples ->
    lists; other values are kept as they are."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _type_error(path, 'string keys', key)
            result[key] = _plain(item, _join(path, key))
        return result
    if isinstance(value, list | tuple):
        return [_plain(v, f'{path}[{i}]') for i, v in enumerate(value)]
    return value


# --- config -> TOML ---------------------------------------------------------------


def _to_toml_data(config: Any, path: str) -> dict[str, Any]:
    """A config dataclass as a nested dict without None (see dump_config)."""
    hints = _type_hints(type(config))
    result: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        if not f.init:
            continue
        value = getattr(config, f.name)
        key_path = _join(path, f.name)
        optional_str = _is_optional_str(hints[f.name])
        if value is None:
            if f.default is None:
                continue  # Omitted: loads back as the default, None.
            if not optional_str:
                raise ValueError(
                    f'config key {key_path!r}: None cannot be written to TOML '
                    f'(TOML has no null and the default is not None)'
                )
            result[f.name] = _NONE_STRING
        elif optional_str and isinstance(value, str) and value == _NONE_STRING:
            raise ValueError(
                f'config key {key_path!r}: the string {_NONE_STRING!r} is reserved '
                f'for None in TOML config files'
            )
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            result[f.name] = _to_toml_data(value, key_path)
        else:
            result[f.name] = value
    return result


def _toml_dumps(table: Mapping[str, Any]) -> str:
    """Serialize a nested mapping to TOML 1.0 (the subset configs need)."""
    lines: list[str] = []
    _write_table(lines, (), table)
    return '\n'.join(lines).lstrip('\n') + '\n'


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) > 0
        and all(isinstance(item, Mapping) for item in value)
    )


def _has_plain_keys(table: Mapping[str, Any]) -> bool:
    return any(
        not isinstance(v, Mapping) and not _is_table_array(v) for v in table.values()
    )


def _write_table(lines: list[str], path: tuple[str, ...], table: Mapping) -> None:
    """Append the body of `table` (the caller wrote its header, if any): plain keys
    first, then sub-tables as [a.b] and lists of tables as [[a.b]]."""
    tables = []
    arrays = []
    for key, value in table.items():
        if not isinstance(key, str):
            raise TypeError(
                f'config key {".".join(path)!r}: TOML keys must be strings, got {key!r}'
            )
        if isinstance(value, Mapping):
            tables.append((key, value))
        elif _is_table_array(value):
            arrays.append((key, value))
        else:
            where = '.'.join((*path, key))
            lines.append(f'{_format_key(key)} = {_format_value(value, where)}')
    for key, value in tables:
        sub = (*path, key)
        # A non-empty table without plain keys is implied by its sub-tables' headers.
        if not value or _has_plain_keys(value):
            lines.extend(['', f'[{_format_path(sub)}]'])
        _write_table(lines, sub, value)
    for key, value in arrays:
        sub = (*path, key)
        for item in value:
            lines.extend(['', f'[[{_format_path(sub)}]]'])
            _write_table(lines, sub, item)


def _format_path(path: tuple[str, ...]) -> str:
    return '.'.join(_format_key(key) for key in path)


def _format_key(key: str) -> str:
    return key if _BARE_KEY_RE.fullmatch(key) else _format_string(key)


def _format_value(value: Any, where: str) -> str:
    """An inline TOML value; `where` is the key path used in error messages."""
    if value is None:
        raise ValueError(f'config key {where!r}: TOML has no null, cannot write None')
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return _format_float(float(value))
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f'config key {where!r}: TOML keys must be strings, got {key!r}'
                )
            item_where = f'{where}.{key}'
            items.append(f'{_format_key(key)} = {_format_value(item, item_where)}')
        return '{ ' + ', '.join(items) + ' }' if items else '{}'
    if isinstance(value, list | tuple):
        items = [_format_value(v, f'{where}[{i}]') for i, v in enumerate(value)]
        return '[' + ', '.join(items) + ']'
    raise TypeError(
        f'config key {where!r}: cannot write a {type(value).__name__} to TOML'
    )


def _format_float(value: float) -> str:
    if math.isnan(value):
        return 'nan'
    if math.isinf(value):
        return 'inf' if value > 0 else '-inf'
    # repr is the shortest string that parses back to the same float, and it is
    # always valid TOML (it contains a '.' or an exponent), e.g. '0.0001', '1e-08'.
    return repr(value)


def _format_string(text: str) -> str:
    chars = []
    for char in text:
        if char in _STRING_ESCAPES:
            chars.append(_STRING_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            chars.append(f'\\u{ord(char):04X}')
        else:
            chars.append(char)
    return '"' + ''.join(chars) + '"'
