"""Hyperparameter sampling for TabPack members (a24).

Space format (official lib/tools/tune.py::_sample_config):
* constants are returned as is; dicts/lists are sampled recursively;
* ``['_tune_', 'int', low, high]`` -> trial.suggest_int(label, low, high)
* ``['_tune_', 'uniform', low, high]`` -> trial.suggest_float(label, low, high)
* ``['_tune_', 'loguniform', low, high]`` -> suggest_float(..., log=True)
* ``['_tune_', '?<dist>', default, *args]`` -> if
  trial.suggest_categorical(f'?{label}', [False, True]) then sample <dist>(*args)
  else `default`.
Labels are dot-joined key paths, e.g. 'model.n_blocks', 'optimizer.lr'.

To reproduce the official member configs exactly, use an
``optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed),
direction='maximize')`` and call ``study.ask()`` once per member, in order.

Also supported, as in the official code:
* a third numeric argument is the step: ``['_tune_', 'int', 1, 9, 2]``
  -> suggest_int(label, 1, 9, step=2) (same for 'uniform'/'loguniform');
* ``['_tune_', 'categorical', choices]`` -> trial.suggest_categorical(label, choices);
* ``['_tune_', '$list', size, <dist>, *args]`` -> a list of `size` values, the i-th
  sampled from <dist>(*args) under the label ``f'{label}.{i}'``.
"""

from __future__ import annotations

import copy
from typing import Any

import optuna

_TUNE = '_tune_'
_CONSTANT_TYPES = (bool, int, float, str, bytes, type(None))
_NUMERIC_DISTRIBUTIONS = ('int', 'uniform', 'loguniform')


def _suggest(trial: Any, distribution: str, label: str, args: list[Any]) -> Any:
    """Draw one value of a built-in distribution (official ``_sample_value``)."""
    kwargs: dict[str, Any] = {}
    if distribution in _NUMERIC_DISTRIBUTIONS and len(args) == 3:
        args, kwargs['step'] = args[:2], args[2]
    if distribution == 'int':
        return trial.suggest_int(label, *args, **kwargs)
    if distribution == 'uniform':
        return trial.suggest_float(label, *args, **kwargs)
    if distribution == 'loguniform':
        return trial.suggest_float(label, *args, log=True, **kwargs)
    if distribution == 'categorical':
        return trial.suggest_categorical(label, *args)
    raise ValueError(
        f'Unknown distribution {distribution!r} for the hyperparameter {label!r};'
        " expected one of 'int', 'uniform', 'loguniform', 'categorical'"
        " (optionally prefixed with '?'), or '$list'"
    )


def _sample_tuned(trial: Any, spec: list[Any], label: str) -> Any:
    """Sample a ``['_tune_', distribution, *args]`` leaf."""
    if len(spec) < 2 or not isinstance(spec[1], str):
        raise ValueError(
            f'Malformed search space entry {spec!r} for {label!r}:'
            " expected ['_tune_', <distribution>, *args]"
        )
    distribution, args = spec[1], list(spec[2:])

    if distribution.startswith('?'):
        # ['_tune_', '?dist', default, *args]: a coin flip decides whether the value
        # is sampled or equals the default. The flip is recorded under '?<label>'.
        if not args:
            raise ValueError(f'{spec!r} for {label!r} lacks the default value')
        default, dist_args = args[0], args[1:]
        if trial.suggest_categorical(f'?{label}', [False, True]):
            return _suggest(trial, distribution.lstrip('?'), label, dist_args)
        return copy.deepcopy(default)

    if distribution == '$list':
        # ['_tune_', '$list', size, dist, *args]: `size` independent values.
        if len(args) < 2:
            raise ValueError(
                f'{spec!r} for {label!r}:'
                " expected ['_tune_', '$list', size, dist, *args]"
            )
        size, item_distribution, *item_args = args
        return [
            _suggest(trial, item_distribution, f'{label}.{i}', item_args)
            for i in range(size)
        ]

    return _suggest(trial, distribution, label, args)


def sample_config(trial: Any, space: Any, label_parts: list[str] | None = None) -> Any:
    """Sample one config from `space` with an optuna trial."""
    parts = [] if label_parts is None else list(label_parts)

    if isinstance(space, _CONSTANT_TYPES):
        return space

    if isinstance(space, list):
        if space and isinstance(space[0], str) and space[0] == _TUNE:
            return _sample_tuned(trial, space, '.'.join(map(str, parts)))
        # A plain list: every item is a subspace labelled by its index.
        return [
            sample_config(trial, subspace, [*parts, str(i)])
            for i, subspace in enumerate(space)
        ]

    if isinstance(space, dict):
        if _TUNE in space:
            # The official code reserves dict-level '_tune_' for custom samplers and
            # ships none; reject instead of silently treating it as a constant.
            raise ValueError(
                f'Unknown custom distribution {space[_TUNE]!r} at'
                f' {".".join(map(str, parts)) or "<root>"!r}'
            )
        return {
            key: sample_config(trial, subspace, [*parts, str(key)])
            for key, subspace in space.items()
        }

    raise TypeError(
        f'Unsupported search space value of type {type(space).__name__} at'
        f' {".".join(map(str, parts)) or "<root>"!r}: {space!r}'
    )


def sample_configs(space: dict[str, Any], n: int, *, seed: int) -> list[dict[str, Any]]:
    """Sample `n` member configs deterministically (seeded optuna RandomSampler)."""
    if not isinstance(space, dict):
        raise TypeError(f'space must be a dict, got {type(space).__name__}')
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f'n must be a non-negative int, got {n!r}')

    # optuna logs "A new study created in memory ..." at INFO level; silence it
    # for the duration of this call only.
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=seed), direction='maximize'
        )
        return [sample_config(study.ask(), space) for _ in range(n)]
    finally:
        optuna.logging.set_verbosity(verbosity)
