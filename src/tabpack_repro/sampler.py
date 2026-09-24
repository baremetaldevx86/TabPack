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
"""

from __future__ import annotations

from typing import Any


def sample_config(trial: Any, space: Any, label_parts: list[str] | None = None) -> Any:
    """Sample one config from `space` with an optuna trial."""
    raise NotImplementedError


def sample_configs(space: dict[str, Any], n: int, *, seed: int) -> list[dict[str, Any]]:
    """Sample `n` member configs deterministically (seeded optuna RandomSampler)."""
    raise NotImplementedError
