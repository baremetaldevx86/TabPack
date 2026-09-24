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
    """
    raise NotImplementedError


def load_config(path: str | Path) -> AnyMethodConfig:
    """Load a TOML config file (a28). Uses the stdlib ``tomllib``."""
    raise NotImplementedError


def dump_config(config: AnyMethodConfig, path: str | Path) -> None:
    """Write a config as TOML (a28), such that load_config(path) == config."""
    raise NotImplementedError
