"""Reduced heterogeneous TabPack (a32).

1. member configs: config.configs, or sample_configs(config.space, n_models, seed);
2. ModelPack with per-member n_blocks / dropout (d_block fixed);
3. MuonAdamWPack with per-member lr / weight_decay / muon_lr,
   make_param_groups(muon=True);
4. train_pack with an OnlineGreedyEnsemble (val score, max_ensemble_size, patience);
5. final prediction = the online ensemble; report members (with configs), the
   ensemble ids/steps, and the best single member.

Report extensions (on top of methods/report.py), all JSON-serializable:
* ``member_configs``: the configs of ALL ``n_models`` members, index = member id.
  The online ensemble may select a member that never finished (training stops when
  the ensemble's patience runs out, and unfinished members are dropped), so its
  config is not in ``members``; this list plays the role of the official
  ``experiments.json`` for the conservative protocol.
* ``n_models`` (K) and ``n_finished`` (``len(members)``).
* ``online_ensemble``: the full OnlineGreedyEnsemble.report() (ids, steps, size,
  n_unique, score_val, metrics per stored part).
* ``best_member`` additionally carries its ``config``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabpack_repro.config import TabPackConfig
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.methods.common import (
    PREDICTION_PARTS,
    base_report,
    best_member,
    setup_run,
    write_run,
)
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups
from tabpack_repro.sampler import sample_configs
from tabpack_repro.training.trainer import train_pack

METHOD = 'tabpack'

# The hyperparameters that vary per member, by config section. Every one of them
# must be present in every member config, and nothing else may be (the official
# main() asserts that the transposed configs are fully consumed).
_MEMBER_KEYS: dict[str, tuple[str, ...]] = {
    'model': ('n_blocks', 'dropout'),
    'optimizer': ('lr', 'weight_decay', 'muon_lr'),
}


def _transpose(dicts: list[Any], label: str) -> dict[str, list[Any]]:
    """[{k: v_i}] -> {k: [v_i]}; every dict must have the same keys (official
    project.utils.transpose_list_of_dicts, raising instead of asserting)."""
    for i, d in enumerate(dicts):
        if not isinstance(d, dict):
            raise TypeError(
                f'{label} of member {i} must be a dict, got {type(d).__name__}'
            )
    keys = list(dicts[0])
    for i, d in enumerate(dicts):
        if set(d) != set(keys):
            raise ValueError(
                f'All member configs must have the same keys: {label} of member 0'
                f' has {sorted(keys)}, {label} of member {i} has {sorted(d)}'
            )
    return {key: [d[key] for d in dicts] for key in keys}


def _split_member_configs(
    configs: list[dict[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    """Member configs -> {section: {key: [value of member 0, 1, ...]}}.

    Raises ValueError if a required per-member key is missing, or if any sampled key
    would be left unused.
    """
    if not configs:
        raise ValueError('There must be at least one member config')
    configs_t = _transpose(configs, 'the config')
    result: dict[str, dict[str, list[Any]]] = {}
    missing: list[str] = []
    unused: list[str] = []
    for section, keys in _MEMBER_KEYS.items():
        section_t = (
            _transpose(configs_t.pop(section), f'config[{section!r}]')
            if section in configs_t
            else {}
        )
        missing.extend(f'{section}.{key}' for key in keys if key not in section_t)
        result[section] = {key: section_t.pop(key) for key in keys if key in section_t}
        unused.extend(f'{section}.{key}' for key in section_t)
    # Unknown top-level sections.
    unused.extend(configs_t)
    if missing:
        raise ValueError(
            f'The member configs lack the per-member hyperparameters: {missing}'
        )
    if unused:
        raise ValueError(
            'The following fields of the member configs were not used: '
            + ', '.join(unused)
        )
    return result


def _member_configs(config: TabPackConfig) -> list[dict[str, Any]]:
    """The explicit configs (a deep copy), or n_models configs sampled from space."""
    if isinstance(config.n_models, bool) or config.n_models < 1:
        raise ValueError(f'n_models must be >= 1, got {config.n_models!r}')
    if config.configs is None:
        return sample_configs(config.space, config.n_models, seed=config.seed)
    if len(config.configs) != config.n_models:
        raise ValueError(
            f'n_models={config.n_models} does not match the number of explicit'
            f' member configs ({len(config.configs)})'
        )
    return copy.deepcopy(list(config.configs))


def _check_online_ensemble(config: TabPackConfig) -> None:
    # OnlineGreedyEnsemble implements exactly one official variant.
    oe = config.online_ensemble
    supported = {
        'type': 'greedy',
        'update_type': 'latest',
        'include_current_ensemble_in_pool': True,
    }
    for key, value in supported.items():
        if getattr(oe, key) != value:
            raise ValueError(
                f'online_ensemble.{key}={getattr(oe, key)!r} is not supported'
                f' (only {value!r})'
            )


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # .float() first: numpy has no bfloat16.
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def run(config: TabPackConfig, output_dir: str | Path) -> dict[str, Any]:
    # Validate everything that does not need the data before loading it.
    _check_online_ensemble(config)
    member_configs = _member_configs(config)
    per_member = _split_member_configs(member_configs)
    n_models = len(member_configs)

    ctx = setup_run(config.seed, config.data, config.training)
    dataset = ctx.dataset

    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=dataset.task.n_classes,
        pack_size=n_models,
        d_block=config.d_block,
        n_blocks=per_member['model']['n_blocks'],
        dropout=[float(x) for x in per_member['model']['dropout']],
        activation=config.activation,
    ).to(ctx.device)

    opt = config.optimizer
    optimizer = MuonAdamWPack(
        make_param_groups(model, muon=True),
        **{
            key: [float(x) for x in values]
            for key, values in per_member['optimizer'].items()
        },
        muon_momentum=opt.muon_momentum,
        muon_nesterov=opt.muon_nesterov,
        muon_ns_steps=opt.muon_ns_steps,
        beta1=opt.beta1,
        beta2=opt.beta2,
        eps=opt.eps,
        pack_size=n_models,
        shared_step=opt.shared_step,
    )

    online_ensemble = OnlineGreedyEnsemble(
        score_fn=ctx.score_fns['val'],
        task=dataset.task,
        max_ensemble_size=config.online_ensemble.max_ensemble_size,
        patience=config.online_ensemble.patience,
    )

    training = config.training
    result = train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=training.batch_size,
        patience=training.patience,
        max_epochs=training.max_epochs,
        seed=config.seed,
        eval_batch_size=training.eval_batch_size,
        autocast=ctx.autocast,
        online_ensemble=online_ensemble,
    )

    # >>> Final prediction: the online ensemble.
    if online_ensemble.score is None:
        raise RuntimeError('The online ensemble was never updated during training')
    ensemble_predictions = online_ensemble.predictions()
    missing_parts = [p for p in PREDICTION_PARTS if p not in ensemble_predictions]
    if missing_parts:
        raise RuntimeError(
            f'The online ensemble has no predictions for the parts {missing_parts}'
        )
    predictions = {
        part: _to_numpy(ensemble_predictions[part]) for part in PREDICTION_PARTS
    }
    metrics = {
        part: compute_metrics(ctx.y_true[part], predictions[part], dataset.task)
        for part in PREDICTION_PARTS
    }
    ensemble_ids = online_ensemble.ids
    ensemble = {
        'ids': ensemble_ids.tolist(),
        'steps': online_ensemble.steps.tolist(),
        'size': len(ensemble_ids),
        'n_unique': len(np.unique(ensemble_ids)),
    }
    online_report = (
        result.online_ensemble
        if result.online_ensemble is not None
        else online_ensemble.report(ctx.y_true)
    )

    # >>> Members: the finished ones, each with its config.
    members: list[dict[str, Any]] = []
    for finished in result.members:
        member_id = int(finished['id'])
        member = {
            'id': member_id,
            'best_step': int(finished['best_step']),
            'config': copy.deepcopy(member_configs[member_id]),
            'metrics': finished['metrics'],
        }
        member.update({k: v for k, v in finished.items() if k not in member})
        members.append(member)
    best = best_member(members)
    if best is not None:
        best['config'] = copy.deepcopy(member_configs[best['id']])

    common = base_report(METHOD, config, config.seed, ctx)
    report: dict[str, Any] = {
        'schema_version': common['schema_version'],
        'method': common['method'],
        'dataset': common['dataset'],
        'seed': common['seed'],
        'config': common['config'],
        'metrics': metrics,
        'members': members,
        'ensemble': ensemble,
        'best_member': best,
        'n_epochs': int(result.n_epochs),
        'n_steps': int(result.n_steps),
        'time_sec': float(result.time_sec),
        'env': common['env'],
        'history': result.history,
        'n_models': n_models,
        'n_finished': len(members),
        'member_configs': copy.deepcopy(member_configs),
        'online_ensemble': online_report,
    }
    write_run(output_dir, report, predictions, config)
    return report
