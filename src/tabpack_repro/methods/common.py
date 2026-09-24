"""Shared plumbing for all methods (a55): setup, reports and artifacts.

Keeps methods/*.py focused on the method itself.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabpack_repro.config import (
    AnyMethodConfig,
    DataConfig,
    TrainingConfig,
    config_to_dict,
    dump_config,
)
from tabpack_repro.data.pipeline import PreparedDataset, build_dataset
from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.metrics import ScoreFn, make_score_fn
from tabpack_repro.types import PARTS, PartKey
from tabpack_repro.utils.device import describe_device, make_autocast, resolve_device
from tabpack_repro.utils.io import dump_json, git_commit
from tabpack_repro.utils.seed import seed_everything

REPORT_FILENAME = 'report.json'
PREDICTIONS_FILENAME = 'predictions.npz'
CONFIG_FILENAME = 'config.toml'
# The parts stored in predictions.npz (methods/report.py).
PREDICTION_PARTS: tuple[PartKey, ...] = ('val', 'test')


@dataclass(kw_only=True)
class RunContext:
    device: torch.device
    autocast: contextlib.AbstractContextManager | None
    # Already moved to `device`.
    dataset: PreparedDataset
    # part -> score function over (M, N[, C]) predictions on `device`.
    score_fns: dict[PartKey, ScoreFn]
    # part -> numpy labels (for compute_metrics).
    y_true: dict[PartKey, np.ndarray]
    env: dict[str, Any]


def setup_run(seed: int, data: DataConfig, training: TrainingConfig) -> RunContext:
    """seed_everything(seed); resolve the device; build the dataset; move it to the
    device; make autocast, score functions for train/val/test and env info
    (utils.device.describe_device + utils.io.git_commit)."""
    # Seed first, so that everything a method does after setup_run is reproducible.
    # (The data transform has its own seed, data.seed, and does not use the global RNG.)
    seed_everything(seed)
    device = resolve_device(training.device)
    dataset = build_dataset(data).to(device)
    autocast = make_autocast(training.amp_dtype, device)
    # The labels are already on `device`, so scoring never copies them per call.
    score_fns = {part: make_score_fn(dataset.y[part], dataset.task) for part in PARTS}
    # An owned host copy: on CPU, .numpy() would alias the dataset's label tensors.
    y_true = {part: dataset.y[part].detach().cpu().numpy().copy() for part in PARTS}
    env = {**describe_device(device), 'git_commit': git_commit()}
    return RunContext(
        device=device,
        autocast=autocast,
        dataset=dataset,
        score_fns=score_fns,
        y_true=y_true,
        env=env,
    )


def base_report(
    method: str, config: AnyMethodConfig, seed: int, ctx: RunContext
) -> dict[str, Any]:
    """Fill the common fields of the run report (methods/report.py): schema_version,
    method, dataset (basename of config.data.path), seed, config, env."""
    data = getattr(config, 'data', None)
    # ConservativeEvalConfig has no data section; its per-seed runs pass the
    # TabPackConfig they train, so this fallback only guards against misuse.
    dataset = None if data is None else Path(data.path).name
    return {
        'schema_version': RUN_REPORT_SCHEMA_VERSION,
        'method': method,
        'dataset': dataset,
        'seed': int(seed),
        'config': config_to_dict(config),
        'env': dict(ctx.env),
    }


def _val_score(member: dict[str, Any]) -> float:
    try:
        return float(member['metrics']['val']['score'])
    except (KeyError, TypeError) as err:
        raise ValueError(
            f'member {member.get("id")!r} has no metrics["val"]["score"]'
        ) from err


def best_member(members: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The member with the highest val score (first on ties) as {"id", "metrics"}."""
    best: dict[str, Any] | None = None
    best_score = -math.inf
    for member in members:
        score = _val_score(member)
        # NaN (e.g. an undefined ROC-AUC) never wins; strict ">" keeps the first tie.
        if not math.isnan(score) and (best is None or score > best_score):
            best, best_score = member, score
    if best is None:
        return None
    return {'id': int(best['id']), 'metrics': copy.deepcopy(best['metrics'])}


def _to_float32_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        # .float() first: numpy has no bfloat16.
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        # A file object (not a name): np.savez would append '.npz' to a name.
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_run(
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: dict[str, np.ndarray],
    config: AnyMethodConfig,
) -> None:
    """Write report.json (utils.io.dump_json), predictions.npz (keys val/test) and
    config.toml (config.dump_config) into output_dir (created if needed)."""
    keys = set(predictions)
    if keys != set(PREDICTION_PARTS):
        raise ValueError(
            f'predictions must have exactly the keys {list(PREDICTION_PARTS)}, '
            f'got {sorted(keys)}'
        )
    # Convert before writing anything, so a bad array does not leave a partial run.
    arrays = {part: _to_float32_numpy(predictions[part]) for part in PREDICTION_PARTS}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / REPORT_FILENAME, report)
    _save_npz_atomic(output_dir / PREDICTIONS_FILENAME, arrays)
    dump_config(config, output_dir / CONFIG_FILENAME)
