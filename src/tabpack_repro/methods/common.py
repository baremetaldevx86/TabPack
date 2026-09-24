"""Shared plumbing for all methods (a55): setup, reports and artifacts.

Keeps methods/*.py focused on the method itself.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabpack_repro.config import AnyMethodConfig, DataConfig, TrainingConfig
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.metrics import ScoreFn
from tabpack_repro.types import PartKey


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
    raise NotImplementedError


def base_report(
    method: str, config: AnyMethodConfig, seed: int, ctx: RunContext
) -> dict[str, Any]:
    """Fill the common fields of the run report (methods/report.py): schema_version,
    method, dataset (basename of config.data.path), seed, config, env."""
    raise NotImplementedError


def best_member(members: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The member with the highest val score (first on ties) as {"id", "metrics"}."""
    raise NotImplementedError


def write_run(
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: dict[str, np.ndarray],
    config: AnyMethodConfig,
) -> None:
    """Write report.json (utils.io.dump_json), predictions.npz (keys val/test) and
    config.toml (config.dump_config) into output_dir (created if needed)."""
    raise NotImplementedError
