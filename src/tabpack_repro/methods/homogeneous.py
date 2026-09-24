"""Homogeneous MLP ensemble (a31): K identical-hyperparameter MLPs trained as a pack.

Members differ only by initialization and batch order. AdamWPack with scalar
hyperparameters, make_param_groups(muon=False). Every member early-stops on its own;
the final prediction is the UNIFORM average of all members' best-checkpoint
probabilities (no selection). Also report each member's individual metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from tabpack_repro.config import HomogeneousEnsembleConfig
from tabpack_repro.ensembles.aggregate import average_predictions
from tabpack_repro.methods.common import (
    PREDICTION_PARTS,
    base_report,
    best_member,
    setup_run,
    write_run,
)
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups
from tabpack_repro.training.trainer import PackTrainResult, train_pack

logger = logging.getLogger(__name__)

METHOD = 'homogeneous'


def run(config: HomogeneousEnsembleConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train the ensemble, write report.json / predictions.npz / config.toml into
    `output_dir` (methods/report.py) and return the report.

    * ``members``: every member (all of them finish, there is no online ensemble),
      sorted by id, with ``config`` None (the member hyperparameters are the
      top-level ``config.model`` / ``config.optimizer``).
    * ``metrics`` / ``predictions.npz``: the uniform average of the members'
      best-checkpoint val/test probabilities.
    * ``ensemble``: all members (``size == n_unique == n_models``), in id order.
    """
    n_models = config.n_models
    if isinstance(n_models, bool) or not isinstance(n_models, int) or n_models < 1:
        raise ValueError(f'n_models must be a positive integer, got {n_models!r}')

    training = config.training
    # Seeds the global RNG (initialization, dropout) with config.seed.
    ctx = setup_run(config.seed, config.data, training)
    dataset = ctx.dataset
    task = dataset.task

    # Built on CPU and then moved: the initialization does not depend on the device.
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=task.n_classes,
        pack_size=n_models,
        d_block=config.model.d_block,
        n_blocks=config.model.n_blocks,
        dropout=config.model.dropout,
        activation=config.model.activation,
    ).to(ctx.device)
    optimizer = AdamWPack(
        make_param_groups(model, muon=False),
        # Python floats: the same value for every member.
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
        beta1=float(config.optimizer.beta1),
        beta2=float(config.optimizer.beta2),
        eps=float(config.optimizer.eps),
        pack_size=n_models,
    )

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
        online_ensemble=None,
    )
    ids, best_steps, member_predictions, members = _sorted_members(result, n_models)

    # >>> The final prediction: the uniform average over ALL members (no selection).
    predictions = {
        part: np.asarray(average_predictions(member_predictions[part]))
        for part in PREDICTION_PARTS
    }
    metrics = {
        part: compute_metrics(ctx.y_true[part], predictions[part], task)
        for part in PREDICTION_PARTS
    }
    logger.info(
        'homogeneous (K=%d, seed=%d): val %.4f, test %.4f, %d epochs, %.1f s',
        n_models,
        config.seed,
        metrics['val']['score'],
        metrics['test']['score'],
        result.n_epochs,
        result.time_sec,
    )

    report = base_report(METHOD, config, config.seed, ctx)
    report.update(
        {
            'metrics': metrics,
            'members': members,
            'ensemble': {
                'ids': [int(i) for i in ids],
                'steps': [int(s) for s in best_steps],
                'size': len(ids),
                'n_unique': len(set(ids.tolist())),
            },
            'best_member': best_member(members),
            'n_epochs': int(result.n_epochs),
            'n_steps': int(result.n_steps),
            'time_sec': float(result.time_sec),
            'history': result.history,
        }
    )
    write_run(output_dir, report, predictions, config)
    return report


def _sorted_members(
    result: PackTrainResult, n_models: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """The finished members of `result` in id order: ids, best steps, predictions
    (part -> (K, N[, C])) and member reports (with ``config`` None)."""
    ids = np.asarray(result.ids, dtype=np.int64)
    # Without an online ensemble, train_pack runs until every member has stopped.
    if sorted(ids.tolist()) != list(range(n_models)):
        raise RuntimeError(
            f'Expected all {n_models} members to finish, got ids {ids.tolist()}'
        )
    order = np.argsort(ids, kind='stable')
    member_by_id = {int(m['id']): m for m in result.members}
    members = [
        {
            'id': int(ids[i]),
            'best_step': int(member_by_id[int(ids[i])]['best_step']),
            'config': None,
            'metrics': member_by_id[int(ids[i])]['metrics'],
        }
        for i in order
    ]
    predictions = {part: result.predictions[part][order] for part in PREDICTION_PARTS}
    return ids[order], np.asarray(result.best_steps)[order], predictions, members
