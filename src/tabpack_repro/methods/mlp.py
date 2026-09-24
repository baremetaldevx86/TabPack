"""Ordinary MLP baseline (a30): a plain nn.Sequential MLP trained with torch AdamW.

Deliberately independent of the pack machinery (it doubles as a sanity reference):
Linear -> ReLU -> Dropout blocks (n_blocks x d_block), linear head; weight decay on
weight matrices only (biases: 0.0), same batch size / early stopping / AMP settings /
data pipeline as the other methods; evaluation on val+test after every epoch; the
test metric is taken at the best-val epoch (strict improvement).
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tabpack_repro.config import MLPMethodConfig, MLPModelConfig, config_to_dict
from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.methods.common import (
    RunContext,
    base_report,
    best_member,
    setup_run,
    write_run,
)
from tabpack_repro.metrics import compute_metrics
from tabpack_repro.types import PARTS, PartKey, TaskType

logger = logging.getLogger(__name__)

METHOD = 'mlp'
# The parts evaluated after every epoch (train is evaluated once, at the end).
_EVAL_PARTS: tuple[PartKey, ...] = ('val', 'test')
# Activations the config may name (the same set as the pack models).
_ACTIVATIONS = ('ReLU', 'GELU', 'SiLU')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _one_hot(x_cat: Tensor, cardinalities: Sequence[int]) -> Tensor:
    """(N, n_cat) int64 codes -> (N, sum(cards)) float32; codes >= card -> zeros."""
    columns = [
        # An unknown code (>= cardinality) is clamped to the extra class `card`,
        # whose column is dropped, so it encodes to all zeros.
        nn.functional.one_hot(x_cat[:, i].long().clamp(max=card), card + 1)[:, :-1]
        for i, card in enumerate(cardinalities)
    ]
    return torch.cat(columns, dim=1).to(torch.float32)


def _encode_inputs(dataset: PreparedDataset, part: PartKey) -> Tensor:
    """The model input of a whole part: concat(x_num, one_hot(x_cat)), float32."""
    features = []
    if dataset.x_num is not None:
        features.append(dataset.x_num[part].to(torch.float32))
    if dataset.x_cat is not None and dataset.cat_cardinalities:
        features.append(_one_hot(dataset.x_cat[part], dataset.cat_cardinalities))
    if not features:
        raise ValueError('The dataset has neither numerical nor categorical features')
    return features[0] if len(features) == 1 else torch.cat(features, dim=1)


def _output_dim(task: TaskInfo) -> int:
    if task.type_ == TaskType.MULTICLASS:
        assert task.n_classes is not None
        return task.n_classes
    return 1


def _make_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f'activation must be one of {_ACTIVATIONS}, got {name!r}')
    return getattr(nn, name)()


def _make_model(d_in: int, d_out: int, config: MLPModelConfig) -> nn.Sequential:
    """n_blocks x [Linear -> activation -> Dropout] (width d_block), Linear head."""
    if config.n_blocks < 1:
        raise ValueError(f'n_blocks must be >= 1, got {config.n_blocks}')
    if config.d_block < 1:
        raise ValueError(f'd_block must be >= 1, got {config.d_block}')
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f'dropout must be in [0, 1), got {config.dropout}')
    layers: list[nn.Module] = []
    for i in range(config.n_blocks):
        layers += [
            nn.Linear(d_in if i == 0 else config.d_block, config.d_block),
            _make_activation(config.activation),
            nn.Dropout(config.dropout),
        ]
    layers.append(nn.Linear(config.d_block, d_out))
    return nn.Sequential(*layers)


def _make_optimizer(model: nn.Module, config: MLPMethodConfig) -> torch.optim.AdamW:
    """AdamW with two groups: weight matrices decay, biases do not."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = config.optimizer
    return torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': opt.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=opt.lr,
        betas=(opt.beta1, opt.beta2),
        eps=opt.eps,
    )


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


def _autocast(ctx: RunContext) -> contextlib.AbstractContextManager:
    return contextlib.nullcontext() if ctx.autocast is None else ctx.autocast


def _loss(logits: Tensor, y: Tensor, task: TaskInfo) -> Tensor:
    """Mean loss of a batch; logits are float32 (N,) or (N, C)."""
    if task.type_ == TaskType.BINCLASS:
        return nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    if task.type_ == TaskType.MULTICLASS:
        return nn.functional.cross_entropy(logits, y.long())
    return nn.functional.mse_loss(logits, y.float())


def _forward(model: nn.Module, x: Tensor, task: TaskInfo) -> Tensor:
    """float32 logits: (N,) for binclass/regression, (N, C) for multiclass."""
    logits = model(x).float()
    return logits if task.type_ == TaskType.MULTICLASS else logits.squeeze(-1)


def _to_predictions(logits: Tensor, task: TaskInfo) -> Tensor:
    if task.type_ == TaskType.BINCLASS:
        return torch.sigmoid(logits)
    if task.type_ == TaskType.MULTICLASS:
        return torch.softmax(logits, dim=-1)
    return logits


@torch.inference_mode()
def _predict(model: nn.Module, x: Tensor, ctx: RunContext, batch_size: int) -> Tensor:
    """Probabilities (binclass (N,), multiclass (N, C)) or regression predictions,
    float32 on the model device. Halves the batch size on CUDA OOM."""
    task = ctx.dataset.task
    was_training = model.training
    model.eval()
    try:
        while True:
            try:
                chunks = []
                for start in range(0, len(x), batch_size):
                    with _autocast(ctx):
                        logits = _forward(model, x[start : start + batch_size], task)
                    chunks.append(_to_predictions(logits, task))
                return torch.cat(chunks)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                logger.warning('CUDA OOM in evaluation; eval batch_size=%d', batch_size)
    finally:
        model.train(was_training)


def _check_stopping(patience: int, max_epochs: int) -> None:
    if patience < -1 or max_epochs < -1:
        raise ValueError(
            f'patience and max_epochs must be >= -1, got {patience}, {max_epochs}'
        )
    if patience == -1 and max_epochs == -1:
        raise ValueError(
            'patience=-1 and max_epochs=-1: training would never stop; '
            'enable early stopping or set an epoch limit'
        )


def _numpy(x: Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def run(config: MLPMethodConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train one MLP with early stopping, write report.json / predictions.npz /
    config.toml into `output_dir` and return the report.

    One epoch visits every training row once, in a fresh random order drawn from a
    torch.Generator (on the data device) seeded with ``config.seed``; the last batch
    of an epoch may be smaller. After every epoch the model is evaluated on val and
    test; an epoch improves if its val score is strictly higher than the best so
    far (the first evaluation always does). Training stops once the number of
    consecutive non-improving evaluations exceeds ``patience`` (-1 disables) or
    ``max_epochs`` epochs are done (-1 disables). The final predictions and metrics
    are those of the best epoch; the model state of that epoch is restored to
    compute the train metrics.
    """
    training = config.training
    _check_stopping(training.patience, training.max_epochs)
    if training.batch_size < 1 or training.eval_batch_size < 1:
        raise ValueError(
            'batch_size and eval_batch_size must be >= 1, got '
            f'{training.batch_size}, {training.eval_batch_size}'
        )

    # Seeds every global RNG (model init and dropout below) and builds the data.
    ctx = setup_run(config.seed, config.data, training)
    dataset = ctx.dataset
    task = dataset.task
    device = ctx.device

    x = {part: _encode_inputs(dataset, part) for part in PARTS}
    y_train = dataset.y['train']
    n_train = len(y_train)
    if n_train == 0:
        raise ValueError('The training part is empty')

    # Build on CPU, then move: the initialization does not depend on the device.
    model = _make_model(x['train'].shape[1], _output_dim(task), config.model)
    model.to(device)
    optimizer = _make_optimizer(model, config)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    history: list[dict[str, Any]] = []
    best_val_score = -math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    best_predictions: dict[PartKey, Tensor] = {}
    n_bad_updates = 0
    epoch = 0
    step = 0

    start_time = time.perf_counter()
    while True:
        # ---- one epoch
        model.train()
        loss_sum = torch.zeros((), device=device)
        order = torch.randperm(n_train, generator=generator, device=device)
        for batch_idx in order.split(training.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(ctx):
                logits = _forward(model, x['train'][batch_idx], task)
            loss = _loss(logits, y_train[batch_idx], task)
            loss.backward()
            optimizer.step()
            step += 1
            loss_sum += loss.detach() * len(batch_idx)
        epoch += 1

        # ---- evaluation
        predictions = {
            part: _predict(model, x[part], ctx, training.eval_batch_size)
            for part in _EVAL_PARTS
        }
        scores = {
            part: float(ctx.score_fns[part](predictions[part][None])[0])
            for part in _EVAL_PARTS
        }
        # The first evaluation always improves (also if its score is NaN).
        if best_state is None or scores['val'] > best_val_score:
            best_val_score = scores['val']
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = predictions
            n_bad_updates = 0
        else:
            n_bad_updates += 1
        history.append(
            {
                'epoch': epoch,
                'step': step,
                'time': time.perf_counter() - start_time,
                # Mean loss over the epoch's training rows.
                'train_loss': float(loss_sum) / n_train,
                'val_score': scores['val'],
                'test_score': scores['test'],
            }
        )
        logger.debug('mlp %s', history[-1])

        if (training.patience >= 0 and n_bad_updates > training.patience) or (
            training.max_epochs >= 0 and epoch >= training.max_epochs
        ):
            break
    time_sec = time.perf_counter() - start_time

    # ---- final (best-epoch) predictions and metrics
    assert best_state is not None
    model.load_state_dict(best_state)
    final_predictions = {part: _numpy(best_predictions[part]) for part in _EVAL_PARTS}
    all_predictions = {
        'train': _numpy(_predict(model, x['train'], ctx, training.eval_batch_size)),
        **final_predictions,
    }
    member_metrics = {
        part: compute_metrics(ctx.y_true[part], all_predictions[part], task)
        for part in PARTS
    }
    config_dict = config_to_dict(config)
    members = [
        {
            'id': 0,
            'best_step': best_step,
            # The member's hyperparameters, in the per-member format of TabPack.
            'config': {key: config_dict[key] for key in ('model', 'optimizer')},
            'metrics': member_metrics,
        }
    ]

    report = base_report(METHOD, config, config.seed, ctx)
    report.update(
        {
            'metrics': {part: member_metrics[part] for part in _EVAL_PARTS},
            'members': members,
            'ensemble': None,
            'best_member': best_member(members),
            'n_epochs': epoch,
            'n_steps': step,
            'time_sec': time_sec,
            'history': history,
        }
    )
    write_run(output_dir, report, final_predictions, config)
    return report
