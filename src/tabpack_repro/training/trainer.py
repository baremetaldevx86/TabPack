"""The pack training loop (a23), shared by the homogeneous ensemble and TabPack.

One epoch = every running member sees every training row once, in its own order.
After each epoch: evaluate val/test for every running member, update PackState,
stop members by early stopping, move stopped members (at their best checkpoint) to
the FinishedPool, remove them from model/optimizer/state, then update the online
ensemble (if any). The loop ends when no member is running, or when the online
ensemble's patience is exhausted (official semantics; unfinished members are then
simply dropped).
"""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.metrics import ScoreFn, compute_metrics, make_score_fn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.nn.pack_ops import make_keep_idx, pack_load_members_, pack_select_
from tabpack_repro.optim.pack_utils import optimizer_select_
from tabpack_repro.training.batches import epoch_size, generate_member_batches
from tabpack_repro.training.evaluate import evaluate_pack, predict_pack
from tabpack_repro.training.losses import make_pack_loss
from tabpack_repro.training.state import PackState
from tabpack_repro.training.stopping import FinishedPool, compute_stop_idx
from tabpack_repro.types import BATCH_DIM, PACK_DIM, PARTS, PartKey, TaskType

# Parts evaluated after every epoch (the online ensemble works on these as well).
_EVAL_PARTS: list[PartKey] = ['val', 'test']


@dataclass
class PackTrainResult:
    # Finished members, in order of finishing.
    ids: np.ndarray
    best_steps: np.ndarray
    # part -> (M, N[, C]) numpy predictions at each finished member's best epoch;
    # parts: train, val, test.
    predictions: dict[PartKey, np.ndarray]
    # One dict per finished member: {'id', 'best_step', 'metrics': {part: {...}}}.
    members: list[dict[str, Any]]
    # One dict per epoch: {'epoch', 'step', 'time', 'n_running', 'n_finished',
    #  'train_loss', 'ensemble_val', 'ensemble_test'} (ensemble_* None if no ensemble)
    history: list[dict[str, Any]] = field(default_factory=list)
    n_epochs: int = 0
    n_steps: int = 0
    time_sec: float = 0.0
    # Online ensemble final report (OnlineGreedyEnsemble.report()) or None.
    online_ensemble: dict[str, Any] | None = None


def train_pack(
    *,
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    dataset: PreparedDataset,
    batch_size: int,
    patience: int,
    max_epochs: int,
    seed: int,
    eval_batch_size: int = 32768,
    autocast: contextlib.AbstractContextManager | None = None,
    online_ensemble: OnlineGreedyEnsemble | None = None,
    progress: bool = False,
) -> PackTrainResult:
    """Train every member of `model` with early stopping; see module docstring.

    `dataset` must already be on the model's device. The batch generator is a
    torch.Generator on that device seeded with `seed`. Uses make_pack_loss, the loss
    of the pack is the SUM of per-member mean losses. Metrics of finished members
    are computed with tabpack_repro.metrics.compute_metrics on train/val/test.

    Details (official ``project/tabpack.py::main`` semantics):
    * ``patience < 0`` disables early stopping, ``max_epochs < 0`` removes the epoch
      limit (at least one of them, or an online ensemble, must bound the run).
    * The global RNG (dropout) is not seeded here: seed it before building the model.
    * ``history[i]['train_loss']`` is the epoch mean (over rows) of the loss averaged
      over the members that were running during that epoch.
    * On return, ``model``/``optimizer`` hold only the members that were still
      running when the loop ended (their latest weights), possibly none.
    """
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if dataset.size('train') < 1:
        raise ValueError('The training part is empty')
    if max_epochs == 0:
        raise ValueError('max_epochs must be positive, or negative for no limit')
    if patience < 0 and max_epochs < 0 and online_ensemble is None:
        raise ValueError(
            'Training would never stop: set patience >= 0, max_epochs > 0 or pass '
            'an online ensemble'
        )
    device = next(model.parameters()).device
    if dataset.y['train'].device != device:
        raise ValueError(
            f'The dataset is on {dataset.y["train"].device}, the model on {device}; '
            'move the dataset first (PreparedDataset.to)'
        )

    task = dataset.task
    train_size = dataset.size('train')
    n_batches = epoch_size(train_size, batch_size)
    loss_fn = make_pack_loss(task.type_)
    x_num_train = None if dataset.x_num is None else dataset.x_num['train']
    x_cat_train = None if dataset.x_cat is None else dataset.x_cat['train']
    y_train = dataset.y['train']
    score_fns = {part: make_score_fn(dataset.y[part], task) for part in _EVAL_PARTS}
    y_true = {part: dataset.y[part].cpu().numpy() for part in PARTS}
    use_autocast = contextlib.nullcontext() if autocast is None else autocast

    state = PackState(model.pack_size)
    pool = FinishedPool()
    members: list[dict[str, Any]] = []
    finished_predictions: dict[PartKey, list[np.ndarray]] = {p: [] for p in PARTS}
    history: list[dict[str, Any]] = []
    generator = torch.Generator(device).manual_seed(seed)
    ensemble_scores: dict[PartKey, float | None] = dict.fromkeys(_EVAL_PARTS)
    epoch = 0
    step = 0

    start_time = time.perf_counter()
    progress_bar = tqdm(
        total=max_epochs if max_epochs > 0 else None,
        unit='epoch',
        disable=not progress,
        leave=False,
    )
    try:
        while state.pack_size > 0 and (
            online_ensemble is None or online_ensemble.is_running
        ):
            pack_size = state.pack_size
            if model.pack_size != pack_size:
                raise RuntimeError(
                    f'The model has {model.pack_size} members, the state {pack_size}'
                )

            # >>> Training: one pass over the training set for every member.
            model.train()
            batches = generate_member_batches(
                train_size=train_size,
                batch_size=batch_size,
                pack_size=pack_size,
                generator=generator,
            )
            # Sum over batches of (batch size * summed member loss), kept on the
            # device so that the epoch needs no host-device sync until evaluation.
            loss_accumulator = torch.zeros((), dtype=torch.float32, device=device)
            for batch_idx in batches:  # (K, b): the rows of each member.
                with use_autocast:
                    logits = model(
                        None if x_num_train is None else x_num_train[batch_idx],
                        None if x_cat_train is None else x_cat_train[batch_idx],
                    )
                # The scale of the gradients must not depend on the number of
                # members, so the per-member losses are summed, not averaged.
                loss = loss_fn(logits.float(), y_train[batch_idx]).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state.step()
                step += 1
                loss_accumulator += loss.detach() * batch_idx.shape[BATCH_DIM]
            epoch += 1
            # No autograd graph may be alive when members are removed below.
            del batches, batch_idx, logits, loss

            # >>> Evaluation of the latest weights of every running member.
            scores, predictions = evaluate_pack(
                model,
                dataset,
                _EVAL_PARTS,
                score_fns,
                batch_size=eval_batch_size,
                autocast=autocast,
            )
            state.update(scores['val'], predictions, _model_state(model))
            del scores
            # After state.update has synced with the device, this sync is free.
            train_loss = loss_accumulator.item() / (train_size * pack_size)

            # >>> Early stopping: finished members leave the pack.
            stop_idx = compute_stop_idx(
                state.n_bad_updates,
                state.steps,
                patience=patience,
                epoch_size=n_batches,
                max_epochs=max_epochs,
            )
            if stop_idx is None:
                latest_predictions = predictions
            else:
                stop_idx_torch = torch.as_tensor(stop_idx, dtype=torch.int64)
                keep_idx_torch = make_keep_idx(pack_size, stop_idx_torch)
                keep_idx_device = keep_idx_torch.to(device)
                latest_predictions = {
                    part: x.index_select(PACK_DIM, keep_idx_device)
                    for part, x in predictions.items()
                }
                stopped_ids = state.ids[stop_idx]
                stopped_steps = state.best_step[stop_idx]

                # Predictions of the stopped members at their best checkpoints.
                pack_load_members_(model, state.best_model_state, stop_idx_torch)
                with _members_selected(model, stop_idx_torch):
                    final_predictions = {
                        part: predict_pack(
                            model,
                            dataset,
                            part,
                            batch_size=eval_batch_size,
                            autocast=autocast,
                        )
                        for part in PARTS
                    }
                pool.extend(
                    stopped_ids,
                    stopped_steps,
                    {part: final_predictions[part] for part in _EVAL_PARTS},
                )
                final_predictions_np = {
                    part: x.cpu().numpy() for part, x in final_predictions.items()
                }
                del final_predictions
                for part, x in final_predictions_np.items():
                    finished_predictions[part].append(x)
                members.extend(
                    _member_reports(
                        stopped_ids, stopped_steps, final_predictions_np, y_true, task
                    )
                )

                # Remove the stopped members everywhere (no autograd graph is alive).
                pack_select_(model, keep_idx_torch)
                optimizer_select_(optimizer, keep_idx_device)
                state.select_(keep_idx_torch)
            del predictions

            # >>> Online ensemble: running members at their latest epoch + finished.
            if online_ensemble is not None:
                improved = online_ensemble.update(
                    running_ids=state.ids,
                    running_steps=state.steps,
                    running_predictions=latest_predictions,
                    finished_ids=pool.ids,
                    finished_steps=pool.steps,
                    finished_predictions=pool.predictions,
                )
                if improved or ensemble_scores['val'] is None:
                    ensemble_scores = _ensemble_scores(online_ensemble, score_fns)
            del latest_predictions

            history.append(
                {
                    'epoch': epoch,
                    'step': step,
                    'time': time.perf_counter() - start_time,
                    'n_running': state.pack_size,
                    'n_finished': len(pool),
                    'train_loss': train_loss,
                    'ensemble_val': ensemble_scores['val'],
                    'ensemble_test': ensemble_scores['test'],
                }
            )
            progress_bar.update()
            if progress:
                progress_bar.set_postfix(
                    running=state.pack_size,
                    finished=len(pool),
                    loss=f'{train_loss:.4f}',
                    **(
                        {}
                        if ensemble_scores['val'] is None
                        else {'ens_val': f'{ensemble_scores["val"]:.4f}'}
                    ),
                )
    finally:
        progress_bar.close()

    return PackTrainResult(
        ids=pool.ids.copy(),
        best_steps=pool.steps.copy(),
        predictions={
            part: _concat_predictions(finished_predictions[part], dataset, part)
            for part in PARTS
        },
        members=members,
        history=history,
        n_epochs=epoch,
        n_steps=step,
        time_sec=time.perf_counter() - start_time,
        online_ensemble=(
            None if online_ensemble is None else online_ensemble.report(y_true)
        ),
    )


def _model_state(model: nn.Module) -> dict[str, Tensor]:
    """The keys of ``pack_state_dict(model)``, but detached references, not clones.

    PackState clones this on its first update and afterwards copies only the
    improved members' slices, so a full per-epoch copy of the model is not needed.
    """
    return {
        name: x.detach()
        for name, x in itertools.chain(model.named_parameters(), model.named_buffers())
    }


@contextlib.contextmanager
def _members_selected(model: nn.Module, member_idx: Tensor) -> Iterator[None]:
    """Temporarily keep only the members `member_idx` of `model`.

    ``pack_select_`` rebinds every parameter's ``.data`` and every buffer to a new
    slice, so keeping references to the old tensors is enough to undo it exactly.
    Gradients are dropped (pack_select_ sets ``.grad = None``); call this only when
    no autograd graph that used the parameters is alive.
    """
    parameters = [(p, p.data) for p in model.parameters()]
    buffers = [
        (module, name, buffer)
        for module in model.modules()
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False)
    ]
    pack_select_(model, member_idx)
    try:
        yield
    finally:
        for parameter, data in parameters:
            parameter.data = data
        for module, name, buffer in buffers:
            setattr(module, name, buffer)


def _member_reports(
    ids: np.ndarray,
    best_steps: np.ndarray,
    predictions: dict[PartKey, np.ndarray],
    y_true: dict[PartKey, np.ndarray],
    task: TaskInfo,
) -> list[dict[str, Any]]:
    return [
        {
            'id': int(member_id),
            'best_step': int(best_step),
            'metrics': {
                part: compute_metrics(y_true[part], predictions[part][i], task)
                for part in PARTS
            },
        }
        for i, (member_id, best_step) in enumerate(zip(ids, best_steps, strict=True))
    ]


def _ensemble_scores(
    online_ensemble: OnlineGreedyEnsemble, score_fns: dict[PartKey, ScoreFn]
) -> dict[PartKey, float | None]:
    """Val/test scores of the current online ensemble (None before its first update).

    The val score is the ensemble's own; the test score is computed here once per
    ensemble change (the ensemble does not change when it does not improve).
    """
    if online_ensemble.score is None:
        return dict.fromkeys(_EVAL_PARTS)
    test_predictions = online_ensemble.predictions()['test']
    return {
        'val': float(online_ensemble.score),
        'test': float(score_fns['test'](test_predictions[None]).item()),
    }


def _concat_predictions(
    chunks: list[np.ndarray], dataset: PreparedDataset, part: PartKey
) -> np.ndarray:
    if chunks:
        return np.concatenate(chunks)
    # No member finished: an empty (0, N[, C]) array keeps the shapes consistent.
    task = dataset.task
    shape: tuple[int, ...] = (0, dataset.size(part))
    if task.type_ == TaskType.MULTICLASS:
        assert task.n_classes is not None
        shape = (*shape, task.n_classes)
    return np.zeros(shape, dtype=np.float32)
