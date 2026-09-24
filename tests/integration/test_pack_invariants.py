"""Cross-module invariants: a pack of K members is K independent models (a53).

A ModelPack trained with AdamWPack / MuonAdamWPack on per-member batches (loss =
sum of the per-member mean losses) must give every member exactly the trajectory
of that member trained alone, as a K=1 pack with its own depth, dropout rate and
hyperparameters, from the same initial weights and on the same rows. The tests
below check this end to end through the real modules (nn, optim, training):

1. independence: no gradient leaks between members, and training trajectories
   match the member trained alone (AdamWPack and MuonAdamWPack, shared and
   per-param step counters);
2. removal: pack_select_ + optimizer_select_ + PackState.select_ in the middle of
   training leave the kept members on their independent trajectories;
3. checkpointing: pack_state_dict / pack_load_members_ restore members exactly, and
   PackState's best predictions are reproduced by re-predicting (predict_pack)
   after loading PackState's best model state;
4. heterogeneous dropout: members with p=0 are unaffected by the other members'
   dropout.

Everything runs on CPU in float32. Tolerances are tight (atol=rtol=1e-6): only
kernel-level reassociation differences between a K-member and a 1-member batched
matmul are allowed, not any algorithmic difference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch
from _helpers import make_synthetic_dataset
from torch import Tensor

from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.metrics import make_score_fn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.nn.pack_ops import (
    make_keep_idx,
    pack_load_members_,
    pack_select_,
    pack_state_dict,
)
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups, optimizer_select_
from tabpack_repro.training.batches import epoch_size, generate_member_batches
from tabpack_repro.training.evaluate import evaluate_pack, predict_pack
from tabpack_repro.training.losses import make_pack_loss
from tabpack_repro.training.state import PackState
from tabpack_repro.types import PartKey, TaskType

# The pack: heterogeneous depths (member 0 applies only block 0, members 1 and 3
# apply all three blocks) and per-member optimizer hyperparameters.
K = 4
N_BLOCKS = (1, 3, 2, 3)
D_BLOCK = 16
LR = (1e-2, 3e-3, 2e-2, 5e-3)
WEIGHT_DECAY = (0.0, 1e-2, 1e-1, 3e-2)
MUON_LR = (2e-2, 5e-3, 1e-2, 3e-2)
NO_DROPOUT = (0.0,) * K

# Data: 200 training rows in batches of 64 -> 4 steps per epoch, the last one with
# a smaller batch (8 rows).
N_TRAIN = 200
BATCH_SIZE = 64
N_NUM = 5
CAT_CARDINALITIES = (3, 2)

ATOL = RTOL = 1e-6

OPTIMIZERS = ['adamw', 'muon']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset() -> PreparedDataset:
    return make_synthetic_dataset(
        n_train=N_TRAIN,
        n_val=96,
        n_test=80,
        n_num=N_NUM,
        cat_cardinalities=CAT_CARDINALITIES,
        seed=0,
    )


def _make_model(
    n_blocks: Sequence[int], dropout: Sequence[float], *, seed: int = 0
) -> ModelPack:
    torch.manual_seed(seed)
    return ModelPack(
        n_num_features=N_NUM,
        cat_cardinalities=CAT_CARDINALITIES,
        n_classes=2,
        pack_size=len(n_blocks),
        d_block=D_BLOCK,
        n_blocks=list(n_blocks),
        dropout=list(dropout),
    )


def _member_model(pack: ModelPack, k: int, dropout: Sequence[float]) -> ModelPack:
    """Member k of `pack` (at its current weights) as a standalone K=1 pack.

    The standalone model has only the blocks member k applies, and it gets its
    weights through a plain ``load_state_dict`` (independent of nn.pack_ops).
    """
    n_blocks_k = int(pack.backbone.n_blocks[k])
    single = _make_model([n_blocks_k], [dropout[k]], seed=12345)
    full = pack.state_dict()
    single.load_state_dict(
        {name: full[name][k : k + 1].clone() for name in single.state_dict()},
        strict=True,
    )
    return single


def _make_optimizer(
    kind: str, model: ModelPack, member_ids: Sequence[int], *, shared_step: bool
) -> torch.optim.Optimizer:
    """The optimizer of the members `member_ids` (in pack order)."""
    ids = list(member_ids)
    assert len(ids) == model.pack_size
    kwargs: dict[str, Any] = {
        'lr': [LR[i] for i in ids],
        'weight_decay': [WEIGHT_DECAY[i] for i in ids],
        'pack_size': model.pack_size,
        'shared_step': shared_step,
    }
    if kind == 'adamw':
        return AdamWPack(make_param_groups(model, muon=False), **kwargs)
    assert kind == 'muon'
    return MuonAdamWPack(
        make_param_groups(model, muon=True),
        muon_lr=[MUON_LR[i] for i in ids],
        **kwargs,
    )


def _make_batches(n_epochs: int, *, seed: int = 0) -> list[Tensor]:
    """Per-member batch indices (K, b) for n_epochs epochs; row k = member id k."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_epochs):
        batches.extend(
            generate_member_batches(
                train_size=N_TRAIN,
                batch_size=BATCH_SIZE,
                pack_size=K,
                generator=generator,
            )
        )
    return batches


def _train_step(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    dataset: PreparedDataset,
    idx: Tensor,
) -> Tensor:
    """One optimizer step; idx (K', b) holds the training rows of every member.

    Returns the detached (K',) per-member losses. Nothing that references the
    autograd graph outlives the call, so members can be removed afterwards.
    """
    assert dataset.x_num is not None and dataset.x_cat is not None
    loss_fn = make_pack_loss(TaskType.BINCLASS)
    optimizer.zero_grad()
    losses = loss_fn(
        model(dataset.x_num['train'][idx], dataset.x_cat['train'][idx]),
        dataset.y['train'][idx],
    )
    # Sum of per-member means: member k's gradient is that of its own mean loss.
    losses.sum().backward()
    optimizer.step()
    return losses.detach()


def _train(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    dataset: PreparedDataset,
    batches: Sequence[Tensor],
    member_ids: Sequence[int],
) -> Tensor:
    """Train members `member_ids` (pack order) on their rows of `batches`."""
    ids = torch.tensor(list(member_ids))
    losses = [_train_step(model, optimizer, dataset, b[ids]) for b in batches]
    return torch.stack(losses)  # (n_steps, K')


def _train_alone(
    kind: str,
    initial: ModelPack,
    member_id: int,
    dataset: PreparedDataset,
    batches: Sequence[Tensor],
    *,
    dropout: Sequence[float] = NO_DROPOUT,
    shared_step: bool = True,
) -> tuple[ModelPack, Tensor]:
    """Member `member_id` of `initial` trained alone as a K=1 pack."""
    single = _member_model(initial, member_id, dropout)
    optimizer = _make_optimizer(kind, single, [member_id], shared_step=shared_step)
    losses = _train(single, optimizer, dataset, batches, [member_id])
    return single, losses[:, 0]


def _assert_member_equals(
    pack: ModelPack, k: int, single: ModelPack, *, exact: bool = False
) -> None:
    """Member k of `pack` has exactly the tensors of the K=1 pack `single`.

    Every tensor of `single` is compared; blocks that member k does not apply
    exist only in the pack and are not compared here.
    """
    pack_tensors = pack.state_dict()
    for name, x in single.state_dict().items():
        actual = pack_tensors[name][k]
        if exact:
            assert torch.equal(actual, x[0]), name
        else:
            torch.testing.assert_close(actual, x[0], atol=ATOL, rtol=RTOL, msg=name)


def _unused_block_names(pack: ModelPack, k: int) -> list[str]:
    """Parameters of the blocks that member k skips (its slice is dead weight)."""
    n_blocks_k = int(pack.backbone.n_blocks[k])
    return [
        name
        for name, _ in pack.named_parameters()
        if name.startswith('backbone.blocks.') and int(name.split('.')[2]) >= n_blocks_k
    ]


# ---------------------------------------------------------------------------
# 1. Independence
# ---------------------------------------------------------------------------


def test_gradients_do_not_leak_between_members() -> None:
    """Member k's gradient in the pack == the gradient of member k trained alone."""
    dataset = _dataset()
    pack = _make_model(N_BLOCKS, NO_DROPOUT)
    idx = _make_batches(1)[0]
    loss_fn = make_pack_loss(TaskType.BINCLASS)
    assert dataset.x_num is not None and dataset.x_cat is not None

    def backward(model: ModelPack, rows: Tensor) -> Tensor:
        losses = loss_fn(
            model(dataset.x_num['train'][rows], dataset.x_cat['train'][rows]),
            dataset.y['train'][rows],
        )
        losses.sum().backward()
        return losses.detach()

    pack_losses = backward(pack, idx)
    pack_grads = {name: p.grad for name, p in pack.named_parameters()}
    for k in range(K):
        single = _member_model(pack, k, NO_DROPOUT)
        single_losses = backward(single, idx[k : k + 1])
        torch.testing.assert_close(pack_losses[k], single_losses[0], atol=0, rtol=0)
        for name, p in single.named_parameters():
            assert p.grad is not None and bool(p.grad.abs().sum() > 0), name
            torch.testing.assert_close(
                pack_grads[name][k], p.grad[0], atol=ATOL, rtol=RTOL, msg=name
            )
        # Blocks that member k skips receive exactly zero gradient from it.
        for name in _unused_block_names(pack, k):
            assert torch.equal(
                pack_grads[name][k], torch.zeros_like(pack_grads[name][k])
            ), name


def test_scaling_one_members_loss_does_not_change_the_others() -> None:
    """The loss is a sum of per-member terms: member j's loss weight is irrelevant
    for every other member's gradient (bitwise)."""
    dataset = _dataset()
    idx = _make_batches(1)[0]
    loss_fn = make_pack_loss(TaskType.BINCLASS)
    assert dataset.x_num is not None and dataset.x_cat is not None

    def grads(weights: Tensor) -> dict[str, Tensor]:
        pack = _make_model(N_BLOCKS, NO_DROPOUT)
        losses = loss_fn(
            pack(dataset.x_num['train'][idx], dataset.x_cat['train'][idx]),
            dataset.y['train'][idx],
        )
        (losses * weights).sum().backward()
        return {name: p.grad.clone() for name, p in pack.named_parameters()}

    reference = grads(torch.ones(K))
    scaled = grads(torch.tensor([1.0, 1000.0, 1.0, 0.0]))
    for name, g in reference.items():
        for k in (0, 2):
            assert torch.equal(scaled[name][k], g[k]), (name, k)
        assert not torch.equal(scaled[name][1], g[1]) or not bool(g[1].any()), name


@pytest.mark.parametrize('shared_step', [True, False], ids=['shared', 'per_param'])
@pytest.mark.parametrize('kind', OPTIMIZERS)
def test_pack_training_equals_training_each_member_alone(
    kind: str, shared_step: bool
) -> None:
    dataset = _dataset()
    batches = _make_batches(2)
    initial = _make_model(N_BLOCKS, NO_DROPOUT)
    initial_state = pack_state_dict(initial)

    pack = _make_model(N_BLOCKS, NO_DROPOUT)
    pack.load_state_dict(initial.state_dict())
    optimizer = _make_optimizer(kind, pack, range(K), shared_step=shared_step)
    pack_losses = _train(pack, optimizer, dataset, batches, range(K))

    for k in range(K):
        single, single_losses = _train_alone(
            kind, initial, k, dataset, batches, shared_step=shared_step
        )
        torch.testing.assert_close(
            pack_losses[:, k], single_losses, atol=ATOL, rtol=RTOL
        )
        _assert_member_equals(pack, k, single)
        # Training moved every parameter of the member (the check is not vacuous).
        for name, p in single.named_parameters():
            assert not torch.equal(p[0], initial_state[name][k]), name

    # Members differ from each other (per-member lr / wd / batches / depth).
    head = pack.head.weight.detach()
    assert all(not torch.equal(head[i], head[j]) for i in range(K) for j in range(i))


@pytest.mark.parametrize('kind', OPTIMIZERS)
def test_skipped_blocks_only_decay(kind: str) -> None:
    """A member's slice of a block it skips gets zero gradient, so the optimizer
    only applies decoupled weight decay to it (and never touches its moments)."""
    dataset = _dataset()
    batches = _make_batches(1)
    pack = _make_model(N_BLOCKS, NO_DROPOUT)
    initial = pack_state_dict(pack)
    optimizer = _make_optimizer(kind, pack, range(K), shared_step=True)
    _train(pack, optimizer, dataset, batches, range(K))

    params = dict(pack.named_parameters())
    n_steps = len(batches)
    for k in range(K):
        for name in _unused_block_names(pack, k):
            p = params[name]
            is_muon_weight = kind == 'muon' and p.ndim == 3
            lr = MUON_LR[k] if is_muon_weight else LR[k]
            # Biases are in the zero weight decay group.
            wd = WEIGHT_DECAY[k] if p.ndim == 3 else 0.0
            expected = initial[name][k] * (1 - lr * wd) ** n_steps
            torch.testing.assert_close(p[k], expected, atol=ATOL, rtol=RTOL, msg=name)
            for key, value in optimizer.state[p].items():
                if isinstance(value, Tensor) and value.shape == p.shape:
                    assert not bool(value[k].any()), (name, key)


# ---------------------------------------------------------------------------
# 2. Removal
# ---------------------------------------------------------------------------


def _score_fns(dataset: PreparedDataset) -> dict[PartKey, Callable[[Tensor], Tensor]]:
    return {
        part: make_score_fn(dataset.y[part], dataset.task) for part in ('val', 'test')
    }


@pytest.mark.parametrize('shared_step', [True, False], ids=['shared', 'per_param'])
@pytest.mark.parametrize('kind', OPTIMIZERS)
@pytest.mark.parametrize(
    'keep',
    [[0, 2], [3, 1, 0], [2]],
    # [0, 2] removes both 3-block members: block 2 then has no member at all.
    ids=['drop_deepest', 'reorder', 'single'],
)
def test_removal_mid_training_keeps_independent_trajectories(
    kind: str, shared_step: bool, keep: list[int]
) -> None:
    dataset = _dataset()
    n_epochs_before, n_epochs_after = 1, 2
    batches = _make_batches(n_epochs_before + n_epochs_after)
    steps_per_epoch = epoch_size(N_TRAIN, BATCH_SIZE)
    split = n_epochs_before * steps_per_epoch
    initial = _make_model(N_BLOCKS, NO_DROPOUT)
    score_fns = _score_fns(dataset)

    pack = _make_model(N_BLOCKS, NO_DROPOUT)
    pack.load_state_dict(initial.state_dict())
    optimizer = _make_optimizer(kind, pack, range(K), shared_step=shared_step)
    state = PackState(K)

    def run_epoch(epoch_batches: Sequence[Tensor]) -> None:
        for idx in epoch_batches:
            _train_step(pack, optimizer, dataset, idx[torch.from_numpy(state.ids)])
            state.step()
        scores, predictions = evaluate_pack(
            pack, dataset, ['val', 'test'], score_fns, batch_size=37
        )
        state.update(scores['val'], predictions, pack_state_dict(pack))
        state.validate()

    run_epoch(batches[:split])

    # Remove members like the trainer does: after the evaluation, when no autograd
    # graph that uses the parameters is alive.
    remove_idx = make_keep_idx(K, torch.tensor(keep))  # complement of `keep`
    assert sorted(set(range(K)) - set(keep)) == remove_idx.tolist()
    keep_idx = torch.tensor(keep)
    pack_select_(pack, keep_idx)
    optimizer_select_(optimizer, keep_idx)
    state.select_(keep_idx)
    state.validate()
    assert pack.pack_size == len(keep)
    assert state.ids.tolist() == keep

    for epoch in range(n_epochs_before, n_epochs_before + n_epochs_after):
        run_epoch(batches[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch])
    assert state.steps.tolist() == [len(batches)] * len(keep)

    for position, member_id in enumerate(keep):
        single, _ = _train_alone(
            kind, initial, member_id, dataset, batches, shared_step=shared_step
        )
        _assert_member_equals(pack, position, single)
        # The kept member's predictions are those of the member trained alone.
        torch.testing.assert_close(
            predict_pack(pack, dataset, 'test')[position],
            predict_pack(single, dataset, 'test')[0],
            atol=ATOL,
            rtol=RTOL,
        )


# ---------------------------------------------------------------------------
# 3. Checkpointing
# ---------------------------------------------------------------------------


def _assert_state_equal(
    actual: dict[str, Tensor], expected: dict[str, Tensor], members: Sequence[int]
) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        for k in members:
            assert torch.equal(actual[name][k], expected[name][k]), (name, k)


@pytest.mark.parametrize('kind', OPTIMIZERS)
def test_pack_load_members_restores_members_exactly(kind: str) -> None:
    dataset = _dataset()
    batches = _make_batches(2)
    pack = _make_model(N_BLOCKS, (0.0, 0.1, 0.2, 0.0))
    optimizer = _make_optimizer(kind, pack, range(K), shared_step=True)
    _train(pack, optimizer, dataset, batches[:3], range(K))

    checkpoint = pack_state_dict(pack)
    checkpoint_copy = {name: x.clone() for name, x in checkpoint.items()}
    checkpoint_predictions = predict_pack(pack, dataset, 'val')

    _train(pack, optimizer, dataset, batches[3:], range(K))
    trained = pack_state_dict(pack)
    # Training after the snapshot does not write through to the snapshot.
    _assert_state_equal(checkpoint, checkpoint_copy, range(K))
    params_before = {name: p for name, p in pack.named_parameters()}

    restored, untouched = [1, 3], [0, 2]
    pack_load_members_(pack, checkpoint, torch.tensor(restored))
    current = pack_state_dict(pack)
    _assert_state_equal(current, checkpoint, restored)
    _assert_state_equal(current, trained, untouched)
    # In place: the optimizer still holds the very same Parameter objects.
    assert all(p is params_before[name] for name, p in pack.named_parameters())

    predictions = predict_pack(pack, dataset, 'val')
    assert torch.equal(predictions[restored], checkpoint_predictions[restored])
    assert not torch.equal(predictions[untouched], checkpoint_predictions[untouched])

    # Loading every member into a freshly initialized pack reproduces the
    # checkpointed pack bit for bit, including its predictions.
    fresh = _make_model(N_BLOCKS, (0.0, 0.1, 0.2, 0.0), seed=7)
    pack_load_members_(fresh, checkpoint, torch.arange(K))
    _assert_state_equal(pack_state_dict(fresh), checkpoint, range(K))
    assert torch.equal(predict_pack(fresh, dataset, 'val'), checkpoint_predictions)

    # Training continues normally after an in-place load.
    _train(pack, optimizer, dataset, batches[:1], range(K))


# Scripted validation scores (epoch, member id): each member's best epoch is known
# in advance and the updates exercise every PackState copy path (first update,
# "all improved", "some improved", ties are not improvements).
SCRIPTED_VAL_SCORES = (
    (0.50, 0.50, 0.50, 0.50),  # first evaluation: everybody improves
    (0.60, 0.40, 0.60, 0.50),  # 0 and 2 improve; 3 ties (not strict)
    (0.70, 0.60, 0.70, 0.60),  # everybody improves
    (0.60, 0.70, 0.80, 0.50),  # 1 and 2 improve
    (0.60, 0.70, 0.90, 0.95),  # after removing member 2: only 3 improves
    (0.65, 0.75, 0.90, 0.80),  # only 1 improves
)
BEST_EPOCH = {0: 2, 1: 5, 3: 4}
REMOVE_AFTER_EPOCH = 3
KEEP_AFTER_REMOVAL = [3, 1, 0]


@pytest.mark.parametrize('kind', OPTIMIZERS)
def test_best_predictions_match_the_best_checkpoint(kind: str) -> None:
    dataset = _dataset()
    batches = _make_batches(len(SCRIPTED_VAL_SCORES))
    steps_per_epoch = epoch_size(N_TRAIN, BATCH_SIZE)
    dropout = (0.0, 0.1, 0.0, 0.2)
    pack = _make_model(N_BLOCKS, dropout)
    optimizer = _make_optimizer(kind, pack, range(K), shared_step=True)
    score_fns = _score_fns(dataset)
    state = PackState(K)
    # member id -> epoch -> predictions of that member (for the final check).
    history: dict[int, dict[int, dict[PartKey, Tensor]]] = {i: {} for i in range(K)}

    for epoch, epoch_scores in enumerate(SCRIPTED_VAL_SCORES):
        for idx in batches[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]:
            _train_step(pack, optimizer, dataset, idx[torch.from_numpy(state.ids)])
            state.step()
        _, predictions = evaluate_pack(pack, dataset, ['val', 'test'], score_fns)
        for position, member_id in enumerate(state.ids.tolist()):
            history[member_id][epoch] = {
                part: x[position].clone() for part, x in predictions.items()
            }
        val_scores = torch.tensor(
            [epoch_scores[i] for i in state.ids], dtype=torch.float64
        )
        state.update(val_scores, predictions, pack_state_dict(pack))
        state.validate()
        if epoch == REMOVE_AFTER_EPOCH:
            keep_idx = torch.tensor(KEEP_AFTER_REMOVAL)
            pack_select_(pack, keep_idx)
            optimizer_select_(optimizer, keep_idx)
            state.select_(keep_idx)
            state.validate()

    assert state.ids.tolist() == KEEP_AFTER_REMOVAL
    expected_best_step = [(BEST_EPOCH[i] + 1) * steps_per_epoch for i in state.ids]
    assert state.best_step.tolist() == expected_best_step
    assert state.best_val_score.tolist() == [
        SCRIPTED_VAL_SCORES[BEST_EPOCH[i]][i] for i in state.ids
    ]

    # The stored predictions are those of each member's best epoch...
    for position, member_id in enumerate(state.ids.tolist()):
        for part, x in state.best_predictions.items():
            assert torch.equal(
                x[position], history[member_id][BEST_EPOCH[member_id]][part]
            ), (member_id, part)

    # ... and loading the stored model state reproduces them exactly. The pack is
    # left in training mode on purpose: predict_pack must switch dropout off.
    pack.train()
    pack_load_members_(pack, state.best_model_state, torch.arange(pack.pack_size))
    for part in ('val', 'test'):
        assert torch.equal(
            predict_pack(pack, dataset, part), state.best_predictions[part]
        ), part
    assert pack.training


# ---------------------------------------------------------------------------
# 4. Heterogeneous dropout
# ---------------------------------------------------------------------------

HETERO_DROPOUT = (0.0, 0.5, 0.0, 0.25)
NO_DROPOUT_MEMBERS = [0, 2]
DROPOUT_MEMBERS = [1, 3]


def test_zero_dropout_members_are_unaffected_in_forward_and_backward() -> None:
    dataset = _dataset()
    idx = _make_batches(1)[0]
    assert dataset.x_num is not None and dataset.x_cat is not None
    x_num, x_cat = dataset.x_num['train'][idx], dataset.x_cat['train'][idx]
    y = dataset.y['train'][idx]
    loss_fn = make_pack_loss(TaskType.BINCLASS)
    pack = _make_model(N_BLOCKS, HETERO_DROPOUT)

    pack.eval()
    with torch.no_grad():
        eval_logits = pack(x_num, x_cat)
    pack.train()
    torch.manual_seed(1)
    train_logits = pack(x_num, x_cat)
    for k in NO_DROPOUT_MEMBERS:
        assert torch.equal(train_logits[k], eval_logits[k]), k
    for k in DROPOUT_MEMBERS:
        assert not torch.equal(train_logits[k], eval_logits[k]), k

    loss_fn(train_logits, y).sum().backward()
    for k in NO_DROPOUT_MEMBERS:
        single = _member_model(pack, k, HETERO_DROPOUT)
        single.train()
        loss_fn(
            single(x_num[k : k + 1], x_cat[k : k + 1]), y[k : k + 1]
        ).sum().backward()
        for name, p in single.named_parameters():
            assert p.grad is not None
            grad = dict(pack.named_parameters())[name].grad
            assert grad is not None
            torch.testing.assert_close(
                grad[k], p.grad[0], atol=ATOL, rtol=RTOL, msg=name
            )


@pytest.mark.parametrize('kind', OPTIMIZERS)
def test_zero_dropout_members_train_like_members_without_dropout(kind: str) -> None:
    dataset = _dataset()
    batches = _make_batches(2)
    initial = _make_model(N_BLOCKS, HETERO_DROPOUT)

    def train_pack(seed: int) -> ModelPack:
        pack = _make_model(N_BLOCKS, HETERO_DROPOUT)
        pack.load_state_dict(initial.state_dict())
        optimizer = _make_optimizer(kind, pack, range(K), shared_step=True)
        pack.train()
        torch.manual_seed(seed)  # the dropout masks
        _train(pack, optimizer, dataset, batches, range(K))
        return pack

    pack_a, pack_b = train_pack(seed=1), train_pack(seed=2)
    for k in NO_DROPOUT_MEMBERS:
        # Different dropout masks for the other members: bitwise the same member.
        _assert_member_equals(pack_a, k, _member_model(pack_b, k, HETERO_DROPOUT))
        single, _ = _train_alone(
            kind, initial, k, dataset, batches, dropout=HETERO_DROPOUT
        )
        _assert_member_equals(pack_a, k, single)
    for k in DROPOUT_MEMBERS:
        # Non-vacuous: the dropout members did depend on the masks.
        assert not torch.equal(pack_a.head.weight[k], pack_b.head.weight[k]), k
