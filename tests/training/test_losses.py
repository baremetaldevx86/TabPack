"""Tests for tabpack_repro.training.losses (a19)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tabpack_repro.training.losses import make_pack_loss
from tabpack_repro.types import TaskType

K, B, C = 4, 7, 3


def _data(task_type: TaskType, dtype: torch.dtype = torch.float32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    if task_type == TaskType.BINCLASS:
        logits = torch.randn(K, B, generator=g, dtype=dtype) * 3
        y = torch.randint(0, 2, (K, B), generator=g)
    elif task_type == TaskType.MULTICLASS:
        logits = torch.randn(K, B, C, generator=g, dtype=dtype) * 3
        y = torch.randint(0, C, (K, B), generator=g)
    else:
        logits = torch.randn(K, B, generator=g, dtype=dtype)
        y = torch.randn(K, B, generator=g, dtype=dtype)
    return logits, y


def _reference_member_loss(task_type: TaskType, logits_k, y_k):
    if task_type == TaskType.BINCLASS:
        return F.binary_cross_entropy_with_logits(logits_k, y_k.float())
    if task_type == TaskType.MULTICLASS:
        return F.cross_entropy(logits_k, y_k)
    return F.mse_loss(logits_k, y_k)


@pytest.mark.parametrize('task_type', list(TaskType))
def test_per_member_values_match_torch(task_type):
    logits, y = _data(task_type)
    losses = make_pack_loss(task_type)(logits, y)
    assert losses.shape == (K,)
    assert losses.dtype == torch.float32
    expected = torch.stack(
        [_reference_member_loss(task_type, logits[k], y[k]) for k in range(K)]
    )
    torch.testing.assert_close(losses, expected)


@pytest.mark.parametrize('task_type', list(TaskType))
def test_gradient_of_sum_is_per_member_gradient(task_type):
    logits, y = _data(task_type, seed=1)
    logits.requires_grad_(True)
    make_pack_loss(task_type)(logits, y).sum().backward()
    for k in range(K):
        member_logits = logits.detach()[k].clone().requires_grad_(True)
        _reference_member_loss(task_type, member_logits, y[k]).backward()
        torch.testing.assert_close(logits.grad[k], member_logits.grad)


@pytest.mark.parametrize('task_type', list(TaskType))
def test_members_are_independent(task_type):
    # Changing member 0's logits must not change the loss of any other member.
    logits, y = _data(task_type, seed=2)
    loss_fn = make_pack_loss(task_type)
    before = loss_fn(logits, y)
    perturbed = logits.clone()
    # Random (not constant) shift: softmax is invariant to a constant shift.
    perturbed[0] += 5.0 * torch.randn(
        perturbed[0].shape, generator=torch.Generator().manual_seed(3)
    )
    after = loss_fn(perturbed, y)
    torch.testing.assert_close(after[1:], before[1:])
    assert not torch.allclose(after[0], before[0])


def test_binclass_int64_targets_cast_to_logits_dtype():
    logits, y = _data(TaskType.BINCLASS, dtype=torch.float64)
    assert y.dtype == torch.int64
    losses = make_pack_loss(TaskType.BINCLASS)(logits, y)
    assert losses.dtype == torch.float64
    torch.testing.assert_close(
        losses,
        F.binary_cross_entropy_with_logits(logits, y.double(), reduction='none').mean(
            1
        ),
    )


def test_binclass_is_numerically_stable_for_extreme_logits():
    logits = torch.tensor([[1e4, -1e4], [-1e4, 1e4]])
    y = torch.tensor([[1, 0], [1, 0]])
    logits.requires_grad_(True)
    losses = make_pack_loss(TaskType.BINCLASS)(logits, y)
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses[0], torch.tensor(0.0))
    torch.testing.assert_close(losses[1], torch.tensor(1e4))
    losses.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_multiclass_is_numerically_stable_for_extreme_logits():
    logits = torch.tensor([[[1e4, 0.0, -1e4]], [[-1e4, 0.0, 1e4]]])
    y = torch.tensor([[0], [0]])
    losses = make_pack_loss(TaskType.MULTICLASS)(logits, y)
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses, torch.tensor([0.0, 2e4]))


def test_accepts_task_type_strings():
    logits, y = _data(TaskType.REGRESSION)
    torch.testing.assert_close(
        make_pack_loss('regression')(logits, y),
        make_pack_loss(TaskType.REGRESSION)(logits, y),
    )


def test_invalid_task_type_raises():
    with pytest.raises(ValueError, match='Unknown task type'):
        make_pack_loss('ranking')  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('task_type', 'logits_shape', 'y_shape'),
    [
        (TaskType.BINCLASS, (K, B, 1), (K, B)),
        (TaskType.REGRESSION, (K, B, 1), (K, B)),
        (TaskType.MULTICLASS, (K, B), (K, B)),
        (TaskType.BINCLASS, (K, B), (K, B + 1)),
        (TaskType.MULTICLASS, (K, B, C), (K - 1, B)),
    ],
)
def test_shape_mismatch_raises(task_type, logits_shape, y_shape):
    logits = torch.zeros(logits_shape)
    y = torch.zeros(y_shape, dtype=torch.int64)
    with pytest.raises(ValueError, match='must have'):
        make_pack_loss(task_type)(logits, y)


def test_single_member_pack():
    logits = torch.randn(1, B)
    y = torch.randn(1, B)
    losses = make_pack_loss(TaskType.REGRESSION)(logits, y)
    assert losses.shape == (1,)
    torch.testing.assert_close(losses[0], F.mse_loss(logits[0], y[0]))
