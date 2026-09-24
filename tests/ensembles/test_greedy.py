"""Tests for tabpack_repro.ensembles.greedy (a25)."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.ensembles.greedy import greedy_ensemble
from tabpack_repro.metrics import make_score_fn
from tabpack_repro.types import TaskType

BINCLASS_ACCURACY = TaskInfo(type_=TaskType.BINCLASS, score='accuracy', n_classes=2)


def neg_mse_score_fn(target: Tensor):
    """Score (M, N[, C]) predictions by the negative MSE to ``target``."""

    def score_fn(predictions: Tensor) -> Tensor:
        dims = tuple(range(1, predictions.ndim))
        return -(predictions - target).square().mean(dim=dims)

    return score_fn


def accuracy_score_fn(y_true: list[int]):
    return make_score_fn(torch.tensor(y_true), BINCLASS_ACCURACY)


def reference_path(
    predictions: Tensor, score_fn, max_ensemble_size: int | None = None
) -> tuple[list[int], list[float]]:
    """Naive greedy selection: every candidate ensemble is averaged from scratch.

    Returns the selection order and the ensemble score after each step.
    """
    n = len(predictions)
    limit = n if max_ensemble_size is None else min(n, max_ensemble_size)
    individual = score_fn(predictions).tolist()
    first = max(range(n), key=individual.__getitem__)
    path = [first]
    scores = [score_fn(predictions[[first]])[0].item()]
    while len(path) < limit:
        candidates = [i for i in range(n) if i not in path]
        candidate_scores = [
            score_fn(predictions[[*path, c]].mean(0, keepdim=True))[0].item()
            for c in candidates
        ]
        best = max(candidate_scores)
        if best <= scores[-1]:
            break
        ties = [
            c for c, s in zip(candidates, candidate_scores, strict=True) if s == best
        ]
        path.append(max(ties, key=individual.__getitem__))
        scores.append(best)
    return path, scores


def random_problem(seed: int, m: int = 12, n: int = 40) -> tuple[Tensor, Tensor]:
    """Noisy, partly biased float64 predictions of a random target."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(n, generator=gen, dtype=torch.float64)
    noise_scale = 0.2 + torch.rand(m, 1, generator=gen, dtype=torch.float64)
    bias = 0.1 * torch.randn(m, 1, generator=gen, dtype=torch.float64)
    noise = torch.randn(m, n, generator=gen, dtype=torch.float64)
    return target + bias + noise_scale * noise, target


def ensemble_score(predictions: Tensor, idx: Tensor, score_fn) -> float:
    return score_fn(predictions[idx].mean(0, keepdim=True))[0].item()


# ---------------------------------------------------------------------------
# Hand-built paths
# ---------------------------------------------------------------------------


def test_neg_mse_hand_built_path():
    target = torch.zeros(1)
    # Individual scores: -1, -1, -0.25, -9 -> start with 2 (0.5).
    # Step 2: pairs {2,0} -> 0.75, {2,1} -> -0.25, {2,3} -> 1.75: add 1.
    # Step 3: {2,1,0} -> 1/6 (score -1/36 > -1/16): add 0.
    # Step 4: {2,1,0,3} -> 0.875 is worse: stop.
    predictions = torch.tensor([[1.0], [-1.0], [0.5], [3.0]])
    idx = greedy_ensemble(predictions, score_fn=neg_mse_score_fn(target))
    assert idx.tolist() == [0, 1, 2]
    assert idx.dtype == torch.int64


def test_neg_mse_matches_hand_built_path_with_size_limit():
    target = torch.zeros(1)
    predictions = torch.tensor([[1.0], [-1.0], [0.5], [3.0]])
    score_fn = neg_mse_score_fn(target)
    assert greedy_ensemble(
        predictions, score_fn=score_fn, max_ensemble_size=1
    ).tolist() == [2]
    assert greedy_ensemble(
        predictions, score_fn=score_fn, max_ensemble_size=2
    ).tolist() == [1, 2]
    assert greedy_ensemble(
        predictions, score_fn=score_fn, max_ensemble_size=3
    ).tolist() == [0, 1, 2]
    assert greedy_ensemble(
        predictions, score_fn=score_fn, max_ensemble_size=100
    ).tolist() == [0, 1, 2]


# y = [1, 1, 1, 0, 0, 0]; each member is wrong exactly where noted.
Y_ACC = [1, 1, 1, 0, 0, 0]
MEMBER_A = [0.4, 0.9, 0.9, 0.1, 0.1, 0.1]  # wrong on 0 -> 5/6
MEMBER_B = [0.9, 0.3, 0.9, 0.1, 0.1, 0.1]  # wrong on 1 -> 5/6, A+B -> 6/6
MEMBER_C = [0.9, 0.9, 0.9, 0.6, 0.6, 0.1]  # wrong on 3, 4 -> 4/6, A+C -> 6/6
MEMBER_BAD = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]  # 0/6


def test_accuracy_hand_built_path():
    predictions = torch.tensor([MEMBER_BAD, MEMBER_A, MEMBER_B])
    idx = greedy_ensemble(predictions, score_fn=accuracy_score_fn(Y_ACC))
    # Start with A (5/6), add B (6/6), then nothing can be strictly better.
    assert idx.tolist() == [1, 2]


def test_tie_break_prefers_best_individual_score():
    # A+C and A+B both reach 6/6; C comes first in index order, but B has the better
    # individual score (5/6 vs 4/6), so B wins the tie.
    predictions = torch.tensor([MEMBER_A, MEMBER_C, MEMBER_B])
    score_fn = accuracy_score_fn(Y_ACC)
    assert score_fn(predictions).tolist() == pytest.approx([5 / 6, 4 / 6, 5 / 6])
    idx = greedy_ensemble(predictions, score_fn=score_fn)
    assert idx.tolist() == [0, 2]


def test_tie_break_takes_first_index_on_equal_individual_scores():
    # Two identical copies of B: equal candidate and individual scores -> first one.
    predictions = torch.tensor([MEMBER_A, MEMBER_C, MEMBER_B, MEMBER_B])
    idx = greedy_ensemble(predictions, score_fn=accuracy_score_fn(Y_ACC))
    assert idx.tolist() == [0, 2]


def test_initial_member_is_first_best_individual():
    # A and B tie individually (5/6); the first one (index 1 here: B) starts.
    predictions = torch.tensor([MEMBER_C, MEMBER_B, MEMBER_A])
    idx = greedy_ensemble(
        predictions, score_fn=accuracy_score_fn(Y_ACC), max_ensemble_size=1
    )
    assert idx.tolist() == [1]


def test_stops_without_strict_improvement():
    # Pairs vote 0.45 on the disagreeing samples, which is worse than any single
    # member (2/3). The full average would reach 3/3, but greedy never sees it.
    predictions = torch.tensor([[0.9, 0.9, 0.0], [0.9, 0.0, 0.9], [0.0, 0.9, 0.9]])
    score_fn = accuracy_score_fn([1, 1, 1])
    assert greedy_ensemble(predictions, score_fn=score_fn).tolist() == [0]
    assert ensemble_score(predictions, torch.arange(3), score_fn) == 1.0


def test_equal_score_candidate_is_not_added():
    # Adding B to A keeps the accuracy at 2/3 (not strictly better): stop at A.
    predictions = torch.tensor([[0.9, 0.9, 0.2], [0.8, 0.8, 0.1]])
    idx = greedy_ensemble(predictions, score_fn=accuracy_score_fn([1, 1, 1]))
    assert idx.tolist() == [0]


def test_all_equal_predictions_stop_at_one():
    predictions = torch.full((5, 7), 0.3)
    target = torch.linspace(0, 1, 7)
    assert greedy_ensemble(predictions, score_fn=neg_mse_score_fn(target)).tolist() == [
        0
    ]
    y = [0, 1, 0, 1, 0, 1, 1]
    assert greedy_ensemble(predictions, score_fn=accuracy_score_fn(y)).tolist() == [0]


def test_single_prediction():
    predictions = torch.tensor([[0.2, 0.7, 0.9]])
    idx = greedy_ensemble(predictions, score_fn=accuracy_score_fn([0, 1, 1]))
    assert idx.tolist() == [0]
    assert idx.dtype == torch.int64


def test_multiclass_predictions():
    task = TaskInfo(type_=TaskType.MULTICLASS, score='accuracy', n_classes=3)
    y = torch.tensor([0, 1, 2, 0])
    # Member 0 is wrong on sample 3; member 1 is wrong on sample 1; their average is
    # right everywhere. Member 2 is wrong on everything.
    m0 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1]]
    m1 = [[0.8, 0.1, 0.1], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]
    m2 = [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]
    predictions = torch.tensor([m2, m1, m0])
    idx = greedy_ensemble(predictions, score_fn=make_score_fn(y, task))
    assert idx.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Properties on random problems
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(8))
def test_matches_naive_reference(seed):
    predictions, target = random_problem(seed)
    score_fn = neg_mse_score_fn(target)
    path, _ = reference_path(predictions, score_fn)
    assert len(path) > 2  # the problem is non-trivial
    idx = greedy_ensemble(predictions, score_fn=score_fn)
    assert idx.tolist() == sorted(path)


@pytest.mark.parametrize('seed', range(4))
def test_max_ensemble_size_gives_path_prefix(seed):
    predictions, target = random_problem(seed)
    score_fn = neg_mse_score_fn(target)
    path, _ = reference_path(predictions, score_fn)
    for k in range(1, len(predictions) + 2):
        idx = greedy_ensemble(predictions, score_fn=score_fn, max_ensemble_size=k)
        assert len(idx) <= k
        assert idx.tolist() == sorted(path[:k])


@pytest.mark.parametrize('seed', range(4))
def test_no_duplicates_and_non_decreasing_score_along_path(seed):
    predictions, target = random_problem(seed, m=10)
    # Duplicate rows (as in the online pool) must still be selected at most once each.
    predictions = torch.cat([predictions, predictions[:4]])
    score_fn = neg_mse_score_fn(target)
    previous_idx: set[int] = set()
    previous_score = -float('inf')
    for k in range(1, len(predictions) + 1):
        idx = greedy_ensemble(predictions, score_fn=score_fn, max_ensemble_size=k)
        values = idx.tolist()
        assert values == sorted(values)
        assert len(set(values)) == len(values)
        assert 0 <= min(values) and max(values) < len(predictions)
        assert previous_idx <= set(values)
        score = ensemble_score(predictions, idx, score_fn)
        if len(values) > len(previous_idx):
            assert score > previous_score
        else:
            assert values == sorted(previous_idx)
        previous_idx, previous_score = set(values), score


def test_candidate_scoring_is_vectorized():
    predictions, target = random_problem(0, m=6)
    base_score_fn = neg_mse_score_fn(target)
    shapes = []

    def score_fn(y_pred: Tensor) -> Tensor:
        shapes.append(tuple(y_pred.shape))
        return base_score_fn(y_pred)

    idx = greedy_ensemble(predictions, score_fn=score_fn)
    size = len(idx)
    n = predictions.shape[1]
    # Individual scores, the initial ensemble, then one call per step with all the
    # remaining candidates (the last call is the one that fails to improve, unless
    # every member was selected).
    expected = [(6, n), (1, n)] + [(6 - s, n) for s in range(1, min(size + 1, 6))]
    assert shapes == expected


def test_float32_accuracy_with_score_fn_from_metrics():
    gen = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (64,), generator=gen)
    logits = (2 * y - 1).float() + 1.5 * torch.randn(9, 64, generator=gen)
    predictions = torch.sigmoid(logits)
    score_fn = make_score_fn(y, BINCLASS_ACCURACY)
    path, scores = reference_path(predictions, score_fn)
    idx = greedy_ensemble(predictions, score_fn=score_fn)
    assert idx.tolist() == sorted(path)
    assert all(a < b for a, b in itertools.pairwise(scores))
    assert ensemble_score(predictions, idx, score_fn) == pytest.approx(scores[-1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_rejects_non_finite_predictions(bad):
    predictions = torch.rand(3, 4)
    predictions[1, 2] = bad
    with pytest.raises(ValueError, match='finite'):
        greedy_ensemble(predictions, score_fn=neg_mse_score_fn(torch.zeros(4)))


@pytest.mark.parametrize('shape', [(0, 4), (3, 0), (0,)])
def test_rejects_empty_predictions(shape):
    with pytest.raises(ValueError, match='non-empty'):
        greedy_ensemble(torch.empty(shape), score_fn=lambda p: p.sum(1))


def test_rejects_integer_predictions():
    with pytest.raises(TypeError, match='floating'):
        greedy_ensemble(torch.ones(3, 4, dtype=torch.int64), score_fn=lambda p: p)


@pytest.mark.parametrize('size', [0, -1, 1.5, True])
def test_rejects_bad_max_ensemble_size(size):
    with pytest.raises(ValueError, match='max_ensemble_size'):
        greedy_ensemble(
            torch.rand(3, 4),
            score_fn=neg_mse_score_fn(torch.zeros(4)),
            max_ensemble_size=size,
        )


def test_rejects_bad_score_shape():
    with pytest.raises(ValueError, match='shape'):
        greedy_ensemble(torch.rand(3, 4), score_fn=lambda p: p.mean())


def test_rejects_nan_scores():
    with pytest.raises(ValueError, match='NaN'):
        greedy_ensemble(
            torch.rand(3, 4), score_fn=lambda p: torch.full((len(p),), torch.nan)
        )


@pytest.mark.gpu
def test_cuda(cuda_device):
    predictions, target = random_problem(1)
    score_fn = neg_mse_score_fn(target)
    cpu_idx = greedy_ensemble(predictions, score_fn=score_fn)
    idx = greedy_ensemble(
        predictions.to(cuda_device), score_fn=neg_mse_score_fn(target.to(cuda_device))
    )
    assert idx.device.type == 'cuda'
    assert idx.dtype == torch.int64
    assert idx.cpu().tolist() == cpu_idx.tolist()
