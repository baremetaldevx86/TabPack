"""Tests for tabpack_repro.metrics (a08)."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import sklearn.metrics
import torch

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.metrics import (
    HIGHER_IS_BETTER,
    compute_metrics,
    make_score_fn,
    score_pack,
)
from tabpack_repro.types import TaskType


def binclass(score: str = 'accuracy') -> TaskInfo:
    return TaskInfo(type_=TaskType.BINCLASS, score=score, n_classes=2)


def multiclass(score: str = 'accuracy', n_classes: int = 3) -> TaskInfo:
    return TaskInfo(type_=TaskType.MULTICLASS, score=score, n_classes=n_classes)


REGRESSION = TaskInfo(type_=TaskType.REGRESSION, score='rmse')

ALL_TASKS = [
    binclass('accuracy'),
    binclass('roc-auc'),
    binclass('log-loss'),
    multiclass('accuracy'),
    multiclass('log-loss'),
    REGRESSION,
]


def random_case(task: TaskInfo, m: int = 5, n: int = 200, seed: int = 0):
    """Random labels (numpy int64/float) and float32 predictions (M, N[, C])."""
    rng = np.random.default_rng(seed)
    if task.type_ == TaskType.REGRESSION:
        y = rng.standard_normal(n).astype(np.float32)
        pred = (y + rng.standard_normal((m, n))).astype(np.float32)
    elif task.type_ == TaskType.BINCLASS:
        y = rng.integers(0, 2, n)
        # Round to 2 decimals so that exact ties (incl. p=0.5) are frequent.
        pred = np.round(rng.random((m, n)), 2).astype(np.float32)
    else:
        assert task.n_classes is not None
        y = rng.integers(0, task.n_classes, n)
        logits = torch.as_tensor(rng.standard_normal((m, n, task.n_classes)))
        pred = torch.softmax(logits.float(), dim=-1).numpy()
    return y, pred


# --------------------------------------------------------------------------------
# compute_metrics: hand-computed cases
# --------------------------------------------------------------------------------


def test_binclass_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8])
    m = compute_metrics(y, p, binclass())
    assert set(m) == {'accuracy', 'roc-auc', 'log-loss', 'score'}
    # labels [0, 1, 0, 1] -> 2 of 4 correct.
    assert m['accuracy'] == 0.5
    # Pairs (neg, pos): (0.1,0.4) ok, (0.1,0.8) ok, (0.6,0.4) bad, (0.6,0.8) ok.
    assert m['roc-auc'] == pytest.approx(0.75)
    expected_ll = -(math.log(0.9) + math.log(0.4) + math.log(0.4) + math.log(0.8)) / 4
    assert m['log-loss'] == pytest.approx(expected_ll)
    assert m['score'] == m['accuracy']
    assert all(isinstance(v, float) for v in m.values())


def test_binclass_score_follows_task_score():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8])
    m_auc = compute_metrics(y, p, binclass('roc-auc'))
    m_ll = compute_metrics(y, p, binclass('log-loss'))
    assert m_auc['score'] == m_auc['roc-auc']
    assert m_ll['score'] == -m_ll['log-loss']


def test_multiclass_hand_computed():
    y = np.array([0, 1, 2, 2])
    p = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.5, 0.3, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )
    m = compute_metrics(y, p, multiclass())
    assert set(m) == {'accuracy', 'log-loss', 'score'}
    assert m['accuracy'] == 0.75
    expected_ll = -(math.log(0.7) + math.log(0.8) + math.log(0.2) + math.log(0.6)) / 4
    assert m['log-loss'] == pytest.approx(expected_ll)
    assert m['score'] == 0.75
    assert compute_metrics(y, p, multiclass('log-loss'))['score'] == pytest.approx(
        -expected_ll
    )


def test_multiclass_argmax_ties_pick_first_class():
    y = np.array([0, 1])
    p = np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], dtype=np.float32)
    assert compute_metrics(y, p, multiclass())['accuracy'] == 0.5
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(p)[None], multiclass())
    assert scores.tolist() == [0.5]


def test_regression_hand_computed():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 5.0, 2.0])
    m = compute_metrics(y, p, REGRESSION)
    assert set(m) == {'rmse', 'score'}
    assert m['rmse'] == pytest.approx(math.sqrt(2.0))
    assert m['score'] == pytest.approx(-math.sqrt(2.0))
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(p)[None], REGRESSION)
    assert scores.item() == pytest.approx(-math.sqrt(2.0), rel=1e-6)


def test_log_loss_clips_extreme_probabilities():
    y = np.array([0, 1])
    p = np.array([1.0, 0.0])  # both confidently wrong
    m = compute_metrics(y, p, binclass('log-loss'))
    assert math.isfinite(m['log-loss'])
    assert m['log-loss'] > 10
    p32 = torch.as_tensor(p, dtype=torch.float32)[None]
    s = score_pack(torch.as_tensor(y), p32, binclass('log-loss'))
    assert torch.isfinite(s).all()
    assert s.item() == pytest.approx(m['score'], rel=1e-5)


def test_single_class_roc_auc_is_nan():
    y = np.ones(6, dtype=np.int64)
    p = np.linspace(0.1, 0.9, 6)
    m = compute_metrics(y, p, binclass('roc-auc'))
    assert math.isnan(m['roc-auc'])
    assert math.isnan(m['score'])
    assert math.isfinite(m['accuracy']) and math.isfinite(m['log-loss'])
    for labels in (np.zeros(6, dtype=np.int64), y):
        s = score_pack(
            torch.as_tensor(labels),
            torch.as_tensor(p, dtype=torch.float32).expand(3, -1),
            binclass('roc-auc'),
        )
        assert s.shape == (3,) and torch.isnan(s).all()


# --------------------------------------------------------------------------------
# The p=0.5 tie rule (round half to even -> class 0)
# --------------------------------------------------------------------------------


def test_half_probability_is_class_zero():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.49, 0.51])
    # 0.5 -> 0: first correct, second wrong.
    assert compute_metrics(y, p, binclass())['accuracy'] == 0.75
    p32 = torch.as_tensor(p, dtype=torch.float32)[None]
    s = score_pack(torch.as_tensor(y), p32, binclass())
    assert s.tolist() == [0.75]
    # All-0.5 predictions: only the negatives are correct.
    s = score_pack(torch.tensor([0, 0, 1]), torch.full((2, 3), 0.5), binclass())
    assert s.tolist() == pytest.approx([2 / 3, 2 / 3])


# --------------------------------------------------------------------------------
# score_pack vs compute_metrics
# --------------------------------------------------------------------------------


@pytest.mark.parametrize('task', ALL_TASKS, ids=lambda t: f'{t.type_}-{t.score}')
@pytest.mark.parametrize('seed', [0, 1])
def test_score_pack_matches_compute_metrics(task: TaskInfo, seed: int):
    y, pred = random_case(task, seed=seed)
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(pred), task)
    assert scores.shape == (len(pred),)
    assert scores.dtype == torch.float32
    expected = [compute_metrics(y, p, task)['score'] for p in pred]
    np.testing.assert_allclose(scores.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_score_pack_accuracy_is_exact():
    task = binclass()
    y, pred = random_case(task, m=8, n=997)
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(pred), task)
    expected = [np.float32(compute_metrics(y, p, task)['accuracy']) for p in pred]
    assert scores.numpy().tolist() == expected


def test_roc_auc_with_ties_matches_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    # Only 4 distinct values -> massive ties.
    pred = rng.integers(0, 4, (6, 300)).astype(np.float32) / 4
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(pred), binclass('roc-auc'))
    expected = [sklearn.metrics.roc_auc_score(y, p) for p in pred]
    np.testing.assert_allclose(scores.numpy(), expected, rtol=1e-6)


def test_score_pack_is_row_independent():
    """Each member's score depends only on its own predictions."""
    for task in ALL_TASKS:
        y, pred = random_case(task, m=4)
        y_t, pred_t = torch.as_tensor(y), torch.as_tensor(pred)
        full = score_pack(y_t, pred_t, task)
        for i in range(len(pred)):
            single = score_pack(y_t, pred_t[i : i + 1], task)
            torch.testing.assert_close(single, full[i : i + 1])


# --------------------------------------------------------------------------------
# Sign convention
# --------------------------------------------------------------------------------


def test_higher_is_better_table():
    assert HIGHER_IS_BETTER == {
        'accuracy': True,
        'roc-auc': True,
        'log-loss': False,
        'rmse': False,
    }


@pytest.mark.parametrize('task', ALL_TASKS, ids=lambda t: f'{t.type_}-{t.score}')
def test_better_predictions_score_higher(task: TaskInfo):
    y, pred = random_case(task, m=1, n=300)
    noisy = torch.as_tensor(pred[0])
    if task.type_ == TaskType.REGRESSION:
        good = torch.as_tensor(y) + 0.1 * (noisy - torch.as_tensor(y))
    elif task.type_ == TaskType.BINCLASS:
        good = 0.8 * torch.as_tensor(y, dtype=torch.float32) + 0.1
        good = 0.7 * good + 0.3 * noisy
    else:
        onehot = torch.nn.functional.one_hot(torch.as_tensor(y), task.n_classes)
        good = 0.7 * onehot.float() + 0.3 * noisy
    stacked = torch.stack([noisy, good.float()])
    scores = score_pack(torch.as_tensor(y), stacked, task)
    assert scores[1] > scores[0]
    m_noisy = compute_metrics(y, stacked[0].numpy(), task)
    m_good = compute_metrics(y, stacked[1].numpy(), task)
    assert m_good['score'] > m_noisy['score']
    if HIGHER_IS_BETTER[task.score]:
        assert m_good['score'] == m_good[task.score]
    else:
        assert m_good['score'] == -m_good[task.score] <= 0


def test_perfect_predictions():
    y = np.array([0, 1, 1, 0])
    p = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert score_pack(torch.as_tensor(y), p, binclass()).item() == 1.0
    assert score_pack(torch.as_tensor(y), p, binclass('roc-auc')).item() == 1.0
    assert score_pack(torch.as_tensor(y), p, binclass('log-loss')).item() == (
        pytest.approx(0.0, abs=1e-6)
    )
    yr = torch.tensor([1.5, -2.0])
    assert score_pack(yr, yr[None], REGRESSION).item() == 0.0


# --------------------------------------------------------------------------------
# dtypes, devices, shapes, make_score_fn
# --------------------------------------------------------------------------------


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize('task', ALL_TASKS, ids=lambda t: f'{t.type_}-{t.score}')
def test_output_is_float32_for_any_input_dtype(task: TaskInfo, dtype: torch.dtype):
    y, pred = random_case(task, m=3, n=50)
    scores = score_pack(torch.as_tensor(y), torch.as_tensor(pred).to(dtype), task)
    assert scores.dtype == torch.float32
    assert scores.shape == (3,)
    assert scores.device == torch.device('cpu')


def test_labels_may_be_float_or_int():
    task = binclass()
    y, pred = random_case(task)
    a = score_pack(torch.as_tensor(y, dtype=torch.int64), torch.as_tensor(pred), task)
    b = score_pack(torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(pred), task)
    torch.testing.assert_close(a, b)


def test_empty_pack():
    for task in ALL_TASKS:
        y, pred = random_case(task, m=1, n=20)
        empty = torch.as_tensor(pred)[:0]
        scores = score_pack(torch.as_tensor(y), empty, task)
        assert scores.shape == (0,) and scores.dtype == torch.float32


def test_shape_errors():
    y = torch.tensor([0, 1, 1])
    with pytest.raises(ValueError):
        score_pack(y, torch.rand(3), binclass())  # missing M
    with pytest.raises(ValueError):
        score_pack(y, torch.rand(2, 4), binclass())  # wrong N
    with pytest.raises(ValueError):
        score_pack(y, torch.rand(2, 3), multiclass())  # missing C
    with pytest.raises(ValueError):
        compute_metrics(y.numpy(), np.random.rand(1, 3), binclass())


def test_unsupported_score_raises():
    y = torch.tensor([0, 1])
    with pytest.raises(ValueError, match='not supported'):
        score_pack(y, torch.rand(1, 2), binclass('rmse'))
    with pytest.raises(ValueError, match='not supported'):
        compute_metrics(y.numpy(), np.random.rand(2, 3), multiclass('roc-auc'))
    with pytest.raises(ValueError, match='not supported'):
        make_score_fn(y.float(), TaskInfo(type_=TaskType.REGRESSION, score='accuracy'))


@pytest.mark.parametrize('task', ALL_TASKS, ids=lambda t: f'{t.type_}-{t.score}')
def test_make_score_fn_matches_score_pack(task: TaskInfo):
    y, pred = random_case(task)
    y_t, pred_t = torch.as_tensor(y), torch.as_tensor(pred)
    score_fn = make_score_fn(y_t, task)
    torch.testing.assert_close(score_fn(pred_t), score_pack(y_t, pred_t, task))
    # Reusable: subsets of members give the matching subset of scores.
    torch.testing.assert_close(
        score_fn(pred_t[1:3]), score_pack(y_t, pred_t, task)[1:3]
    )


@pytest.mark.gpu
@pytest.mark.parametrize('task', ALL_TASKS, ids=lambda t: f'{t.type_}-{t.score}')
def test_cuda_matches_cpu(task: TaskInfo, cuda_device: torch.device):
    y, pred = random_case(task)
    y_t, pred_t = torch.as_tensor(y), torch.as_tensor(pred)
    cpu = score_pack(y_t, pred_t, task)
    gpu = make_score_fn(y_t.to(cuda_device), task)(pred_t.to(cuda_device))
    assert gpu.device.type == 'cuda'
    assert gpu.dtype == torch.float32
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------
# Parity with the official calculate_metrics_pack (skipped without the clone)
# --------------------------------------------------------------------------------


def _official_metrics_torch():
    default = Path(__file__).resolve().parents[1] / '.reference' / 'tabpack'
    path = Path(os.environ.get('TABPACK_REFERENCE_DIR', default))
    if not (path / 'src' / 'project' / 'metrics_torch.py').exists():
        pytest.skip('Official TabPack clone not found (set TABPACK_REFERENCE_DIR)')
    src = str(path / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        return importlib.import_module('project.metrics_torch')
    except ImportError as err:  # pragma: no cover - depends on the environment
        pytest.skip(f'Cannot import the official metrics: {err}')


@pytest.mark.parity
@pytest.mark.parametrize(
    ('task', 'official_score'),
    [
        (binclass('accuracy'), 'accuracy'),
        (binclass('log-loss'), 'cross-entropy'),
        (multiclass('accuracy'), 'accuracy'),
        (multiclass('log-loss'), 'cross-entropy'),
        (REGRESSION, 'rmse'),
    ],
    ids=lambda x: f'{x.type_}-{x.score}' if isinstance(x, TaskInfo) else x,
)
def test_parity_with_official(task: TaskInfo, official_score: str):
    official = _official_metrics_torch()
    lib_data = importlib.import_module('lib.data')
    y, pred = random_case(task, m=6, n=300)
    y_t = torch.as_tensor(y)
    if task.type_ == TaskType.REGRESSION:
        y_t = y_t.float()
    pred_t = torch.as_tensor(pred)
    if task.type_ == TaskType.BINCLASS:
        # Away from 0/1, where torch BCE clamps log at -100 and we clip p instead.
        pred_t = pred_t.clamp(0.01, 0.99)
    expected = official.calculate_metrics_pack(
        y_true=y_t,
        y_pred=pred_t,
        task_type=task.type_.value,
        prediction_type='labels' if task.type_ == TaskType.REGRESSION else 'probs',
        score=lib_data.Score(official_score),
    )['score']
    ours = score_pack(y_t, pred_t, task)
    if task.score == 'accuracy':
        # Bit-exact, including the p=0.5 rounding: greedy selection relies on it.
        assert torch.equal(ours, expected.float())
    else:
        # The official code adds eps=1e-8 inside the log instead of clipping.
        torch.testing.assert_close(ours, expected.float(), rtol=1e-4, atol=1e-5)
