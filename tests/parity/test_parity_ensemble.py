"""Parity of metrics, greedy selection and the online ensemble with the official code.

Compared against the official clone (imported through the ``official`` fixture):

* ``tabpack_repro.metrics.score_pack`` / ``make_score_fn`` vs
  ``project.metrics_torch.calculate_metrics_pack`` / ``project.ensemble_utils_torch
  .make_emsemble_score_fn`` (binclass on probabilities). Accuracy must be bit-identical,
  including the p = 0.5 rounding (half to even -> class 0), because the greedy
  selection compares these scores with ``==`` and ``<=``.
* ``tabpack_repro.ensembles.greedy.greedy_ensemble`` vs the official
  ``greedy_ensemble`` with default options (+ ``max_ensemble_size``): identical
  selected indices on random prediction pools with shared structure, quantized
  members (ties, exact 0.5) and duplicated rows.
* ``tabpack_repro.ensembles.online.OnlineGreedyEnsemble`` vs the official
  ``project.tabpack.OnlineEnsemble`` (type='greedy', update_type='latest',
  include_current_ensemble_in_pool=True, update_part='val', prediction_type='probs'),
  driven with the same simulated training runs: members evaluated every epoch, some
  finishing (moved to the finished pool at their best epoch, with 'train' predictions
  too, like the official trainer), running predictions with 0 rows once every member
  has finished. Every epoch: same acceptance decision, ids, steps, score, is_running,
  and bit-identical stored predictions and averages.

Every random case is generated from a fixed seed, and assertion messages name the
case, so a failure can be reproduced by rerunning the test.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.ensembles.greedy import greedy_ensemble
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.metrics import make_score_fn, score_pack
from tabpack_repro.types import TaskType

CPU = torch.device('cpu')


@pytest.fixture(scope='module')
def ref(official) -> SimpleNamespace:
    """The official modules used in this file."""
    return SimpleNamespace(
        metrics=official('project.metrics_torch'),
        ens=official('project.ensemble_utils_torch'),
        ens_numpy=official('project.ensemble_utils'),
        data=official('lib.data'),
        types=official('lib.types'),
    )


@pytest.fixture(scope='module')
def official_online_ensemble(official) -> type:
    return official('project.tabpack').OnlineEnsemble


def _task(score: str) -> TaskInfo:
    return TaskInfo(type_=TaskType.BINCLASS, score=score, n_classes=2)


def _official_task(ref: SimpleNamespace, labels: dict[str, np.ndarray], score: str):
    return ref.data.Task(
        labels=labels,
        type_=ref.types.TaskType.BINCLASS,
        score=ref.data.Score(score),
    )


def _official_score(ref, y_true: Tensor, y_pred: Tensor, score: str) -> Tensor:
    return ref.metrics.calculate_metrics_pack(
        y_true=y_true,
        y_pred=y_pred,
        task_type='binclass',
        prediction_type='probs',
        score=ref.data.Score(score),
    )['score']


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _labels(rng: np.random.Generator, signal: np.ndarray) -> np.ndarray:
    return (rng.random(signal.shape) < _sigmoid(2.0 * signal)).astype(np.int64)


def _quantize(probs: np.ndarray, levels: int) -> np.ndarray:
    # Multiples of 1/levels (levels even): exact 0.5 predictions and many ties.
    return np.round(probs * levels) / levels


# ---------------------------------------------------------------------------
# (1) Metrics
# ---------------------------------------------------------------------------


def _tie_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    """Rows that probe the binclass rounding at and around p = 0.5 (float32)."""
    half = np.float32(0.5)
    below = np.nextafter(half, np.float32(0))
    above = np.nextafter(half, np.float32(1))
    mixed = rng.choice(np.array([0.0, below, half, above, 1.0], dtype=np.float32), n)
    return np.stack(
        [
            np.full(n, half),
            np.full(n, below),
            np.full(n, above),
            np.zeros(n),
            np.ones(n),
            mixed,
        ]
    ).astype(np.float32)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64], ids=str)
def test_accuracy_matches_official_bitwise_including_half_ties(ref, dtype):
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 700))
        signal = rng.standard_normal(n)
        y = _labels(rng, signal)
        logits = signal + rng.standard_normal((12, n))
        probs = _sigmoid(logits)
        probs[::2] = _quantize(probs[::2], int(rng.choice([2, 4, 8, 16])))
        probs = np.concatenate([probs, _tie_rows(n, rng)])
        y_true = torch.as_tensor(y)
        y_pred = torch.as_tensor(probs, dtype=dtype)

        expected = _official_score(ref, y_true, y_pred, 'accuracy')
        ours = score_pack(y_true, y_pred, _task('accuracy'))
        assert expected.dtype == ours.dtype == torch.float32
        assert torch.equal(ours, expected), f'seed={seed} n={n} dtype={dtype}'


def test_accuracy_rounds_half_to_class_zero_like_official(ref):
    y_pred = torch.full((1, 10), 0.5)
    for label, accuracy in [(0, 1.0), (1, 0.0)]:
        y_true = torch.full((10,), label)
        expected = _official_score(ref, y_true, y_pred, 'accuracy')
        assert expected.tolist() == [accuracy]
        assert score_pack(y_true, y_pred, _task('accuracy')).tolist() == [accuracy]


def test_accuracy_score_fn_matches_official_score_fn(ref):
    """The score functions used for greedy / online ensembling on 'val'."""
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(300, 600))
        signal = rng.standard_normal(n)
        labels = {part: _labels(rng, signal) for part in ('train', 'val', 'test')}
        official_fn = ref.ens.make_emsemble_score_fn(
            _official_task(ref, labels, 'accuracy'), 'probs', part='val', device=CPU
        )
        our_fn = make_score_fn(torch.as_tensor(labels['val']), _task('accuracy'))
        probs = _quantize(_sigmoid(signal + rng.standard_normal((25, n))), 16)
        y_pred = torch.as_tensor(probs, dtype=torch.float32)
        assert torch.equal(our_fn(y_pred), official_fn(y_pred)), f'seed={seed}'
        # A single candidate ensemble, as scored by the online ensemble.
        assert torch.equal(our_fn(y_pred[:1]), official_fn(y_pred[:1]))


def test_roc_auc_close_to_official(ref):
    # Tie-free predictions: the official trapezoid ROC-AUC ignores ties (a08 report).
    for seed in range(10):
        rng = np.random.default_rng(2000 + seed)
        n = int(rng.integers(100, 800))
        signal = rng.standard_normal(n)
        y_true = torch.as_tensor(_labels(rng, signal))
        y_pred = torch.as_tensor(
            _sigmoid(signal + rng.standard_normal((8, n))), dtype=torch.float32
        )
        expected = _official_score(ref, y_true, y_pred, 'roc-auc')
        ours = score_pack(y_true, y_pred, _task('roc-auc'))
        torch.testing.assert_close(ours, expected, rtol=0, atol=2e-6)


def test_log_loss_close_to_official_cross_entropy(ref):
    # Away from p in {0, 1}: torch BCE clamps log at -100, we clip p at float32 eps.
    for seed in range(10):
        rng = np.random.default_rng(3000 + seed)
        n = int(rng.integers(100, 800))
        signal = rng.standard_normal(n)
        y_true = torch.as_tensor(_labels(rng, signal))
        y_pred = torch.as_tensor(
            _sigmoid(signal + rng.standard_normal((8, n))), dtype=torch.float32
        ).clamp(0.01, 0.99)
        expected = _official_score(ref, y_true, y_pred, 'cross-entropy')
        ours = score_pack(y_true, y_pred, _task('log-loss'))
        torch.testing.assert_close(ours, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (2) Greedy selection
# ---------------------------------------------------------------------------


def _greedy_pool(rng: np.random.Generator, m: int, n: int) -> tuple[np.ndarray, Tensor]:
    """Labels (N,) and M correlated binclass predictions (M, N) float32.

    All members share a signal and a common error term, so ensembles of a few good
    members beat the best member but not everything is worth adding. Some members are
    quantized (ties, exact 0.5) and some rows are exact duplicates.
    """
    signal = rng.standard_normal(n)
    y = _labels(rng, signal)
    common = rng.standard_normal(n)
    strength = rng.uniform(0.2, 2.0, (m, 1))
    noise = rng.uniform(0.3, 2.0, (m, 1))
    bias = rng.normal(0.0, 0.3, (m, 1))
    logits = (
        strength * signal + 0.6 * common + noise * rng.standard_normal((m, n)) + bias
    )
    probs = _sigmoid(logits)
    quantized = rng.random(m) < 0.3
    probs[quantized] = _quantize(probs[quantized], int(rng.choice([4, 8, 16])))
    n_duplicates = int(rng.integers(0, m // 4 + 1))
    if n_duplicates:
        target = rng.choice(m, n_duplicates, replace=False)
        probs[target] = probs[rng.integers(0, m, n_duplicates)]
    return y, torch.as_tensor(probs, dtype=torch.float32)


def _brier_fn(y: np.ndarray, levels: int | None):
    """A continuous (optionally quantized) score, shared by both implementations."""
    y_float = torch.as_tensor(y, dtype=torch.float32)

    def score_fn(y_pred: Tensor) -> Tensor:
        score = -(y_pred - y_float).square().mean(-1)
        return score if levels is None else torch.round(score * levels) / levels

    return score_fn


def _greedy_score_fns(ref, score: str, y: np.ndarray):
    """(official score_fn, our score_fn) on the labels y."""
    if score == 'accuracy':
        official_fn = ref.ens.make_emsemble_score_fn(
            _official_task(ref, {'val': y}, 'accuracy'), 'probs', part='val', device=CPU
        )
        return official_fn, make_score_fn(torch.as_tensor(y), _task('accuracy'))
    shared = _brier_fn(y, None if score == 'brier' else 500)
    return shared, shared


class _TieBreakCounter:
    """Wraps the official score_fn and counts the official greedy's tie-breaks.

    Official call order: individual scores, the initial ensemble score, then one call
    per step with the candidate scores. A tie-break is a step whose best candidate
    score improves the ensemble and is reached by more than one candidate.
    """

    def __init__(self, score_fn) -> None:
        self._score_fn = score_fn
        self.calls: list[Tensor] = []

    def __call__(self, y_pred: Tensor) -> Tensor:
        scores = self._score_fn(y_pred)
        self.calls.append(scores)
        return scores

    def n_tie_breaks(self) -> int:
        current = self.calls[1][0]
        count = 0
        for scores in self.calls[2:]:
            if scores.numel() == 0:
                break
            best = scores.max()
            if best <= current:
                break
            count += int((scores == best).sum() > 1)
            current = best
        return count


GREEDY_CASES_PER_SCORE = 150
GREEDY_SIZE_LIMITS = (1, 2, 3, 5, 8, 'M', 'M+3')


@pytest.mark.parametrize('score', ['accuracy', 'brier', 'brier-quantized'])
def test_greedy_ensemble_matches_official(ref, score):
    stats: Counter[str] = Counter()
    for case in range(GREEDY_CASES_PER_SCORE):
        seed = 10_000 * (1 + ['accuracy', 'brier', 'brier-quantized'].index(score))
        rng = np.random.default_rng(seed + case)
        m = int(rng.integers(2, 41))
        n = int(rng.integers(450, 551))
        y, predictions = _greedy_pool(rng, m, n)
        official_fn, our_fn = _greedy_score_fns(ref, score, y)
        limit = GREEDY_SIZE_LIMITS[int(rng.integers(len(GREEDY_SIZE_LIMITS)))]
        limit = {'M': m, 'M+3': m + 3}.get(limit, limit)

        for max_ensemble_size in (None, limit):
            counter = _TieBreakCounter(official_fn)
            official_kwargs = (
                {} if max_ensemble_size is None else {'max_ensemble_size': limit}
            )
            expected, weights = ref.ens.greedy_ensemble(
                predictions, score_fn=counter, **official_kwargs
            )
            ours = greedy_ensemble(
                predictions, score_fn=our_fn, max_ensemble_size=max_ensemble_size
            )
            assert weights is None
            assert ours.dtype == torch.int64
            assert torch.equal(ours, expected), (
                f'score={score} case={case} (seed {seed + case}) m={m} n={n} '
                f'max_ensemble_size={max_ensemble_size}: '
                f'ours={ours.tolist()} official={expected.tolist()}'
            )
            stats['cases'] += 1
            stats['size>1'] += len(ours) > 1
            stats['size<m'] += len(ours) < m
            stats['tie_break_cases'] += counter.n_tie_breaks() > 0

    # The pools must make the comparison meaningful: non-trivial selections that are
    # neither a single member nor everything, and real tie-breaks between equally
    # good candidates (for the continuous Brier score, from duplicated rows).
    assert stats['cases'] == 2 * GREEDY_CASES_PER_SCORE
    assert stats['size>1'] >= 0.5 * stats['cases'], stats
    assert stats['size<m'] >= 0.5 * stats['cases'], stats
    assert stats['tie_break_cases'] >= 0.1 * stats['cases'], stats


# ---------------------------------------------------------------------------
# (3) Online ensemble
# ---------------------------------------------------------------------------

EVAL_PARTS = ('val', 'test')
FINAL_PARTS = ('train', 'val', 'test')
N_EPOCHS = 20
EPOCH_SIZE = 7


def _simulate_run(rng: np.random.Generator) -> tuple[dict, list[dict]]:
    """A simulated TabPack run: labels per part and the per-epoch ensemble inputs.

    Mirrors the official epoch loop: all running members are evaluated on val/test;
    members that stop are removed from the running predictions and added to the
    finished pool with their best-epoch step and train/val/test predictions. Every
    member stops at the latest at the last epoch in about half of the runs, so the last
    update then sees 0 running members.
    """
    k = int(rng.integers(3, 9))
    sizes = {'train': 200, 'val': int(rng.integers(250, 351)), 'test': 150}
    signal = {part: rng.standard_normal(size) for part, size in sizes.items()}
    labels = {part: _labels(rng, signal[part]) for part in FINAL_PARTS}
    common = {part: rng.standard_normal(size) for part, size in sizes.items()}

    peak = rng.uniform(0.5, 2.5, k)
    tau = rng.uniform(1.0, 6.0, k)
    overfit = rng.uniform(0.0, 0.15, k)
    noise = rng.uniform(0.5, 1.5, k)
    quantized = rng.random(k) < 0.4
    member_noise = {
        part: rng.standard_normal((k, size)) for part, size in sizes.items()
    }
    last = N_EPOCHS - 1
    stop_epoch = rng.integers(3, N_EPOCHS + 4, k)
    if rng.random() < 0.5:
        stop_epoch = np.minimum(stop_epoch, last)

    def member_predictions(i: int, epoch: int, part: str) -> np.ndarray:
        quality = peak[i] * (1.0 - np.exp(-(epoch + 1) / tau[i]))
        quality -= overfit[i] * max(0, epoch - 8)
        drift = 0.3 * rng.standard_normal(sizes[part])
        logits = quality * signal[part] + 0.5 * common[part]
        logits = logits + noise[i] * member_noise[part][i] + drift
        probs = _sigmoid(logits)
        if quantized[i]:
            probs = _quantize(probs, 8)
        return probs.astype(np.float32)

    history: dict[int, list[tuple[int, float, dict[str, np.ndarray]]]] = {}
    finished_ids: list[int] = []
    finished_steps: list[int] = []
    finished_predictions: dict[str, list[np.ndarray]] = {p: [] for p in FINAL_PARTS}
    running = list(range(k))
    epochs = []
    for epoch in range(N_EPOCHS):
        step = (epoch + 1) * EPOCH_SIZE
        latest = {}
        for i in running:
            predictions = {p: member_predictions(i, epoch, p) for p in FINAL_PARTS}
            val_accuracy = float(np.mean(np.round(predictions['val']) == labels['val']))
            history.setdefault(i, []).append((step, val_accuracy, predictions))
            latest[i] = predictions

        stopped = [i for i in running if stop_epoch[i] <= epoch]
        running = [i for i in running if i not in stopped]
        for i in stopped:
            best_step, _, best_predictions = max(history[i], key=lambda x: x[1])
            finished_ids.append(i)
            finished_steps.append(best_step)
            for part in FINAL_PARTS:
                finished_predictions[part].append(best_predictions[part])

        epochs.append(
            {
                'running_ids': np.array(running, dtype=np.int64),
                'running_steps': np.full(len(running), step, dtype=np.int64),
                'running_latest_predictions': {
                    part: np.array(
                        [latest[i][part] for i in running], dtype=np.float32
                    ).reshape(len(running), sizes[part])
                    for part in EVAL_PARTS
                },
                'finished_ids': np.array(finished_ids, dtype=np.int64),
                'finished_steps': np.array(finished_steps, dtype=np.int64),
                'finished_predictions': (
                    {p: np.stack(v) for p, v in finished_predictions.items()}
                    if finished_ids
                    else {}
                ),
            }
        )
        if not running:
            break
    return labels, epochs


def _official_update_kwargs(inputs: dict) -> dict:
    """Keyword arguments of the official OnlineEnsemble.update (own tensor copies)."""

    def torch_copy(predictions: dict[str, np.ndarray]) -> dict[str, Tensor]:
        return {k: torch.from_numpy(v.copy()) for k, v in predictions.items()}

    return {
        'running_ids': inputs['running_ids'].copy(),
        'running_steps': inputs['running_steps'].copy(),
        'running_latest_predictions': {
            k: v.copy() for k, v in inputs['running_latest_predictions'].items()
        },
        'running_latest_predictions_torch': torch_copy(
            inputs['running_latest_predictions']
        ),
        'running_best_predictions': {},
        'running_best_predictions_torch': {},
        'finished_ids': inputs['finished_ids'].copy(),
        'finished_steps': inputs['finished_steps'].copy(),
        'finished_predictions': {
            k: v.copy() for k, v in inputs['finished_predictions'].items()
        },
        'finished_predictions_torch': torch_copy(inputs['finished_predictions']),
    }


def _our_update_kwargs(inputs: dict) -> dict:
    return {
        'running_ids': inputs['running_ids'].copy(),
        'running_steps': inputs['running_steps'].copy(),
        'running_predictions': {
            k: torch.from_numpy(v.copy())
            for k, v in inputs['running_latest_predictions'].items()
        },
        'finished_ids': inputs['finished_ids'].copy(),
        'finished_steps': inputs['finished_steps'].copy(),
        'finished_predictions': {
            k: torch.from_numpy(v.copy())
            for k, v in inputs['finished_predictions'].items()
        },
    }


def _online_score_fns(ref, score: str, labels: dict[str, np.ndarray]):
    if score == 'accuracy':
        official_fn = ref.ens.make_emsemble_score_fn(
            _official_task(ref, labels, 'accuracy'), 'probs', part='val', device=CPU
        )
        return official_fn, make_score_fn(torch.as_tensor(labels['val']), _task(score))
    shared = _brier_fn(labels['val'], None)
    return shared, shared


def _stored_predictions(ensemble: OnlineGreedyEnsemble) -> dict[str, Tensor]:
    # The per-entry snapshots (E, N) behind the averages; not part of the public API.
    return ensemble._predictions


def _check_report(ref, ours: OnlineGreedyEnsemble, official, labels, where: str):
    """Our report vs the official update_online_ensembles report of the same state."""
    official_task = _official_task(ref, labels, 'accuracy')
    official_means = {
        k: ref.ens_numpy.compute_ensemble_prediction(
            v, None, ref.types.PredictionType.PROBS
        )
        for k, v in official._predictions.items()
    }
    expected = official_task.calculate_metrics(official_means, 'probs')
    report = ours.report(labels)
    assert report['ids'] == official.ids.tolist(), where
    assert report['steps'] == official.steps.tolist(), where
    assert report['score_val'] == official._score, where
    assert set(report['metrics']) == set(expected), where
    for part, metrics in report['metrics'].items():
        assert metrics['accuracy'] == expected[part]['accuracy'], (where, part)
        assert metrics['score'] == expected[part]['score'], (where, part)
        assert metrics['roc-auc'] == pytest.approx(
            expected[part]['roc-auc'], rel=1e-12
        ), (where, part)
        assert metrics['log-loss'] == pytest.approx(
            expected[part]['cross-entropy'], rel=1e-5
        ), (where, part)


ONLINE_RUNS_PER_SCORE = 30
MAX_ENSEMBLE_SIZES = (None, 3, 8)
PATIENCES = (0, 1, 2, 4, 50)


@pytest.mark.parametrize('score', ['accuracy', 'brier'])
def test_online_ensemble_matches_official(ref, official_online_ensemble, score):
    stats: Counter[str] = Counter()
    for run in range(ONLINE_RUNS_PER_SCORE):
        seed = 50_000 + 1000 * ['accuracy', 'brier'].index(score) + run
        rng = np.random.default_rng(seed)
        labels, epochs = _simulate_run(rng)
        max_ensemble_size = MAX_ENSEMBLE_SIZES[run % len(MAX_ENSEMBLE_SIZES)]
        patience = PATIENCES[run % len(PATIENCES)]
        official_fn, our_fn = _online_score_fns(ref, score, labels)

        official = official_online_ensemble(
            type='greedy',
            # A fresh dict: the official class adds 'score_fn' to it.
            options=(
                {}
                if max_ensemble_size is None
                else {'max_ensemble_size': max_ensemble_size}
            ),
            update_type='latest',
            update_part='val',
            include_current_ensemble_in_pool=True,
            prediction_type=ref.types.PredictionType.PROBS,
            score_fn=official_fn,
            patience=patience,
        )
        ours = OnlineGreedyEnsemble(
            score_fn=our_fn,
            task=_task('accuracy'),
            max_ensemble_size=max_ensemble_size,
            patience=patience,
        )

        for epoch, inputs in enumerate(epochs):
            where = (
                f'score={score} run={run} (seed {seed}) epoch={epoch} '
                f'max_ensemble_size={max_ensemble_size} patience={patience}'
            )
            assert ours.is_running == official.is_running, where
            if not official.is_running:
                stats['stopped_runs'] += 1
                break
            expected_improved = official.update(**_official_update_kwargs(inputs))
            improved = ours.update(**_our_update_kwargs(inputs))

            assert improved == expected_improved, where
            assert np.array_equal(ours.ids, official.ids), (
                f'{where}: ids ours={ours.ids.tolist()} '
                f'official={official.ids.tolist()}'
            )
            assert np.array_equal(ours.steps, official.steps), where
            assert ours.ids.dtype == ours.steps.dtype == np.int64, where
            assert ours.score == official._score, where
            assert ours.is_running == official.is_running, where

            stored = _stored_predictions(ours)
            assert set(stored) == set(official._predictions_torch), where
            averages = ours.predictions()
            for part, expected in official._predictions_torch.items():
                assert torch.equal(stored[part], expected), (where, part)
                assert np.array_equal(
                    stored[part].numpy(), official._predictions[part]
                ), (where, part)
                assert torch.equal(
                    averages[part],
                    ref.ens.compute_ensemble_prediction(expected, None),
                ), (where, part)

            stats['updates'] += 1
            stats['improved'] += improved
            stats['repeated_ids'] += len(np.unique(ours.ids)) < len(ours.ids)
            stats['with_finished'] += len(inputs['finished_ids']) > 0
            stats['no_running'] += len(inputs['running_ids']) == 0
            stats['size>1'] += len(ours.ids) > 1

        # The report of the final ensemble (sklearn metrics are slow: once per run).
        _check_report(
            ref, ours, official, labels, f'score={score} run={run} (seed {seed})'
        )

    # The simulated runs must cover the interesting situations.
    assert stats['updates'] >= 8 * ONLINE_RUNS_PER_SCORE, stats
    for key in ('improved', 'repeated_ids', 'with_finished', 'no_running', 'size>1'):
        assert stats[key] > 0, (key, stats)
    assert stats['improved'] < stats['updates'], stats
    assert stats['stopped_runs'] > 0, stats
