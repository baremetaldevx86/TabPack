"""Tests for batched pack inference (a21): logits_to_predictions, predict_pack,
evaluate_pack."""

from __future__ import annotations

import contextlib
import dataclasses
import logging

import pytest
import torch
from _helpers import make_synthetic_dataset
from torch import Tensor, nn

from tabpack_repro.data.dataset import TaskInfo
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.metrics import make_score_fn
from tabpack_repro.training.evaluate import (
    evaluate_pack,
    logits_to_predictions,
    predict_pack,
)
from tabpack_repro.types import TaskType

K = 3
N_CLASSES = 4


# ---------------------------------------------------------------------------
# Fixtures: a tiny fake pack (the contract of ModelPack.forward) and datasets
# ---------------------------------------------------------------------------


class FakePack(nn.Module):
    """K linear models over [x_num, one_hot(x_cat)] with ModelPack's interface.

    Takes shared (B, f) inputs and returns (K, B) logits, or (K, B, C) when
    n_out is given. Records every call and can simulate out-of-memory errors.
    """

    def __init__(
        self,
        n_num: int,
        cardinalities: list[int],
        *,
        pack_size: int = K,
        n_out: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        d_in = n_num + sum(cardinalities)
        self.cardinalities = cardinalities
        self.n_out = n_out
        self.weight = nn.Parameter(
            torch.randn(pack_size, d_in, n_out or 1, generator=generator)
        )
        self.bias = nn.Parameter(torch.randn(pack_size, 1, n_out or 1))
        # Dropout makes train mode observable (random outputs).
        self.dropout = nn.Dropout(0.5)
        self.calls: list[dict] = []
        self.oom_above: int | None = None
        self.oom_error: type[BaseException] | None = None

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        batch = x_num if x_num is not None else x_cat
        self.calls.append(
            {
                'batch_size': batch.shape[0],
                'training': self.training,
                'x_num': x_num,
                'x_cat': x_cat,
                'grad_enabled': torch.is_grad_enabled(),
                'inference_mode': torch.is_inference_mode_enabled(),
                'autocast': torch.is_autocast_enabled('cpu'),
            }
        )
        if self.oom_above is not None and batch.shape[0] > self.oom_above:
            raise (self.oom_error or torch.OutOfMemoryError)(
                'CUDA out of memory. Tried to allocate 1 GiB (fake)'
            )
        features = []
        if x_num is not None:
            features.append(x_num)
        if x_cat is not None:
            features.extend(
                nn.functional.one_hot(x_cat[:, i], c).float()
                for i, c in enumerate(self.cardinalities)
            )
        x = self.dropout(torch.cat(features, dim=-1))
        out = torch.einsum('bf,kfo->kbo', x, self.weight) + self.bias
        return out.squeeze(-1) if self.n_out is None else out


def _with_task(dataset: PreparedDataset, task_type: TaskType) -> PreparedDataset:
    if task_type == TaskType.BINCLASS:
        return dataset
    y = {}
    for part, labels in dataset.y.items():
        if task_type == TaskType.MULTICLASS:
            y[part] = torch.arange(len(labels)) % N_CLASSES
        else:
            y[part] = labels.float() * 2.0 - 0.5
    task = (
        TaskInfo(type_=task_type, score='accuracy', n_classes=N_CLASSES)
        if task_type == TaskType.MULTICLASS
        else TaskInfo(type_=task_type, score='rmse', n_classes=None)
    )
    return dataclasses.replace(dataset, y=y, task=task)


def _fake_for(dataset: PreparedDataset, **kwargs) -> FakePack:
    n_out = N_CLASSES if dataset.task.type_ == TaskType.MULTICLASS else None
    return FakePack(
        dataset.n_num_features, dataset.cat_cardinalities, n_out=n_out, **kwargs
    )


def _expected_shape(dataset: PreparedDataset, n: int) -> tuple[int, ...]:
    return (K, n, N_CLASSES) if dataset.task.type_ == TaskType.MULTICLASS else (K, n)


@pytest.fixture
def dataset() -> PreparedDataset:
    return make_synthetic_dataset(n_train=97, n_val=41, n_test=23)


class CountingContext:
    """A reusable context manager that counts how often it is entered."""

    def __init__(self) -> None:
        self.enters = 0
        self.active = False

    def __enter__(self):
        assert not self.active
        self.enters += 1
        self.active = True
        return self

    def __exit__(self, *exc) -> None:
        self.active = False


# ---------------------------------------------------------------------------
# logits_to_predictions
# ---------------------------------------------------------------------------


def test_logits_to_predictions_binclass_is_sigmoid() -> None:
    logits = torch.randn(K, 10)
    predictions = logits_to_predictions(logits, 'binclass')
    assert predictions.shape == (K, 10)
    assert predictions.dtype == torch.float32
    torch.testing.assert_close(predictions, torch.sigmoid(logits), rtol=0, atol=0)


def test_logits_to_predictions_multiclass_is_softmax() -> None:
    logits = torch.randn(K, 10, N_CLASSES)
    predictions = logits_to_predictions(logits, TaskType.MULTICLASS)
    assert predictions.shape == (K, 10, N_CLASSES)
    assert predictions.dtype == torch.float32
    torch.testing.assert_close(predictions, logits.softmax(-1), rtol=0, atol=0)
    torch.testing.assert_close(predictions.sum(-1), torch.ones(K, 10))


def test_logits_to_predictions_regression_is_identity() -> None:
    logits = torch.randn(K, 10)
    predictions = logits_to_predictions(logits, 'regression')
    assert predictions.dtype == torch.float32
    torch.testing.assert_close(predictions, logits, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize('task_type', list(TaskType))
def test_logits_to_predictions_casts_to_float32_before_activation(
    dtype: torch.dtype, task_type: TaskType
) -> None:
    shape = (K, 10, N_CLASSES) if task_type == TaskType.MULTICLASS else (K, 10)
    logits = (torch.randn(shape) * 5).to(dtype)
    predictions = logits_to_predictions(logits, task_type)
    assert predictions.dtype == torch.float32
    # The activation runs in float32 on the float32 logits (not in low precision).
    expected = logits_to_predictions(logits.float(), task_type)
    torch.testing.assert_close(predictions, expected, rtol=0, atol=0)


def test_logits_to_predictions_extreme_logits_are_finite() -> None:
    logits = torch.tensor([[-1e4, 0.0, 1e4]], dtype=torch.bfloat16)
    predictions = logits_to_predictions(logits, 'binclass')
    assert torch.equal(predictions, torch.tensor([[0.0, 0.5, 1.0]]))
    probs = logits_to_predictions(logits[:, None, :] * 10, 'multiclass')
    assert torch.isfinite(probs).all()


def test_logits_to_predictions_unknown_task_type() -> None:
    with pytest.raises(ValueError):
        logits_to_predictions(torch.zeros(K, 2), 'ranking')


# ---------------------------------------------------------------------------
# predict_pack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('task_type', list(TaskType))
def test_predict_pack_shape_dtype_and_values(
    dataset: PreparedDataset, task_type: TaskType
) -> None:
    dataset = _with_task(dataset, task_type)
    model = _fake_for(dataset)
    for part in ('train', 'val', 'test'):
        predictions = predict_pack(model, dataset, part, batch_size=10)
        n = dataset.size(part)
        assert predictions.shape == _expected_shape(dataset, n)
        assert predictions.dtype == torch.float32
        assert predictions.device == model.weight.device
        assert not predictions.requires_grad
        # Same as one full-size forward in eval mode.
        with torch.no_grad():
            model.eval()
            logits = model(dataset.x_num[part], dataset.x_cat[part])
            model.train()
        expected = logits_to_predictions(logits, task_type)
        torch.testing.assert_close(predictions, expected, rtol=1e-6, atol=1e-6)
        if task_type == TaskType.BINCLASS:
            assert ((predictions >= 0) & (predictions <= 1)).all()
        if task_type == TaskType.MULTICLASS:
            torch.testing.assert_close(predictions.sum(-1), torch.ones(K, n))


@pytest.mark.parametrize('batch_size', [1, 2, 7, 40, 41, 42, 32768])
def test_predict_pack_batching_does_not_change_predictions(
    dataset: PreparedDataset, batch_size: int
) -> None:
    model = _fake_for(dataset)
    one_batch = predict_pack(model, dataset, 'val', batch_size=dataset.size('val'))
    batched = predict_pack(model, dataset, 'val', batch_size=batch_size)
    torch.testing.assert_close(batched, one_batch, rtol=1e-6, atol=1e-6)
    n = dataset.size('val')
    expected_sizes = [min(batch_size, n - s) for s in range(0, n, batch_size)]
    assert [c['batch_size'] for c in model.calls[-len(expected_sizes) :]] == (
        expected_sizes
    )


def test_predict_pack_batching_multiclass(dataset: PreparedDataset) -> None:
    dataset = _with_task(dataset, TaskType.MULTICLASS)
    model = _fake_for(dataset)
    one_batch = predict_pack(model, dataset, 'train', batch_size=10**6)
    batched = predict_pack(model, dataset, 'train', batch_size=8)
    torch.testing.assert_close(batched, one_batch, rtol=1e-6, atol=1e-6)


def test_predict_pack_passes_shared_2d_slices(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    predict_pack(model, dataset, 'train', batch_size=30)
    x_num_part = dataset.x_num['train']
    x_cat_part = dataset.x_cat['train']
    starts = [0, 30, 60, 90]
    assert len(model.calls) == len(starts)
    for call, start in zip(model.calls, starts, strict=True):
        x_num, x_cat = call['x_num'], call['x_cat']
        # Shared (B, f) inputs, not per-member (K, B, f) copies.
        assert x_num.ndim == 2 and x_cat.ndim == 2
        assert x_num.shape[1] == dataset.n_num_features
        assert x_cat.shape[1] == dataset.n_cat_features
        # Views into the dataset tensors (no gather/copy on the same device).
        assert x_num.data_ptr() == x_num_part[start].data_ptr()
        assert x_cat.data_ptr() == x_cat_part[start].data_ptr()
        assert x_cat.dtype == torch.int64


def test_predict_pack_without_categorical_features() -> None:
    dataset = make_synthetic_dataset(
        n_train=50, n_val=20, n_test=20, cat_cardinalities=()
    )
    assert dataset.x_cat is None
    model = _fake_for(dataset)
    predictions = predict_pack(model, dataset, 'val', batch_size=6)
    assert predictions.shape == (K, 20)
    assert all(call['x_cat'] is None for call in model.calls)


def test_predict_pack_without_numerical_features(dataset: PreparedDataset) -> None:
    dataset = dataclasses.replace(dataset, x_num=None)
    model = _fake_for(dataset)
    predictions = predict_pack(model, dataset, 'val', batch_size=6)
    assert predictions.shape == (K, dataset.size('val'))
    assert all(call['x_num'] is None for call in model.calls)


def test_predict_pack_empty_part(dataset: PreparedDataset) -> None:
    empty = {part: t[:0] for part, t in dataset.y.items()}
    dataset = dataclasses.replace(
        dataset,
        x_num={part: t[:0] for part, t in dataset.x_num.items()},
        x_cat={part: t[:0] for part, t in dataset.x_cat.items()},
        y=empty,
    )
    model = _fake_for(dataset)
    predictions = predict_pack(model, dataset, 'test', batch_size=4)
    assert predictions.shape == (K, 0)
    assert predictions.dtype == torch.float32


@pytest.mark.parametrize('initially_training', [True, False])
def test_predict_pack_uses_eval_mode_and_restores_mode(
    dataset: PreparedDataset, initially_training: bool
) -> None:
    model = _fake_for(dataset)
    model.train(initially_training)
    first = predict_pack(model, dataset, 'val', batch_size=16)
    second = predict_pack(model, dataset, 'val', batch_size=16)
    # Dropout is off during inference: two calls give identical predictions.
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.calls and all(not call['training'] for call in model.calls)
    assert all(m.training == initially_training for m in model.modules())


def test_predict_pack_restores_mixed_submodule_modes(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    model.train()
    model.dropout.eval()
    predict_pack(model, dataset, 'val')
    assert model.training
    assert not model.dropout.training


def test_predict_pack_restores_mode_after_an_error(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    model.train()
    model.oom_above = 0
    model.oom_error = ValueError
    with pytest.raises(ValueError):
        predict_pack(model, dataset, 'val')
    assert model.training and model.dropout.training


def test_predict_pack_runs_in_inference_mode(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    predictions = predict_pack(model, dataset, 'val')
    assert all(c['inference_mode'] and not c['grad_enabled'] for c in model.calls)
    assert predictions.is_inference()
    # Inference mode does not leak out of the call.
    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()


def test_predict_pack_enters_autocast_around_every_forward(
    dataset: PreparedDataset,
) -> None:
    model = _fake_for(dataset)
    autocast = CountingContext()
    predict_pack(model, dataset, 'train', batch_size=25, autocast=autocast)
    assert autocast.enters == 4 == len(model.calls)
    assert not autocast.active


@pytest.mark.parametrize('task_type', list(TaskType))
def test_predict_pack_with_cpu_bfloat16_autocast(
    dataset: PreparedDataset, task_type: TaskType
) -> None:
    dataset = _with_task(dataset, task_type)
    model = _fake_for(dataset)
    autocast = torch.autocast('cpu', dtype=torch.bfloat16)
    predictions = predict_pack(model, dataset, 'val', batch_size=16, autocast=autocast)
    assert all(call['autocast'] for call in model.calls)
    assert not torch.is_autocast_enabled('cpu')
    assert predictions.dtype == torch.float32
    reference = predict_pack(model, dataset, 'val')
    torch.testing.assert_close(predictions, reference, rtol=0.05, atol=0.05)


@pytest.mark.parametrize('batch_size', [0, -1])
def test_predict_pack_rejects_non_positive_batch_size(
    dataset: PreparedDataset, batch_size: int
) -> None:
    with pytest.raises(ValueError, match='batch_size'):
        predict_pack(_fake_for(dataset), dataset, 'val', batch_size=batch_size)


@pytest.mark.parametrize('batch_size', [2.0, True])
def test_predict_pack_rejects_non_int_batch_size(
    dataset: PreparedDataset, batch_size
) -> None:
    with pytest.raises(TypeError, match='batch_size'):
        predict_pack(_fake_for(dataset), dataset, 'val', batch_size=batch_size)


# ---------------------------------------------------------------------------
# Out-of-memory fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError, RuntimeError])
def test_predict_pack_halves_batch_size_on_oom(
    dataset: PreparedDataset, error: type[BaseException], caplog
) -> None:
    model = _fake_for(dataset)
    reference = predict_pack(model, dataset, 'train', batch_size=97)
    model.calls.clear()
    model.oom_above = 20
    # RuntimeError is recognized by its message ('CUDA out of memory').
    model.oom_error = error
    with caplog.at_level(logging.WARNING):
        predictions = predict_pack(model, dataset, 'train', batch_size=64)
    torch.testing.assert_close(predictions, reference, rtol=1e-6, atol=1e-6)
    sizes = [call['batch_size'] for call in model.calls]
    # 64 fails, 32 fails, 16 succeeds and is kept for the rest of the part.
    assert sizes == [64, 32, 16, 16, 16, 16, 16, 16, 1]
    assert 'batch_size=16' in caplog.text


def test_predict_pack_oom_resumes_from_the_failed_batch(
    dataset: PreparedDataset,
) -> None:
    model = _fake_for(dataset)
    reference = predict_pack(model, dataset, 'train', batch_size=97)
    model.calls.clear()
    # N=97, batch_size=40: the first batch [0, 40) succeeds, the second forward
    # (rows [40, 80)) raises OOM once.
    original_forward = model.forward
    n_calls = 0

    def forward(x_num, x_cat):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 2:
            raise torch.OutOfMemoryError('CUDA out of memory (fake)')
        return original_forward(x_num, x_cat)

    model.forward = forward
    predictions = predict_pack(model, dataset, 'train', batch_size=40)
    torch.testing.assert_close(predictions, reference, rtol=1e-6, atol=1e-6)
    # Batch 1 (40 rows) is kept; the failed batch restarts at row 40 with 20 rows.
    assert [c['batch_size'] for c in model.calls] == [40, 20, 20, 17]
    starts = [0, 40, 60, 80]
    for call, start in zip(model.calls, starts, strict=True):
        assert call['x_num'].data_ptr() == dataset.x_num['train'][start].data_ptr()


def test_predict_pack_oom_down_to_batch_size_one(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    reference = predict_pack(model, dataset, 'test', batch_size=64)
    model.calls.clear()
    model.oom_above = 1
    predictions = predict_pack(model, dataset, 'test', batch_size=8)
    torch.testing.assert_close(predictions, reference, rtol=1e-6, atol=1e-6)
    sizes = [call['batch_size'] for call in model.calls]
    assert sizes == [8, 4, 2] + [1] * dataset.size('test')


def test_predict_pack_oom_at_batch_size_one_raises(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    model.train()
    model.oom_above = 0
    with pytest.raises(torch.OutOfMemoryError):
        predict_pack(model, dataset, 'val', batch_size=4)
    assert [call['batch_size'] for call in model.calls] == [4, 2, 1]
    assert model.training


def test_predict_pack_does_not_retry_other_errors(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    model.oom_above = 0
    model.oom_error = lambda _: RuntimeError('shape mismatch')
    with pytest.raises(RuntimeError, match='shape mismatch'):
        predict_pack(model, dataset, 'val', batch_size=16)
    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# evaluate_pack
# ---------------------------------------------------------------------------


def _score_fns(dataset: PreparedDataset) -> dict:
    return {part: make_score_fn(dataset.y[part], dataset.task) for part in dataset.y}


@pytest.mark.parametrize('task_type', list(TaskType))
def test_evaluate_pack_scores_equal_score_fn_of_predictions(
    dataset: PreparedDataset, task_type: TaskType
) -> None:
    dataset = _with_task(dataset, task_type)
    model = _fake_for(dataset)
    score_fns = _score_fns(dataset)
    scores, predictions = evaluate_pack(
        model, dataset, ['val', 'test'], score_fns, batch_size=16
    )
    assert list(scores) == ['val', 'test'] == list(predictions)
    for part in ('val', 'test'):
        assert predictions[part].shape == _expected_shape(dataset, dataset.size(part))
        assert predictions[part].dtype == torch.float32
        torch.testing.assert_close(
            predictions[part],
            predict_pack(model, dataset, part, batch_size=16),
            rtol=0,
            atol=0,
        )
        assert scores[part].shape == (K,)
        torch.testing.assert_close(
            scores[part], score_fns[part](predictions[part]), rtol=0, atol=0
        )


def test_evaluate_pack_restores_mode_and_calls_score_fn_once_per_part(
    dataset: PreparedDataset,
) -> None:
    model = _fake_for(dataset)
    model.train()
    seen = []

    def score_fn(y_pred: Tensor) -> Tensor:
        seen.append(y_pred.shape)
        return y_pred.mean(dim=1)

    scores, _ = evaluate_pack(
        model, dataset, ['train', 'val'], {'train': score_fn, 'val': score_fn}
    )
    assert seen == [(K, 97), (K, 41)]
    assert model.training and model.dropout.training
    assert all(not call['training'] for call in model.calls)
    # The scores are ordinary tensors (usable in autograd-enabled code).
    assert not scores['val'].is_inference()


def test_evaluate_pack_missing_score_fn(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    with pytest.raises(KeyError, match='test'):
        evaluate_pack(model, dataset, ['val', 'test'], {'val': lambda p: p.mean(1)})
    assert not model.calls


def test_evaluate_pack_keeps_the_reduced_batch_size(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    model.oom_above = 10
    evaluate_pack(model, dataset, ['val', 'test'], _score_fns(dataset), batch_size=32)
    sizes = [call['batch_size'] for call in model.calls]
    # val (41 rows): 32 and 16 fail, then 8-row batches; test (23 rows) starts at 8.
    val_sizes, test_sizes = sizes[:8], sizes[8:]
    assert val_sizes == [32, 16, 8, 8, 8, 8, 8, 1]
    assert test_sizes == [8, 8, 7]


def test_evaluate_pack_empty_parts(dataset: PreparedDataset) -> None:
    model = _fake_for(dataset)
    scores, predictions = evaluate_pack(model, dataset, [], {})
    assert scores == {} and predictions == {}


# ---------------------------------------------------------------------------
# GPU (skipped without CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_predict_pack_on_cuda_matches_cpu(
    dataset: PreparedDataset, cuda_device: torch.device
) -> None:
    model = _fake_for(dataset)
    reference = predict_pack(model, dataset, 'train', batch_size=32)
    model.to(cuda_device)
    # The dataset stays on the CPU: batches are moved to the model device.
    predictions = predict_pack(model, dataset, 'train', batch_size=32)
    assert predictions.device.type == 'cuda'
    torch.testing.assert_close(predictions.cpu(), reference, rtol=1e-5, atol=1e-5)
    autocast = contextlib.nullcontext()
    scores, _ = evaluate_pack(
        model,
        dataset,
        ['val'],
        {'val': make_score_fn(dataset.y['val'].to(cuda_device), dataset.task)},
        autocast=autocast,
    )
    assert scores['val'].device.type == 'cuda'
