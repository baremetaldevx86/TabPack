"""Tests for tabpack_repro.ensembles.aggregate.average_predictions (a27)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpack_repro.ensembles import average_predictions


def _probs(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


# ---------------------------------------------------------------- uniform mean


@pytest.mark.parametrize('shape', [(1, 7), (5, 7), (4, 6, 3)])
def test_uniform_mean_numpy(shape):
    x = _probs(shape)
    out = average_predictions(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == shape[1:]
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, x.mean(0), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('shape', [(1, 7), (5, 7), (4, 6, 3)])
def test_uniform_mean_torch(shape):
    x = torch.from_numpy(_probs(shape))
    out = average_predictions(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == shape[1:]
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.mean(0))


def test_single_member_is_identity():
    x = _probs((1, 9))
    np.testing.assert_array_equal(average_predictions(x), x[0])
    np.testing.assert_array_equal(average_predictions(x, np.array([3.0])), x[0])
    xt = torch.from_numpy(x)
    torch.testing.assert_close(average_predictions(xt), xt[0], rtol=0, atol=0)


def test_constant_predictions_stay_constant():
    x = np.full((6, 5, 2), 0.25, dtype=np.float32)
    w = np.array([1.0, 2.0, 0.0, 3.0, 0.5, 7.0])
    np.testing.assert_allclose(average_predictions(x, w), 0.25, rtol=1e-6)


def test_uniform_weights_equal_no_weights():
    x = _probs((5, 8))
    for w in (np.ones(5), np.full(5, 0.2), np.full(5, 13.0)):
        np.testing.assert_allclose(
            average_predictions(x, w), average_predictions(x), rtol=1e-6, atol=1e-7
        )


# --------------------------------------------------------------- weighted mean


@pytest.mark.parametrize('shape', [(4, 10), (4, 10, 3)])
def test_weighted_mean_matches_manual_numpy(shape):
    x = _probs(shape, seed=1)
    w = np.array([1.0, 3.0, 0.0, 2.0])
    out = average_predictions(x, w)
    x64 = x.astype(np.float64)
    expected = sum(w[i] * x64[i] for i in range(4)) / w.sum()
    assert out.dtype == np.float32
    assert out.shape == shape[1:]
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('shape', [(4, 10), (4, 10, 3)])
def test_weighted_mean_matches_manual_torch(shape):
    x = torch.from_numpy(_probs(shape, seed=2))
    w = torch.tensor([0.5, 0.0, 1.5, 2.0])
    out = average_predictions(x, w)
    x64 = x.double()
    expected = (w.double()[:, None, None] * x64.reshape(4, 10, -1)).sum(0) / w.sum()
    assert out.dtype == torch.float32
    assert out.shape == shape[1:]
    torch.testing.assert_close(out, expected.reshape(shape[1:]).float())


def test_weights_are_scale_invariant():
    x = _probs((3, 12))
    w = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        average_predictions(x, w), average_predictions(x, 10.0 * w), rtol=1e-6
    )


def test_integer_weights_act_as_counts():
    # Greedy selection with replacement yields counts: [2, 1] == [a, a, b].
    x = _probs((2, 6))
    out = average_predictions(x, np.array([2, 1]))
    np.testing.assert_allclose(
        out, average_predictions(x[[0, 0, 1]]), rtol=1e-6, atol=1e-7
    )


def test_zero_weight_member_is_ignored():
    x = _probs((3, 6))
    x_bad = x.copy()
    x_bad[1] = 1e6
    w = np.array([1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        average_predictions(x_bad, w), average_predictions(x[[0, 2]]), rtol=1e-6
    )


def test_multiclass_rows_still_sum_to_one():
    logits = np.random.default_rng(3).normal(size=(5, 20, 4))
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    out = average_predictions(probs.astype(np.float32), np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(out.sum(-1), 1.0, rtol=1e-6)


# ------------------------------------------------------------ numpy/torch parity


@pytest.mark.parametrize('shape', [(6, 11), (6, 11, 3)])
@pytest.mark.parametrize('use_weights', [False, True])
def test_numpy_torch_parity(shape, use_weights):
    x = _probs(shape, seed=4)
    w = (
        np.array([0.1, 0.0, 2.0, 1.0, 1.0, 0.7], dtype=np.float32)
        if use_weights
        else None
    )
    out_np = average_predictions(x, w)
    out_t = average_predictions(
        torch.from_numpy(x), None if w is None else torch.from_numpy(w)
    )
    np.testing.assert_allclose(out_t.numpy(), out_np, rtol=1e-6, atol=1e-7)


def test_mixed_weight_types_are_accepted():
    x = _probs((3, 5))
    w = np.array([1.0, 2.0, 3.0])
    ref = average_predictions(x, w)
    out_np_tw = average_predictions(x, torch.from_numpy(w))
    out_t_nw = average_predictions(torch.from_numpy(x), w)
    assert isinstance(out_np_tw, np.ndarray)
    assert isinstance(out_t_nw, torch.Tensor)
    np.testing.assert_allclose(out_np_tw, ref, rtol=1e-6)
    np.testing.assert_allclose(out_t_nw.numpy(), ref, rtol=1e-6)


# ----------------------------------------------------------------------- dtypes


@pytest.mark.parametrize(
    'dtype', [np.float16, np.float32, np.float64, np.int64, np.bool_]
)
def test_numpy_output_is_float32(dtype):
    x = (_probs((4, 5)) > 0.5).astype(dtype)
    out = average_predictions(x, np.array([1.0, 1.0, 2.0, 0.0]))
    assert out.dtype == np.float32
    assert average_predictions(x).dtype == np.float32


@pytest.mark.parametrize(
    'dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64]
)
def test_torch_output_is_float32(dtype):
    x = torch.from_numpy(_probs((4, 5)) > 0.5).to(dtype)
    out = average_predictions(x, torch.tensor([1.0, 1.0, 2.0, 0.0]))
    assert out.dtype == torch.float32
    assert average_predictions(x).dtype == torch.float32


def test_float64_is_reduced_in_float64():
    # float64 inputs are reduced in float64 and cast once at the end.
    x = np.random.default_rng(5).random((7, 1000))
    w = np.array([1.0, 2.0, 0.5, 0.0, 3.0, 1.0, 0.25])
    expected_uniform = x.mean(0).astype(np.float32)
    expected_weighted = ((w[:, None] * x).sum(0) / w.sum()).astype(np.float32)
    np.testing.assert_array_equal(average_predictions(x), expected_uniform)
    np.testing.assert_array_equal(average_predictions(x, w), expected_weighted)
    xt = torch.from_numpy(x)
    np.testing.assert_array_equal(average_predictions(xt).numpy(), expected_uniform)
    np.testing.assert_array_equal(average_predictions(xt, w).numpy(), expected_weighted)
    # Casting before reducing would differ in some entries.
    assert (average_predictions(x.astype(np.float32)) != expected_uniform).any()


def test_inputs_are_not_modified():
    x = _probs((3, 4))
    w = np.array([1.0, 2.0, 3.0])
    x_copy, w_copy = x.copy(), w.copy()
    average_predictions(x, w)
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(w, w_copy)
    xt, wt = torch.from_numpy(x_copy.copy()), torch.from_numpy(w_copy.copy())
    average_predictions(xt, wt)
    np.testing.assert_array_equal(xt.numpy(), x_copy)
    np.testing.assert_array_equal(wt.numpy(), w_copy)


def test_torch_gradients_flow():
    x = torch.from_numpy(_probs((3, 4))).requires_grad_()
    average_predictions(x, torch.tensor([1.0, 0.0, 3.0])).sum().backward()
    expected = torch.tensor([0.25, 0.0, 0.75])[:, None].expand(3, 4)
    torch.testing.assert_close(x.grad, expected)


# ----------------------------------------------------------------------- device


@pytest.mark.gpu
def test_cuda_device_is_preserved(cuda_device):
    x = torch.from_numpy(_probs((5, 8, 2))).to(cuda_device)
    w_cpu = torch.tensor([1.0, 2.0, 0.0, 1.0, 4.0])
    for w in (None, w_cpu, w_cpu.to(cuda_device), w_cpu.numpy()):
        out = average_predictions(x, w)
        assert out.device == x.device
        assert out.dtype == torch.float32
        ref = average_predictions(
            x.cpu(), w if not isinstance(w, torch.Tensor) else w.cpu()
        )
        torch.testing.assert_close(out.cpu(), ref)
    with pytest.raises(ValueError, match='non-negative'):
        average_predictions(x, -w_cpu.to(cuda_device))


def test_cpu_device_is_preserved():
    x = torch.from_numpy(_probs((3, 4)))
    assert average_predictions(x).device == x.device
    assert average_predictions(x, np.ones(3)).device == x.device


# ----------------------------------------------------------------------- errors


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('shape', [(5,), (2, 3, 4, 5), ()])
def test_bad_prediction_ndim_raises(backend, shape):
    x = np.zeros(shape, dtype=np.float32)
    x = torch.from_numpy(x) if backend == 'torch' else x
    with pytest.raises(ValueError, match=r'\(M, N\) or \(M, N, C\)'):
        average_predictions(x)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_empty_members_raises(backend):
    x = np.zeros((0, 4), dtype=np.float32)
    x = torch.from_numpy(x) if backend == 'torch' else x
    with pytest.raises(ValueError, match='at least one member'):
        average_predictions(x)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize(
    ('weights', 'match'),
    [
        (np.ones(3), r'shape \(M,\) = \(4,\)'),
        (np.ones(5), r'shape \(M,\) = \(4,\)'),
        (np.ones((4, 1)), r'shape \(M,\) = \(4,\)'),
        (np.array(1.0), r'shape \(M,\) = \(4,\)'),
        (np.array([1.0, -0.5, 1.0, 1.0]), 'non-negative'),
        (np.zeros(4), 'positive sum'),
        (np.array([1.0, np.nan, 1.0, 1.0]), 'finite'),
        (np.array([1.0, np.inf, 1.0, 1.0]), 'finite'),
    ],
)
def test_bad_weights_raise(backend, weights, match):
    x = _probs((4, 3))
    if backend == 'torch':
        x, weights = torch.from_numpy(x), torch.from_numpy(weights)
    with pytest.raises(ValueError, match=match):
        average_predictions(x, weights)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_complex_inputs_raise(backend):
    x = np.ones((2, 3), dtype=np.complex64)
    w = np.ones(2, dtype=np.complex64)
    xr = _probs((2, 3))
    if backend == 'torch':
        x, w, xr = torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(xr)
    with pytest.raises(TypeError, match='real dtype'):
        average_predictions(x)
    with pytest.raises(TypeError, match='real dtype'):
        average_predictions(xr, w)


def test_unsupported_type_raises():
    with pytest.raises(TypeError, match=r'torch\.Tensor or np\.ndarray'):
        average_predictions([[0.1, 0.2], [0.3, 0.4]])  # type: ignore[arg-type]
