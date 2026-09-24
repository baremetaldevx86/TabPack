# a27-ens-aggregate: agent report

## Summary

Implemented `average_predictions(predictions, weights=None)`: the (weighted) mean
over the member axis of `(M, N)` or `(M, N, C)` predictions. It works for numpy
arrays and torch tensors, keeps the array type (and the torch device), and always
returns float32. Weights are validated and normalized. For float32 inputs the result
is bit-identical to the official `compute_ensemble_prediction` in both
`project/ensemble_utils.py` and `project/ensemble_utils_torch.py`.

## Files

- `src/tabpack_repro/ensembles/aggregate.py`: implementation plus private helpers.
- `tests/ensembles/test_aggregate.py`: 64 tests (1 gpu-marked).
- `evidence/agents/a27-ens-aggregate.md`: this report.

## Design decisions

- **What the official code averages.** Official `compute_ensemble_prediction` averages
  the stored predictions directly. Stored predictions are `PROBS` for classification
  (sigmoid or softmax outputs) and `LABELS` for regression (raw outputs). Logits are
  never stored, see the NOTE at `tabpack.py:1375`. The code does no log-space or
  geometric averaging. With `weights=None` it computes `predictions.mean(0)`.
  Otherwise it computes `(predictions * (w / w.sum())[:, None...]).sum(0)` and, only
  for `PROBS`, clamps the result to at most 1.0. It only asserts `weights.ndim == 1`
  and does no other validation. My implementation performs the same operations in the
  same order, so float32 results match bit for bit.
- **No clamp.** The contract has no `prediction_type`, so the function does not clamp.
  The uniform path cannot exceed 1: a float32 mean of values that are all at most 1 is
  at most 1. The weighted path can go 1 ulp above 1.0. With all-ones inputs and random
  weights, this happened in 411 of 2000 trials. See Open issues.
- **Dtypes.** The output is always float32. float64 inputs are reduced in float64 and
  cast once at the end. float16, bfloat16, integer and bool inputs are cast to float32
  before the reduction. Complex inputs raise `TypeError`.
- **Weights.** Weights may be numpy or torch, independent of the predictions type.
  Torch weights on another device are moved to the predictions' device. Weights are
  cast to the compute dtype, so integer counts, such as those from greedy selection
  with replacement, work. Validation raises `ValueError` in these cases:
  - the shape is not `(M,)`
  - a weight is NaN or inf
  - a weight is negative
  - the weights sum to 0

  On torch, all value checks share one host sync. The per-check error details are
  computed only on the failure path.
- **Predictions validation.** `ndim` must be 2 or 3 and `M` must be at least 1.
  Otherwise the function raises `ValueError`. A type other than ndarray or Tensor
  raises `TypeError`.
- **Autograd and inputs.** Autograd is not detached, so gradients flow to the
  predictions. The function never modifies its inputs.

## Tests

`tools/dev/py -m pytest tests/ensembles/test_aggregate.py -q` gives **63 passed,
1 skipped** (the CUDA test). `TABPACK_GPU=1 tools/dev/py -m pytest
tests/ensembles/test_aggregate.py -q -m gpu` gives **1 passed**.

The tests cover:
- the uniform mean
- the weighted mean against a manual float64 computation (numpy and torch, 2D and 3D)
- scale invariance
- integer counts behaving like repeated members
- zero-weight members being ignored
- multiclass rows still summing to 1
- numpy/torch parity
- mixed weight and prediction types
- float32 output for 5 numpy and 5 torch dtypes
- float64 inputs reduced in float64 before the cast (exact)
- gradients
- CPU and CUDA device preservation, including CPU, CUDA and numpy weights with CUDA
  predictions
- all error cases

`ruff check` and `ruff format --check` pass on the owned files.

I also ran an ad-hoc comparison with the official code outside the repo. It was not
committed because official imports belong in `tests/parity/**`, which I do not own.
The comparison covered 300 random shapes (M 1-39, N 1-299, 2D/3D) x {None, random
float weights, integer counts}. The result was **900/900 bit-exact** for both the
numpy and the torch official functions.

## Coordination

- Merged `checkpoint/00b-data-fix` (fast-forward), as the integrator asked in #60.
- Posted `status` #72 (started) and `done` at the end.
- No messages were addressed to me. I merged no peer branches because I have no
  dependencies.

## Open issues

- **Clamp difference for PROBS.** The official code clamps to at most 1.0 for `PROBS`
  in the weighted path, and this function does not. The difference is at most 1 ulp
  above 1.0, and only when weights are passed. `compute_metrics` clips for log-loss,
  and accuracy and AUC are unaffected, so metrics should not change. If a caller
  (a26 or a31) needs exact official parity for weighted probabilities, it should apply
  `clamp_max(1.0)` or `np.minimum(., 1.0)` itself.
