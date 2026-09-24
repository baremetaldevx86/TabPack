# a21-train-evaluate: agent report

## Summary

Implemented `logits_to_predictions`, `predict_pack` and `evaluate_pack` in
`src/tabpack_repro/training/evaluate.py` following the frozen contract. I compared
the code with the official `_evaluate` + `apply_model_impl` + `adjust_gpu_memory_usage`
(`project/tabpack.py`, `lib/utils.py`). Predictions are float32 `(K, N[, C])` on the
model device and are the same as the official ones (sigmoid/softmax on float logits).
Inference runs in `torch.inference_mode` and eval mode. Each submodule's previous
train/eval mode is restored afterwards. The inputs are shared `(B, f)` views, and a
CUDA OOM halves the batch size and retries, down to a batch size of 1.

## Files

- `src/tabpack_repro/training/evaluate.py`: implementation. The public contract is
  unchanged. Private helpers: `_predict_pack` (returns the batch size that worked
  last), `_eval_mode`, `_is_oom_error`, `_model_device`, `_slice`.
- `tests/training/test_evaluate.py`: 68 tests. 66 run on the CPU and 2 are marked
  `gpu`. They use a fake pack module and a real `ModelPack`.
- `evidence/agents/a21-train-evaluate.md`: this report.

## Design decisions

- **float32 before the activation**: `logits.float()` runs before
  sigmoid/softmax/identity, so bf16/fp16 autocast logits are never activated in low
  precision (the official code does the same with `.float()`). Float64 logits become
  float32 too. `task_type` is parsed with `TaskType(...)`, so an unknown type raises
  `ValueError`.
- **Shared inputs, no gather**: each batch is `x[part][start:end]`, a view with no
  copy. It is moved to the model device with `.to(device)`, which does nothing when
  the tensor is already there. The official code gathers rows with an index tensor.
  Rows are independent in eval mode, so the result is the same. A dataset on the CPU
  with a model on CUDA also works (GPU test).
- **Preallocated output**: the first batch sets the output shape as
  `(K, N, *rest)`, and every batch is written into place. `torch.cat` would briefly
  hold twice the memory. An empty part still runs one empty batch and returns
  `(K, 0[, C])`.
- **Eval mode**: the `training` flag of every submodule is saved and restored in a
  `finally` block, so mixed modes survive and an exception inside the forward also
  restores them. The official code leaves the model in eval mode and relies on the
  loop calling `model.train()`.
- **OOM fallback**: an error counts as OOM if it is a `torch.OutOfMemoryError` (the
  same class as `torch.cuda.OutOfMemoryError`), or a `RuntimeError` whose message
  contains one of the strings the official `is_oom_exception` checks. On OOM the batch
  size is halved and a warning is logged. The retry happens outside the `except`
  block, so the traceback, which holds the failed forward's activations, is freed
  first. Inference then resumes from the failed batch; batches that already finished
  are kept. The official code restarts the whole evaluation. The results are the same
  up to float round-off from batch size. An OOM at `batch_size == 1` re-raises the
  original error. The official code raises
  `RuntimeError('Not enough memory even for batch_size=1')`, which is also a
  RuntimeError. Errors that are not OOM propagate immediately. `batch_size` must be a
  positive `int` (`ValueError`/`TypeError`).
- **evaluate_pack**: it checks that every part has a score function (`KeyError`)
  before running any forward. Predictions for all parts come from one
  inference-mode/eval-mode region. A batch size reduced by OOM is kept for the
  following parts, as with the official decorator, which reduces it for the whole
  call. Scores are `score_fns[part](predictions[part])` and are computed *outside*
  inference mode, so they are ordinary tensors.
- **autocast**: the context manager is entered separately around each forward
  (`with autocast:`). It must therefore be reusable when entered one call after
  another. A `torch.autocast` instance and the a29 `make_autocast` object both are.
- **No finiteness assert**: the official `_evaluate` asserts `np.isfinite(y_pred)`,
  which costs a host sync every evaluation. I left that check to the caller or trainer.

## Tests

```
tools/dev/py -m pytest tests/training/test_evaluate.py -q
66 passed, 1 skipped in 7.09s        (the skip is the CUDA test; CUDA is hidden)
TABPACK_GPU=1 tools/dev/py -m pytest tests/training/test_evaluate.py -q
68 passed in 6.12s
tools/dev/py -m ruff check / ruff format --check (both owned files): clean
```

Coverage:
- `logits_to_predictions`: exact sigmoid/softmax/identity; float32 output from
  bf16/fp16/fp64 inputs, matching activation of the float32 logits; extreme logits;
  unknown task type.
- `predict_pack` (fake pack):
  - shape/dtype/device for all 3 task types;
  - batching invariance for batch sizes 1, 2, 7, 40, 41, 42 and 32768, plus the
    expected batch split;
  - inputs are 2D views whose `data_ptr` points into the dataset tensors;
  - datasets without `x_cat` or without `x_num`, and an empty part;
  - eval mode during the forward and restored afterwards (train or eval, mixed
    submodules, after an exception);
  - inference mode during the call and not leaking out of it;
  - autocast entered once per forward, and CPU bf16 autocast giving float32 output;
  - invalid `batch_size`.
- OOM:
  - halving for `OutOfMemoryError` and for a RuntimeError with an OOM message;
  - resume from the failed batch, checked with `data_ptr`;
  - falling back all the way to batch size 1;
  - re-raise at batch size 1 with the mode restored;
  - errors that are not OOM are not retried.
- `evaluate_pack`:
  - scores equal `score_fn(predictions)` and predictions equal `predict_pack`, for
    all task types;
  - one score call per part, mode restored, scores are not inference tensors;
  - `KeyError` on a missing score function before any forward;
  - the reduced batch size carries over to the next part;
  - empty `parts`.
- Real `ModelPack` (after merging a12), heterogeneous `n_blocks=[1,2,3]` and
  `dropout=[0,.3,.5]`:
  - matches a manual per-member `(K, B, f)` forward for all task types;
  - batching invariance;
  - dropout is off and train mode is restored;
  - the model can still be trained after evaluation;
  - `evaluate_pack` scores match per-member accuracy;
  - a monkeypatched `ModelPack.forward` OOM gives the same predictions;
  - GPU: CUDA + bf16 autocast gives float32 predictions close to fp32.

## Coordination

- Received: integrator #60 (merged `checkpoint/00b-data-fix`), #63 (scheduler note).
  Nothing else was addressed to a21.
- Merged peer branches (after their `done`): `feat/a08-metrics` (#52) and
  `feat/a12-nn-model` (#77, which brings in a09/a10/a11).
- Posted: #70 `status` (started) and #98 `done`, which lists the API details for a23,
  including the inference-tensor note below.

## Open issues

- The predictions returned are **inference tensors**, as in the official code,
  because the contract wraps `predict_pack` in `torch.inference_mode`. Reading them,
  or using them as the source of `index_copy_`/`copy_`, works anywhere. An in-place
  update *of* them outside inference mode raises, so callers must `clone()` first.
  This matters to a23/a20/a26.
- Regression predictions are raw model outputs. The contract says "identity", and
  label de-standardization is out of scope (`build_dataset` raises for regression).
  If regression is ever supported, the trainer must de-standardize them.
- No real-CUDA OOM test: that would mean exhausting the shared 8 GB GPU. The OOM
  path is tested with monkeypatched exceptions of the real exception class.
