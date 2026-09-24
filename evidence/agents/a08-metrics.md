# a08-metrics: agent report

## Summary

Implemented `compute_metrics`, `score_pack` and `make_score_fn` in
`src/tabpack_repro/metrics.py` following the frozen contract and the official semantics
(`project/metrics_torch.py::calculate_metrics_pack`, `lib/metrics.py::calculate_metrics`).
Binclass accuracy rounds half to even (p=0.5 -> class 0) and is bit-identical to the
official `calculate_metrics_pack` (parity test). `score_pack` is vectorized over M
with no Python loops and no host syncs, and returns `(M,)` float32 on the input device
(checked on CPU and CUDA).

## Files

- `src/tabpack_repro/metrics.py`: implementation (public contract unchanged; private
  helpers `_check_score`, `_signed`, `_roc_auc_numpy`, `_log_loss_numpy`,
  `_roc_auc_pack`, constant `_EPS`).
- `tests/test_metrics.py`: 70 tests (64 on CPU, 6 `gpu`-marked).

## Design decisions

- **Supported main scores**: binclass `accuracy | roc-auc | log-loss`, multiclass
  `accuracy | log-loss`, regression `rmse`, which are the keys of `HIGHER_IS_BETTER`.
  Any other combination raises `ValueError` in all three functions. The official name
  `cross-entropy` is **not** accepted as an alias. Config/data use `log-loss`.
- **`score` sign**: `score = metric` if `HIGHER_IS_BETTER[metric]`, otherwise
  `-metric` (so `-log-loss` and `-rmse`).
- **Accuracy in torch**: the correct-prediction count is summed as int64, cast to float32
  and divided by N. This matches the official `.float().mean(1)` bit for bit (count is
  exact below 2**24). Multiclass uses `argmax(-1)`, so ties go to the first class, as in
  numpy.
- **ROC-AUC in torch**: the Mann-Whitney form with average ranks for ties (two
  `searchsorted` calls on the row-sorted predictions, rank sums in exact int64). It equals
  `sklearn.metrics.roc_auc_score`, ties included. The official torch ROC-AUC does not
  handle ties (argsort order), so for tied predictions ours follows sklearn rather than
  the official torch code. This does not affect Churn, whose score is accuracy.
  Single-class labels give NaN (no exception, no sync) in both numpy and torch.
- **Log-loss**: probabilities are clipped to `[eps, 1-eps]` with
  `eps = float32 machine epsilon` (1.19e-7), in both numpy and torch. In float32,
  `1 - eps < 1` still holds, so the numpy and torch values agree. The official code
  instead uses `log(p + 1e-8)` (multiclass) and torch BCE, which clamps the log at -100
  (binclass). The values match away from p in {0, 1}, which the parity test checks
  at rtol 1e-4.
  numpy uses `sklearn.metrics.log_loss(..., labels=arange(C))` and suppresses only the
  "do not sum to one" warning, because float32 softmax rows sum to 1 only to float32
  precision.
- **dtypes**: float16/bfloat16 inputs (autocast) are upcast to float32. float64 inputs
  are computed in float64. The output is always float32. `y_true` may be int64 or float
  and is moved to `y_pred.device` (a no-op when already there).
- **`make_score_fn`** validates the task and label shape once, casts classification
  labels to int64 once, and returns a closure over `score_pack`.
- **Shape checks** only read metadata (no host sync). `y_pred` must be `(M, N)` (binclass
  and regression) or `(M, N, C)` (multiclass). `M = 0` returns an empty float32 tensor.

## Tests

`tools/dev/py -m pytest tests/test_metrics.py -q` gives **64 passed, 6 skipped** (the skips
are the CUDA tests on the CPU-only runner).
`TABPACK_GPU=1 tools/dev/py -m pytest tests/test_metrics.py -q -m gpu` gives **6 passed**.
`ruff check` and `ruff format --check` on both owned files pass.

Coverage:
- hand-computed binclass, multiclass and regression cases;
- the p=0.5 rule;
- multiclass argmax ties;
- log-loss clipping at p in {0, 1};
- single-class ROC-AUC NaN;
- agreement of `score_pack` with `compute_metrics['score']` for random predictions,
  over all 6 task/score pairs and 2 seeds (accuracy exactly equal);
- ROC-AUC with heavy ties against sklearn;
- row independence;
- sign convention (better predictions score higher, and `score` equals the metric with
  the correct sign);
- output float32 for float16, bfloat16 and float64 inputs;
- int or float labels;
- an empty pack;
- shape and unsupported-score errors;
- `make_score_fn` equivalence;
- CUDA vs CPU (`gpu` marker);
- parity with the official `calculate_metrics_pack` (`parity` marker; accuracy
  bit-exact, log-loss/rmse close).

## Coordination

- Posted `status` (#8) at start and `done` at the end.
- No messages addressed to a08 were received. I merged no peer branches (no deps).
- Blocker already reported by a03/a04/a07 (#6, #7, #12): `.gitignore` `data/` also ignores
  `src/tabpack_repro/data/`, so the worktree had no `TaskInfo`. Workaround: I copied the
  skeleton `src/tabpack_repro/data/` from the main checkout into the worktree
  **untracked** (it is ignored and never committed) so that imports and `tests/conftest.py`
  work.

## Open issues

- The integrator must fix the `data/` ignore rule (anchor it to `/data/`) and commit the
  data package. Until then, `metrics.py` cannot be imported from a clean checkout
  because it imports `tabpack_repro.data.dataset.TaskInfo`.
- `compute_metrics` can return NaN for `roc-auc` (single-class part). JSON writers
  (a29 `dump_json`) must accept NaN or map it to null.
- The parity test is in `tests/test_metrics.py` (marked `parity`) and imports the
  official clone itself, because the `official` fixture lives in
  `tests/parity/conftest.py`.
