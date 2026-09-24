# a25-ens-greedy: greedy ensemble selection

## Summary

Implemented `greedy_ensemble(predictions, *, score_fn, max_ensemble_size=None)` in
`src/tabpack_repro/ensembles/greedy.py`. It reproduces the official
`project/ensemble_utils_torch.py::greedy_ensemble` with default options
(`init_top_k=1`, without replacement, no `selection_score_fn`). An ad-hoc check against
the official function (scratchpad only, not committed; a41 owns parity tests) gave
identical selections on 1350/1350 random cases. The cases covered binclass accuracy,
ROC-AUC and log-loss, rounded probabilities with many ties, duplicated rows, and
`max_ensemble_size` in {None, 1, 3}.

## Files

* `src/tabpack_repro/ensembles/greedy.py`: implementation plus private helpers
  `_validate_predictions`, `_score` and `_first_argmax`.
* `tests/ensembles/test_greedy.py`: 42 CPU tests and 1 GPU test.
* `evidence/agents/a25-ens-greedy.md`: this report.

## Design decisions

* **Official arithmetic, host-side decisions.** The initial ensemble is the first
  argmax of the individual scores, and its score is recomputed with
  `score_fn(pred[None])` as in the official code. Candidate ensembles are
  `ens[None] * (s/(s+1)) + predictions[cand] * (1/(s+1))`, with candidates in
  ascending index order, so all candidates are scored in one vectorized call per
  step. The candidate predictions and scores are therefore bit-identical to the
  official ones. The small score vector is moved to the host with one `.tolist()`
  per step. The stopping rule (`best <= current`: stop), the tie detection
  (`== best`) and the tie-break (best individual score, first index) then run on
  Python floats. These comparisons are exact for float32 and float64 scores. The
  cost is one device sync per step, as in the official code, which also syncs on
  the `<=` check.
* **Size limit.** `max_ensemble_size` is capped at M. Values that are not positive
  ints (including `0`, negatives, floats and `bool`) raise `ValueError`. With `0`,
  the official code would silently return one member.
* **Validation.** `predictions` must be a non-empty floating point tensor with
  `ndim >= 1` (otherwise `TypeError`/`ValueError`). It must also be finite
  (`ValueError`), which matches the official `_validate_predictions` asserts.
  `score_fn` must return shape `(M,)`. NaN scores raise `ValueError`: the official
  code would pick a NaN member as the start, or fail an assertion on NaN
  candidates.
* **Return value.** Sorted `int64` indices on the device of `predictions`.
  Duplicate rows (the same pool id appearing twice in the online pool) are distinct
  candidates. Each index is selected at most once.

## Tests

`tools/dev/py -m pytest tests/ensembles/test_greedy.py -q`: **42 passed, 1 skipped**
(the CUDA test; it passes with `TABPACK_GPU=1`). `ruff check` and
`ruff format --check` are clean on both owned files.

Coverage:

* Hand-built known paths: negative MSE over 3 steps, with every size limit; binclass
  accuracy via `metrics.make_score_fn`; multiclass accuracy.
* Tie-break by best individual score beats index order. Equal individual scores
  fall back to the first index. The starting member is the first best individual.
* The strict-improvement stop: an equal-score candidate is not added, and greedy
  stops even when the full average would be better.
* All-equal predictions stop at 1 member. A single prediction returns `[0]`.
* Agreement with a naive from-scratch reference on 8 random problems (paths of 3-9
  members) and on float32 accuracy.
* `max_ensemble_size=k` gives the sorted k-prefix of the full path.
* Indices are unique, sorted and nested across k, and the ensemble score is
  strictly increasing along the path (with duplicated rows in the pool).
* `score_fn` call shapes: (M,N), (1,N), then (M-s,N) per step.
* Validation errors. A CUDA device check.

## Coordination

* Merged `checkpoint/00b-data-fix` and `feat/a08-metrics` (done #52), as
  instructed.
* Board: posted status #78 and `done`. No messages were addressed to me.
  Dependents: a26 (online ensemble) and a41 (parity).

## Open issues

None. Note for a26: the function returns only the indices. The caller recomputes
the ensemble prediction (uniform mean of the selected rows) and its score.
