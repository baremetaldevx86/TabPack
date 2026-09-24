# a26-ens-online: online greedy ensemble

## Summary

I implemented `OnlineGreedyEnsemble` in `src/tabpack_repro/ensembles/online.py`. It
reproduces the official `project/tabpack.py::OnlineEnsemble` with `type='greedy'`,
`update_type='latest'`, `include_current_ensemble_in_pool=True` and
`update_part='val'`, and it provides the report fields that
`update_online_ensembles` records. I also checked it against the official class in an
ad-hoc script (scratchpad only, not committed, because a41 owns the parity tests). It
ran 60 simulated training runs with 486 `update` calls. The runs covered binclass
accuracy, ROC-AUC and log-loss, multiclass accuracy, quantized probabilities with many
ties, members that finish mid-run, `max_ensemble_size` in {None, 3, 8} and patience
from 0 to 4. The two classes matched exactly at every step on all of these: the
`improved` flag, `ids`, `steps`, the val score, the remaining patience, and the stored
per-part predictions (bit for bit). 176 of the 486 updates had the same id more than
once in the ensemble.

## Files

* `src/tabpack_repro/ensembles/online.py`: the implementation, plus the private
  helpers `_Source`, `_as_int64_1d`, `_to_numpy` and `_prepare_pool`.
* `tests/ensembles/test_online.py`: 29 CPU tests and 1 GPU test.
* `evidence/agents/a26-ens-online.md`: this report.

## Design decisions

* **Pool order.** The pool is built as current ensemble entries, then finished
  members, then running members at their latest step. `greedy_ensemble` breaks ties by
  the first maximum, so this order decides ties. When scores are equal, a snapshot that
  is already in the ensemble beats a new entry, and a finished member beats a running
  one. Sources with zero members are skipped, so `finished_predictions` can be `{}`
  (the default `FinishedPool`), and `running_predictions` can be `{}` or hold 0-row
  tensors.
* **Acceptance.** The candidate is `average_predictions(pool_val[idx])`, a plain
  `mean(0)` that matches the official `compute_ensemble_prediction`. Its score is
  `float(score_fn(candidate[None]).item())`. The candidate is accepted if it is the
  first update or if `score > self.score` (a strict improvement). On acceptance, the
  ensemble stores the selected ids, steps and prediction rows, and patience resets.
  Otherwise the remaining patience goes down by 1. `is_running` means remaining
  patience >= 0. If `update()` is called after the ensemble has stopped, it raises
  `RuntimeError`, which mirrors the official `assert self.is_running`.
* **Snapshots.** The ensemble keeps its own copies of the selected rows: `torch.cat`
  and advanced indexing both copy, and the rows are detached. A caller can therefore
  reuse or modify its prediction buffers in place. A running member's old snapshot can
  stay in the ensemble, possibly next to a newer snapshot of the same id.
* **Parts.** The pool uses the parts shared by every source that has at least one
  member, in the order of the newest source. `'val'` is required. The official code
  uses the running members' keys and drops extra keys such as the finished members'
  `'train'`. Taking the intersection behaves the same in that case. It also works when
  no member is running any more, where the official code would hit a `KeyError`.
  Every part in the pool is stored. `predictions()` returns the cached per-part
  torch means, and the val mean is the exact tensor that was scored.
* **Devices.** The pool lives on the device of the newest source (running, else
  finished, else current), and other sources are moved there with `.to()`. This means
  a CPU `FinishedPool` can be mixed with CUDA running predictions.
* **report(y_true).** It returns `{'ids': list, 'steps': list, 'size', 'n_unique',
  'score_val', 'metrics': {part: compute_metrics(...)}}`. The metrics come from the
  numpy average of the stored snapshots, as in the official report. `y_true` values
  may be numpy arrays or tensors. A stored part that has no labels raises
  `KeyError`. Before the first update, `report()` returns empty lists, size 0,
  `score_val` None and `metrics` `{}`.
* **Validation.** `ValueError` is raised for: patience < 0, `max_ensemble_size` < 1,
  an empty pool, id/step count mismatches, prediction row mismatches, and a missing
  `'val'` part. `ids`, `steps` and scalars may be numpy arrays, lists or tensors;
  they are converted to int64. The `ids` and `steps` properties return copies.

## Tests

`tools/dev/py -m pytest tests/ensembles/test_online.py -q`: 29 passed, 1 skipped
(CUDA). The GPU test passes with `TABPACK_GPU=1`. After merging a25 and a27, the
whole `tests/ensembles` directory passes: 133 passed, 2 skipped. `ruff check` and
`ruff format --check` pass on the owned files.

What the tests cover:

* the first update is accepted, even when it is bad;
* greedy selection and averaging for both parts;
* equal or worse updates keep the ensemble and use up patience;
* `is_running` flips after patience+1 misses, for patience in {0, 1, 3}, and a
  later `update()` raises `RuntimeError`;
* an accepted update resets patience;
* a running member's snapshot survives in-place buffer changes and is later
  combined with another member's newer entry;
* the same id can appear twice with different steps (`n_unique` 1, size 2);
* finished members join the pool, including a pool with only finished members
  (`{}` or 0-row running predictions);
* ties go to the current ensemble first, then finished, then running;
* `max_ensemble_size` is respected;
* the parts intersection;
* returned arrays are copies, and tensor or list inputs are accepted;
* validation errors;
* `report()` before the first update, and `report()` against `compute_metrics` with
  the real binclass and multiclass `make_score_fn`;
* a 40-step simulated run that checks the score only increases, patience matches its
  expected countdown, and the stored snapshots reproduce the stored score;
* stored predictions are detached;
* a mixed CPU/CUDA pool (GPU test).

## Coordination

* Merged `checkpoint/00b-data-fix`, `feat/a08-metrics`, `feat/a27-ens-aggregate`
  (after its done post, #95) and `feat/a25-ens-greedy` (after its done post, #108).
  Until a25 was done, I ran the tests with a scratchpad pytest plugin that stood in for
  greedy; the committed tests use the real functions.
* Posted status #81 and the final `done`. Sent a41 a note on how to import the
  official `project.tabpack` for the parity test (see Open issues).

## Open issues

* Importing the official `project.tabpack` in this venv fails on the missing
  `rtdl_num_embeddings` and `rtdl_revisiting_models` modules. My ad-hoc parity script
  replaced them with `MagicMock` modules through a small `sys.meta_path` finder; the
  online-ensemble code does not use them. a41 will need the same workaround, or the
  integrator could add these modules to the parity dependency group.
* a25's `greedy_ensemble` raises `ValueError` on NaN scores. ROC-AUC on a val part
  with only one class gives NaN, and the official code would carry on instead. This
  cannot happen on Churn.
* The contract has no ensemble `weights`, because greedy without replacement always
  gives uniform weights.
