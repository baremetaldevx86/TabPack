# a23-train-trainer: train_pack

## Summary

Implemented `train_pack` and filled `PackTrainResult`
(`src/tabpack_repro/training/trainer.py`). This is the pack training loop that the
homogeneous ensemble (a31) and TabPack (a32) share. It follows the semantics of the
official loop (`.reference/tabpack/src/project/tabpack.py::main`, lines ~1560-1790)
and uses our modules for every step. I wrote the code myself and copied nothing. The
frozen contract is unchanged. I only added private helpers and argument checks.

Each epoch runs these steps in order:

1. Build the member batches with `generate_member_batches`. The batch generator is a
   `torch.Generator(device).manual_seed(seed)`.
2. For each batch, run the forward pass on `x[batch_idx]` with shape `(K, b, f)`
   (inside the autocast, if one is given), compute
   `loss = make_pack_loss(task)(logits.float(), y[batch_idx]).sum()`, then call
   `zero_grad`, `backward`, `step` and `PackState.step()`.
3. Evaluate val and test with `evaluate_pack`, then call `PackState.update`.
4. Call `compute_stop_idx`. For the stopped members:
   * load their best state with `pack_load_members_`;
   * predict train, val and test with those members only;
   * compute per-member metrics with `compute_metrics`;
   * call `FinishedPool.extend`.
5. Call `pack_select_` on the model, `optimizer_select_` on the optimizer and
   `select_` on the state.
6. Update the online ensemble with the latest predictions of the kept running
   members plus the finished pool.
7. Append a record to the history.

The loop ends when no member is running, or when the online ensemble stops. Members
that have not finished at that point are dropped, as in the official code.

## Files

* `src/tabpack_repro/training/trainer.py`: `train_pack`, `PackTrainResult`, and the
  private helpers `_model_state`, `_members_selected`, `_member_reports`,
  `_ensemble_scores` and `_concat_predictions`.
* `tests/training/test_trainer.py`: 19 tests (18 CPU, 1 GPU).
* `evidence/agents/a23-train-trainer.md`: this report.

## Design decisions

* **Stopped members are predicted in a pack of their own.** The official code does
  this with `module_pack_select` (a temporary selection), and so do we.
  `_members_selected` keeps references to every parameter's `.data` and every
  buffer, calls `pack_select_(model, stop_idx)`, and afterwards puts the old tensors
  back. This undoes the selection exactly, because `pack_select_` rebinds tensors
  rather than changing them in place.

  The simpler option was to predict the full pack and slice out the stopped members.
  That would cost one inference of all K members on train, val and test in every
  epoch where someone stops, which is roughly half an epoch. With TabPack, members
  stop at many different epochs, so that cost would add up.
* **No autograd graph is alive during selection.** The last batch's `logits` and
  `loss` are deleted before evaluation. Evaluation runs in inference mode. This
  follows the caveat from a11 and a13 (#41, #54).
* **`PackState.update` gets detached references, not `pack_state_dict` clones.**
  `_model_state` returns the same keys as `pack_state_dict`, but as detached
  references. PackState clones them on its first update and afterwards only copies
  the slices of improved members. Cloning would copy the whole model every epoch:
  about 150 MB with the official Churn config (64 x d_block 384 x 4 blocks), for no
  benefit.
* **Host-device syncs are kept low.**
  * No `.item()` is called per batch. The loss is summed into a device scalar and
    read once per epoch, after `PackState.update` has already synced.
  * Stop, keep and select indices stay on the CPU for `pack_ops` and `PackState`.
    `optimizer_select_` gets the device copy.
  * The ensemble's test score is computed only when the ensemble improves. The val
    score comes from `online_ensemble.score`.
* **What `FinishedPool` holds.** The pool keeps only `val` and `test` predictions on
  the device, since those are the parts the online ensemble uses.
  `PackTrainResult.predictions` covers train, val and test as numpy arrays, built
  from the per-stop numpy chunks that are also used for the member metrics. When no
  member finished, each part is an empty `(0, N[, C])` float32 array.
* **History.**
  * `train_loss` is the epoch's row-weighted mean of the member-averaged loss.
  * `n_running` is counted after this epoch's removals.
  * `ensemble_val` and `ensemble_test` are `None` when there is no ensemble.
* **Argument checks, all raising ValueError.**
  * `batch_size < 1`.
  * An empty train part.
  * `max_epochs == 0`.
  * `patience < 0` and `max_epochs < 0` with no ensemble: the official code relies
    on a timeout in that case, and without one the run would never end.
  * A dataset on a different device from the model.

  A `RuntimeError` is raised if the model and the state ever disagree on the pack
  size.
* **The model is never called with pack size 0.** The loop requires
  `state.pack_size > 0`, and a stopped subset has at least one member. When every
  member stops, the model, optimizer and state are all selected down to 0 members.
  On return, they hold exactly the members that were still running.
* **What train_pack does not do.**
  * It does not seed the global RNG (dropout, model init). The caller seeds it, as
    `delu.random.seed` does in the official code.
  * It has no timeout, `track_*` options, sampler `tell` or report dumping. Those
    are the method's job (a31, a32).

## Tests

`tools/dev/py -m pytest tests/training/test_trainer.py -q`: **18 passed, 1 skipped**
(the CUDA test) in about 10 s. `TABPACK_GPU=1 ... -k cuda`: **1 passed**.
`tools/dev/py -m pytest tests/training -q`: 187 passed, 6 skipped.
`ruff check` and `ruff format --check` pass on the owned files.

What the tests cover:

* The loss decreases, and members beat the majority class.
* All members finish with a small patience.
* `max_epochs` is respected.
* The structure of the result: shapes, dtypes, metrics equal to `compute_metrics`,
  history keys and steps.
* Restored best epochs: the finished predictions and `best_step` match the recorded
  per-epoch evaluation at the first argmax. Dropout makes the best epoch differ from
  the last one.
* Members stop at different epochs, driven by members with lr=0.
* Scripted removal at epochs 2, 3 and 5 with heterogeneous depth, dropout and lr.
  Each finished member equals its best snapshot, and the pack sizes seen by the
  model are {6, 5, 3} for training and {1, 2, 3} for the finished subsets. It is
  never 0.
* Full-batch equivalence: a run where member 1 is removed after epoch 1 matches a
  run without removal for members 0 and 2 (rtol 1e-4).
* MuonAdamWPack with `make_param_groups(muon=True)` and heterogeneous members.
* The online ensemble ends the run: val scores never decrease, and the report ids
  are a subset of the members.
* The online ensemble works while members finish.
* Multiclass and numerical-only data, CPU bf16 autocast, `progress=True`, and a
  single batch per epoch.
* Invalid arguments.
* On return, the model and optimizer hold 0 members.
* Determinism for a fixed seed, and a different result with another seed.

A manual smoke run on real Churn, on GPU with bf16:

* Setup: 16 heterogeneous members, d_block 384, MuonAdamWPack, patience 16,
  max_epochs 40, online greedy ensemble (size 32, patience 32).
* It took 51 s. All 16 members finished.
* Ensemble test accuracy 0.8605; best member 0.863.
* A profile of 5 epochs shows the loop's own overhead is negligible. Most time goes
  to the optimizer step (Muon NS) and to the forward and backward passes.

## Coordination

* Merged: `checkpoint/00b-data-fix`, then a08, a09, a10, a11, a12, a13, a14, a18,
  a19, a20, a22, a15, a07, a27, a21, a25, a17, a26 and a16 once each posted `done`,
  and finally `main`.
* Read a11 #41 and a13 #54 (select only when no graph is alive) and a20 #64
  (PackState API). Both are handled as described above.
* Before a17 and a26 were done, I iterated with an untracked scratch shim. It was
  never committed. The tests now run against the real modules.
* Posted status #79 and `done`. I sent no findings: no dependency bug was found.

## Open issues

* When the online ensemble ends the run, the unfinished members are dropped. This
  matches the official code. Their latest weights are left in `model`.
* The per-forward `.tolist()` sync of `MLPBackbonePack` (a11) and the Muon step
  dominate step time on GPU. They are outside this module.
