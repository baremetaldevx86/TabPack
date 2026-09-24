# a53-test-invariants: pack invariants across modules

## Summary

Added `tests/integration/test_pack_invariants.py`, which holds 27 CPU fp32 tests of
the property that makes a pack equal to K independent models. The tests use the
real modules end to end: ModelPack with heterogeneous `n_blocks = (1, 3, 2, 3)`,
AdamWPack and MuonAdamWPack with per-member lr, weight decay and muon_lr,
`make_param_groups`, `generate_member_batches`, `make_pack_loss`, `evaluate_pack`,
`predict_pack`, PackState and the pack_ops removal and checkpoint functions. All
tests pass, and **no invariant violations were found**, so no `finding` messages
were posted. Each member trained inside the pack differs from the same member
trained alone by at most 6e-8 (1 ulp; batched-matmul reassociation), which is well
inside the atol=rtol=1e-6 the tests use.

## Files

* `tests/integration/test_pack_invariants.py`: the tests.
* `evidence/agents/a53-test-invariants.md`: this report.

## Design decisions

* **Reference = member trained alone.** Member k is pulled out of the pack as a
  K=1 ModelPack with `n_blocks=[n_blocks[k]]` and `dropout=[p_k]`. Its weights are
  copied with a plain `load_state_dict` of the `[k:k+1]` slices, so the reference
  does not depend on pack_ops. It is trained with its own hyperparameters on row k
  of the same `(K, b)` batch indices. The test compares every tensor of the K=1
  model with slice k of the pack. Blocks that member k skips exist only in the pack
  and get their own check: exactly zero gradient, and only decoupled weight decay
  `(1 - lr*wd)^T` with untouched moments/momentum.
* **Loss = sum of per-member means** (`make_pack_loss(...)(...).sum()`), as in the
  trainer. Two gradient tests: pack grads == K=1 grads for every member, and
  scaling one member's loss by 1000 or 0 leaves the other members' grads bitwise
  unchanged.
* **Removal** follows the trainer's order: train an epoch, `evaluate_pack` (inference
  mode), `PackState.update`, then `pack_select_` -> `optimizer_select_` ->
  `PackState.select_` with the same keep_idx (a17's order; a11/a13 #40: no live
  graph). The keep cases are `[0, 2]` (drops both 3-block members, so block 2 has
  no member left and its weight gets `grad=None`), `[3, 1, 0]` (reordered) and
  `[2]`. Each case runs with AdamW and Muon, and with shared and per-param steps.
  Kept members' params and `predict_pack` outputs equal those of the member
  trained alone for all epochs.
* **Checkpointing.** `pack_load_members_` of a subset restores those members
  bitwise (params and buffers), leaves the others bitwise untouched, keeps
  Parameter identity, and lets training continue. Loading all members into a
  freshly initialized pack reproduces the checkpointed predictions bitwise.
  PackState is driven by scripted val scores so that each member's best epoch is
  known in advance and every copy path runs: first update, all improved, some
  improved, ties not improving, and a removal between updates. The test then
  asserts `best_step`, `best_val_score`, and `best_predictions ==` the predictions
  recorded at that member's best epoch. Loading `best_model_state` and running
  `predict_pack` reproduces `best_predictions` bitwise. The model is left in train
  mode with dropout > 0 to check that predict_pack switches dropout off and then
  restores the mode.
* **Heterogeneous dropout** `(0, 0.5, 0, 0.25)`. The p=0 members' train-mode
  logits equal their eval-mode logits bitwise, and their grads equal the K=1
  grads. Their trained weights are bitwise identical under two different dropout
  seeds and equal the K=1 model trained without dropout. The test also checks
  that the p>0 members do change with the seed, so it is not vacuous.
* Mutation check (scratch script, not committed). Each of these planted bugs made
  the relevant test fail: swapped optimizer state on removal, a loss that leaks
  across members (mean over members), one dropout rate applied to all members,
  and a PackState.select_ that misorders best_model_state.

## Tests

`tools/dev/py -m pytest tests/integration/test_pack_invariants.py -q` gives
**27 passed** in about 12-27 s, depending on machine load.
`ruff check` and `ruff format --check` pass on the file.

## Coordination

* Merged `main` twice (setup, and again on resume). main already contained every
  dependency: a09-a13, a14-a17, a18, a19, a20, a21.
* Read a20 #65 (PackState semantics), a11 #40 / a13 #53 (pack_select_ only
  without a live autograd graph) and a17 #117 (pack_select_ then optimizer_select_
  with the same keep_idx). The tests follow all three.
* Posted #122 (`status`) giving the scope, so that a50 does not duplicate it:
  invariants only, no run-to-run determinism of train_pack or the methods.
* No findings: no module violated an invariant.

## Open issues

* The tests do not call `train_pack` (a23 was not merged when they were written).
  They re-create its step/evaluate/remove sequence by hand. Once a23 lands, a
  trainer-level check (members that finish at their best checkpoint == members
  trained alone) could be added. That would sit between this file and a50's
  determinism tests.
