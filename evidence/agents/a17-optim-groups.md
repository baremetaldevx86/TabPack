# a17-optim-groups: optimizer param groups and member removal

## Summary

Implemented `make_param_groups(model, *, muon)` and `optimizer_select_(optimizer,
keep_idx)` in `src/tabpack_repro/optim/pack_utils.py`, following the frozen contract.
`optimizer_select_` depends only on the pack invariant (per-member tensors have the
pack dim first). It does not use AdamWPack or MuonAdamWPack internals, and it works
with both of their state layouts. 37 tests pass. They include end-to-end checks that
training after a removal matches a pack that never contained the removed members.

## Files

* `src/tabpack_repro/optim/pack_utils.py`: implementation (plus private helpers
  `_make_muon_scale`, `_infer_old_pack_size`, `_iter_state_values`,
  `_iter_hyperparameters`, `_is_packed`).
* `tests/optim/test_pack_utils.py`: tests.
* `evidence/agents/a17-optim-groups.md`: this report.

## Design decisions

* **Group layout.** Groups come in the order of the official
  `lib.optim.utils.make_parameter_groups`: default, then zero weight decay, then
  custom (Muon) groups. Muon groups are `{'params': [block.linear.weight], 'muon': True,
  'muon_scale': (K,) float32}`, one for every block in `model.backbone.blocks`, like
  the official `_iter_blocks()`. The zero weight decay group is `{'params': [...],
  'weight_decay': 0.0}` (a Python float) and holds every parameter with `ndim <= 2`,
  which in a ModelPack means exactly the backbone and head biases. That matches the
  official `name.endswith('bias')` rule, because our model has no norms or embeddings.
  Empty groups are left out, and non-Muon groups have no `'muon'` key (a16 defaults it
  to False).
* **Deterministic order.** Inside each group, parameters follow `model.parameters()`
  order, which also removes duplicate shared parameters. The official code builds the
  zero weight decay list from a `frozenset`, so its order can change between runs.
* **muon_scale** = `sqrt(max(1, out/in))` with our `(K, in, out)` weight layout,
  so it is `W.shape[2] / W.shape[1]`. It is computed in float32 exactly as the
  official `_make_muon_scale` does it (float32 divide, `clamp_(min=1)`, `sqrt_`), so it
  is bit-identical. Only block 0 can have a scale above 1 (when `d_in < d_block`).
* **Old pack size is inferred.** `pack_select_` keeps each Parameter object and
  slices it before `optimizer_select_` runs, so parameters already have the new K.
  The old K is read from the dim 0 of every tensor with ndim >= 1 in `optimizer.state`
  (per-param dicts, and entries under non-param keys such as a15's
  `'__shared__': {'step': int}`), in the param groups and in `optimizer.defaults`.
  All of these must agree; if they do not, a `ValueError` is raised. Inferring the old
  K from the stored tensors, rather than from how the new size differs, also makes
  reorderings work, including a pure permutation where K stays the same.
* **What is sliced.** Every tensor with ndim >= 1 whose dim 0 equals the old K, in
  per-param state, top-level state entries, group hyperparameters and `defaults`.
  Slicing `defaults` (a15 and a16 keep per-member lr/weight_decay/muon_lr there as
  CPU (K,) tensors) keeps a later `add_param_group` valid. Python scalars and 0-dim
  tensors are left alone. Advanced indexing gives every result its own storage, so
  tensors that used to be shared between entries are not left aliased. The index
  tensor is moved to each value's device, cached per device.
* **Guards.** The function raises `ValueError` if a parameter's dim 0 is not
  `len(keep_idx)` (meaning `pack_select_` was not called first), and `ValueError` if
  `keep_idx` is not a 1-D integer tensor. It raises `IndexError` if an index is
  outside `[0, old K)`; negative indices count as out of range. If there is no
  per-member tensor at all (fresh optimizer, scalar hyperparameters), it does nothing.
* **Compared with the official `optimizer_pack_remove`.** Same approach (slice every
  ndim > 0 tensor in groups and state). Differences: the official version swaps in new
  Parameter objects through `old_to_new`, while ours keeps parameter identity, per the
  a13 contract. We also slice `defaults` and optimizer-level state, and we check that
  the pack sizes are consistent.

## Tests

`tools/dev/py -m pytest tests/optim/test_pack_utils.py -q` gives **37 passed** (about 5 s,
CPU, fp32). This was run on this branch after merging a09 to a16.

* Param groups: every parameter appears exactly once (muon on and off; uniform and
  heterogeneous `n_blocks`). Biases are in the float `weight_decay=0.0` group and no
  other group overrides weight decay. With `muon=False` the layout is exactly two
  groups. Muon groups: one per block, in order, with `muon_scale` checked for shape,
  dtype, device and value for `d_block` of 16, 32 and 4 against `d_in=8`. `muon_scale`
  is also checked to be bit-exact against the official float32 formula. Empty groups
  are left out (checked on a duck-typed model without biases).
* `optimizer_select_` on a synthetic optimizer that holds every layout: per-param
  (K, ...) tensors, (K,) int64 steps, int steps, 0-dim tensors and strings; top-level
  int, 0-dim and (K,) entries; (K,) tensors in groups and defaults. Tested with keep
  sets `[1,3]`, `[3,0,2]`, the identity and `[2]`. Also covered: no aliasing, a no-op
  on SGD, the errors when `pack_select_` was not called or indices are invalid, and
  the error when state tensors disagree on K.
* End to end, with AdamWPack and MuonAdamWPack, shared_step True and False,
  `n_blocks=[2,1,3,2]`, per-member lr/wd/muon_lr, and keep sets `[0,2]`, `[2,0,3]`
  (reordered) and `[1,2]`. The run is 4 steps, then `pack_select_` and
  `optimizer_select_`, then 4 more steps. Afterwards the parameters equal those of (b)
  a pack that never held the removed members and of (c) the full pack sliced at the
  end. The full optimizer `state_dict` also equals (b)'s. A second test removes twice
  (4 to 2 to 1 members) with both optimizers.
* Mutation checks (run by hand, then reverted): slicing `value[:len(keep)]` instead
  of `value[keep]` makes all 14 end-to-end tests fail, and so does not slicing the
  group hyperparameters.
* `ruff check` and `ruff format --check` pass on both owned files.

## Coordination

* Posted status #21, #82.
* Asked a15 (#26) and a16 (#27) about their state layouts. Answers:
  * a16 (#31): Muon state is `muon_momentum_buffer`; the AdamW part uses a15's
    layout; `muon` defaults to False; the default `muon_scale` matches ours.
  * a15 (#85): `exp_avg` and `exp_avg_sq`; a (K,) int64 `step` when
    `shared_step=False`; `optimizer.state['__shared__'] = {'step': int}` when
    `shared_step=True`; floats stay floats; `defaults` may hold CPU (K,) tensors;
    there are no other ndim >= 1 tensors.

  `optimizer_select_` handles all of these without any special cases.
* Merged these `done` branches: feat/a09-nn-linear, feat/a13-nn-packops,
  feat/a14-optim-ns, feat/a12-nn-model (which includes a10 and a11),
  feat/a15-optim-adamw (which brought the integrator's data-package fix e8e129c) and
  feat/a16-optim-muon.
* Before those merges, I worked around the `.gitignore` `data/` blocker (#6/#7) with
  an untracked, ignored local copy of the skeleton data package. It was never
  committed and was replaced by the tracked files from the merge.

## Open issues

* None in the owned code. For a23 and a31/a32: call `pack_select_(model, keep)`
  first, then `optimizer_select_(optimizer, keep)` with the same `keep_idx`, and only
  when no autograd graph is alive (see a13 #54). Calling it twice with a non-identity
  permutation cannot be detected and would permute twice.
* a15's private `AdamWPack._pack_size` stays at the construction K. a15 says it is
  only used for empty groups in `add_param_group`, so it is left alone.
