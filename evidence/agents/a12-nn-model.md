# a12-nn-model: OneHotEncoding and ModelPack

## Summary

I implemented `OneHotEncoding` and `ModelPack` in `src/tabpack_repro/nn/model_pack.py`.
`ModelPack` builds the input as `concat([x_num, one_hot(x_cat)], -1)` and runs it
through `MLPBackbonePack` (a11) and a `LinearPack` head (a09). Each of `x_num` and
`x_cat` can be shared rows `(B, f)`, as in evaluation, or per-member batches
`(K, B, f)`, as in training. Shared rows are expanded to `(K, B, f)` as a stride-0
view, so no copy is made. The output is `(K, B)` logits for binclass and regression,
and `(K, B, C)` logits for multiclass. Either input may be `None`, but not both. All
38 tests pass on top of the merged a09, a10 and a11 branches.

## Files

* `src/tabpack_repro/nn/model_pack.py`: `OneHotEncoding`, `ModelPack`, and the
  private helper `_cat_last`.
* `tests/nn/test_model_pack.py`: 38 tests.
* `evidence/agents/a12-nn-model.md`: this report.

## Design decisions

* **Unknown categories.** The official `OneHotEncoding` computes
  `one_hot(x_i, card + 1)[..., :-1]`. That covers only code `== card`, which is what
  its data pipeline emits for an unseen category, and it raises for any code `> card`.
  The contract requires "code >= cardinality -> zeros", so I clamp first:
  `one_hot(x_i.clamp(max=card), card + 1)[..., :-1]`. A scratch check (not committed)
  confirmed that the output is identical to the official module for every code in
  `[0, card]`. Negative codes still raise inside `one_hot`. Float or bool codes raise
  `TypeError`. int32 codes are accepted and cast to long.
* **Pack invariant.** `OneHotEncoding` stores the cardinalities as a plain Python
  list, so it has no parameters or buffers and an empty `state_dict`. The model
  exposes it as `model.cat_encoding`, which is `None` when there are no categorical
  features. `ModelPack` itself stores only Python scalars (`_n_num_features`,
  `_n_cat_features`, `_squeeze_output`). Every parameter and buffer has K first.
  `pack_size` is read from `backbone.pack_size` on every call, and nothing derived
  from K is cached. A test removes members in place, in the style of `pack_select_`,
  and checks that the outputs equal `before[keep]`.
* **Shared inputs: expand without copying.** This works like the official
  `PackView`. When every given input is 2-D, the model concatenates once on
  `(B, d_in)` and then calls `unsqueeze(0).expand(K, -1, -1)`. A test checks that the
  tensor reaching the backbone has `stride(0) == 0`, and for numerical-only models
  that it shares storage with `x_num`. A single input is never passed through
  `torch.cat`, so it is never copied.
* **Mixed inputs.** A shared `x_num` can be combined with a per-member `x_cat`, and
  the other way round. The official code cannot do this: `torch.cat` fails there on
  rank mismatch. We expand the 2-D input as a view, and the concatenation then
  materializes `(K, B, d_in)`, which is needed anyway.
* **No training-mode assertion.** The official `PackView` asserts `self.training`
  for 3-D inputs. I dropped that check, because evaluating per-member inputs is
  harmless and our tests use it (`eval()` + `(K, B, f)`). A 3-D input with the wrong
  K still raises `ValueError`.
* **Head and output.** The head is `LinearPack(d_block, 1 if n_classes in (None, 2)
  else n_classes)`, and a single output unit is squeezed inside `forward`. The
  official code squeezes in `apply_model_impl` instead. `n_classes < 2` raises
  `ValueError`, because the contract's `(K, B, 1)` output for `n_classes=1` would be
  a trap.
* **Validation.** `ValueError` is raised when both inputs are `None`, when an input
  is missing or unexpected for the model's features, when the number of numerical
  features is wrong, and when an input is not `(B, f)` or `(K, B, f)`. These are
  cheap Python checks with no device synchronization.
* **Differences from the official `ModelPack`.** Our module has no `pack_idx`
  argument, because the contract's `forward(x_num, x_cat)` has none. It has no
  `num_embeddings`, which Churn TabPack does not use, and no `max_d_block`. The head
  attribute is named `head` (official: `output`). One-hot is always float32, where
  the official code uses `torch.get_default_dtype()`.

## Tests

`tools/dev/py -m pytest tests/nn/test_model_pack.py -q` gives **38 passed in ~5 s**
on CPU. `tools/dev/py -m pytest tests/nn tests/test_package_imports.py -q` gives 163
passed. `ruff check` and `ruff format --check` pass on both owned files.

The tests cover:
* one-hot values for known codes, zeros for unknown codes (`== card` and `> card`),
  arbitrary leading dims, int32 codes, no parameters or buffers, and bad
  shapes, dtypes and cardinalities;
* the pack invariant and the attributes, output shapes and dtype for
  `n_classes` in {None, 2, 3, 7} × shared/per-member inputs, and models without
  `x_num` or without `x_cat`;
* per-member inputs that repeat the same rows, and mixed inputs, which equal the
  shared output; row routing (member k of a per-member forward equals member k of a
  shared forward on its own rows); the stride-0 expand;
* heterogeneous `n_blocks=[1,3,2,1]` and dropout lists, compared in eval mode with
  hand-written member MLPs that use `blocks[i].linear.weight[k]`; members with p=0
  are deterministic in train mode and p>0 members are not;
* gradients: each member gets a nonzero gradient in every parameter it uses, and
  exactly zero in blocks it skips. A loss on one member leaves the other members
  with zero gradient;
* training: 60 AdamW steps with per-member random batches on
  `make_synthetic_dataset`, K=3 and heterogeneous depths/dropout. Binclass and a
  derived 3-class target must reach less than 0.7× the initial full-train loss. The
  observed ratios over 5 seeds were 0.31-0.35 for binclass and 0.40-0.43 for
  3-class, so the threshold is not flaky.

## Coordination

* Posted `status` (#16) at the start and `done` (#77) once the tests passed.
* Merged `feat/a09-nn-linear` after it posted done (#32). Merged
  `feat/a10-nn-dropout` (#58) and `feat/a11-nn-mlp` (#67) after they posted done.
  There were no conflicts.
* Received no questions addressed to a12. Noted a11's finding #40/#41: remove members
  only when no autograd graph that uses the parameters is alive. This does not affect
  ModelPack, which caches nothing.
* Workaround for the `.gitignore` `data/` blocker (#6/#7/#12): I copied the skeleton
  `src/tabpack_repro/data/` from the main checkout into this worktree **untracked**
  (it is git-ignored and was not committed), so that `tests/_helpers.py` can import
  `PreparedDataset`.

## Open issues

* ModelPack has no `member_idx`/`pack_idx` argument, because the contract is frozen.
  If a later agent (for example the trainer) needs to run only a subset of members
  without slicing, that needs a contract change from the integrator.
* A negative category code raises inside `one_hot`. The data pipeline never emits
  one.
