# a13-nn-packops: agent report

## Summary

Implemented `tabpack_repro.nn.pack_ops`: `get_pack_size`, `make_keep_idx`,
`pack_select_`, `pack_state_dict` and `pack_load_members_`. They work on any
`nn.Module` tree that obeys the pack invariant (dim 0 of every parameter and buffer
is K), on CPU and CUDA, with index tensors on the CPU or on the module's device.
`pack_select_` keeps every `nn.Parameter` object (so optimizer param groups stay
valid) and clears its gradient. It re-assigns sliced buffers without changing
whether they are persistent.

## Files

* `src/tabpack_repro/nn/pack_ops.py`: implementation (contract unchanged; private
  helpers `_PackInvariantError`, `_named_tensors`, `_as_index`, `_check_range`,
  `_index_on_device`).
* `tests/nn/test_pack_ops.py`: toy-pack tests plus ModelPack integration tests.
* `evidence/agents/a13-nn-packops.md`: this report.

## Design decisions

* **Parameter identity.** `param.data = param.data.index_select(0, keep)` and
  `param.grad = None`. `index_select` makes a new contiguous tensor, so the result
  shares no memory with the old storage. The `Parameter` object, its
  `requires_grad`, and its place in optimizer param groups all stay the same.
* **Buffers.** The new slice is assigned with `setattr(submodule, name, slice)`.
  Because the buffer name already exists, `nn.Module.__setattr__` only updates
  `_buffers[name]` and does not touch the non-persistent set, so persistence is
  preserved. Buffers set to `None` are skipped.
* **Shared tensors.** The code walks `module.modules()` itself and slices each
  parameter or buffer once, tracked by identity. A buffer shared by two submodules is
  re-assigned in both and stays shared. (Calling `named_buffers()` with its default
  deduplication would have left the second owner holding the unsliced tensor.)
* **Validation before mutation.** `pack_select_` checks the invariant, the index
  shape and dtype, the range (`IndexError`) and duplicates (`ValueError`) before it
  changes anything, so a bad call leaves the module intact. An empty `keep_idx` is
  allowed and leaves a pack of size 0.
* **Devices.** Each index is validated on a CPU copy, which costs one small sync on
  CUDA. It is then moved at most once to each device it is needed on. This works for
  modules whose tensors sit on different devices, and for index tensors on either the
  CPU or the device. `make_keep_idx` returns its result on `remove_idx.device`.
* **`get_pack_size` errors.** The contract says it "asserts", so it raises an explicit
  exception that subclasses both `ValueError` and `AssertionError`. Unlike a bare
  `assert`, it still fires under `python -O`, and callers can catch it either way.
  It also rejects scalar tensors and modules with no parameters or buffers.
* **`pack_state_dict`** returns detached clones of `named_parameters()` followed by
  `named_buffers()`, including non-persistent buffers. These are the same keys that
  `pack_load_members_` iterates over.
* **`pack_load_members_`** runs `index_copy_` in place under `no_grad` on the
  parameter or buffer itself. This keeps both the object identity and its
  `data_ptr`, and autograd's version counter records the change. It is strict about
  keys (`KeyError`), shape and dtype (`ValueError`), and index range (`IndexError`).
  `state` may live on another device, for example a CPU checkpoint loaded into a CUDA
  model. Duplicate indices are removed first so that CUDA `index_copy_` stays
  deterministic.
* **Differences from the official code** (`.reference/tabpack/src/project/nn.py`):
  * `module_pack_remove` creates new `ParameterPack` objects and returns an
    old-to-new map for the optimizer. Here, identity is kept instead.
  * The official code asserts that at least one member is removed. Here, an empty
    removal is a no-op.
  * The official code touches only `ParameterPack`/`BufferPack` tensors. Here, every
    parameter and buffer counts as a pack tensor, as the invariant says.
  * `module_pack_load_state_dict`'s `state_dict_idx` has no equivalent here: callers
    slice the state instead, as `PackState.select_` does.
  * `make_keep_pack_idx` has the same semantics as `make_keep_idx`.
* **Caveat (reported by a11, #40):** because `param.data` is re-assigned, any
  autograd graph still alive that used the parameters caches their old shapes. A
  backward through such a graph would fail. `pack_select_` must therefore run when no
  such graph is alive, for example after `step()` or after evaluating under
  `no_grad`. This is documented in the docstring and was sent to a23.

## Tests

`tools/dev/py -m pytest tests/nn/test_pack_ops.py -q`: **55 passed, 2 skipped**. The two
skipped tests are the CUDA tests. With `TABPACK_GPU=1 ... -k cuda`, both pass.
`tools/dev/py -m pytest tests/nn -q` after merging a09-a12: 217 passed, 2 skipped.
`ruff check` and `ruff format --check` pass on the owned files.

* Toy nested pack module (parameters, a persistent float buffer, a non-persistent
  int64 buffer, and a `None` buffer):
  * The outputs of kept members are unchanged, bit for bit, for a reordered subset,
    a single member, the identity, and a reversal.
  * Every tensor is sliced in `keep_idx` order.
  * Parameter ids and optimizer groups are kept; gradients are cleared; SGD still
    trains the parameters afterwards.
  * No slice aliases the old storage, and persistence flags are kept.
  * Shared parameters and buffers are sliced once.
  * Removing all but one member, and then all members, works; an empty removal
    leaves the module unchanged.
  * An invalid `keep_idx` leaves the module intact.
  * `pack_state_dict` returns clones; `pack_load_members_` restores only the
    selected members, in place, and bumps the version counter.
  * A round trip and loading a sliced state after `pack_select_` both work.
  * Key, shape and dtype validation are tested.
  * A CUDA test runs with the index tensors on the CPU and on the device, including
    a CPU checkpoint loaded into a CUDA module.
* Real `ModelPack` (n_blocks `[1,3,2,3,1]`, per-member dropout, binclass and
  3-class):
  * After `pack_select_` (in `keep_idx` order), the outputs of kept members are
    unchanged bit for bit, parameter ids are unchanged, and `n_blocks` is sliced.
  * Empty removal and removing all but one member work.
  * Training works after removal.
  * The state round trip works; `pack_load_members_` restores the selected members
    only; loading a sliced state after removal works.

## Coordination

* Posted `status` #17 and `done` #50 (core tests passed).
* Received `finding` #40 from a11 (the live-graph caveat). Answered it in #53 and
  forwarded it to a23 in #54.
* Merged `feat/a09-nn-linear`, `feat/a10-nn-dropout`, `feat/a11-nn-mlp` and
  `feat/a12-nn-model` after each posted `done`, for the ModelPack tests. There were
  no conflicts.
* Blocker #6/#7 (the `data/` gitignore rule): I copied the main checkout's skeleton
  `src/tabpack_repro/data` into this worktree, untracked and ignored, so that
  `tests/conftest.py` imports. None of it is committed.

## Open issues

* None in `pack_ops`. Callers must follow the live-graph caveat above.
