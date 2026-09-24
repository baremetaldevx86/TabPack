# a20-train-state: PackState

## Summary

Implemented `PackState` (`src/tabpack_repro/training/state.py`), the per-member
book-keeping of the running pack. It follows the semantics of the official
`StatePack` (`.reference/tabpack/src/project/tabpack.py`: `update`, `remove`,
`validate`). The implementation is our own; nothing was copied. The frozen contract
is unchanged. I added `validate()` and some private helpers.

## Files

* `src/tabpack_repro/training/state.py`: `PackState` (`pack_size`, `step`, `update`,
  `select_`, `validate`).
* `tests/training/test_state.py`: 25 tests (1 GPU).
* `evidence/agents/a20-train-state.md`: this report.

## Design decisions

* **Host/device split.** `ids`, `steps`, `n_bad_updates` and `best_step` are int64
  numpy arrays, and `best_val_score` is a float64 numpy array. They are plain public
  attributes, as the contract lists. `best_predictions` and `best_model_state` are
  dicts of tensors on the device of the tensors passed to the first `update`.
* **Transfers.** `update` makes exactly one device-to-host transfer, which converts
  `val_scores` to float64 on the CPU. All comparisons run on the host. When only some
  members improved, the improved index goes to each device with one non-blocking
  host-to-device copy, cached per device. A non-blocking copy is safe from pageable
  memory, and it avoids a second sync. A GPU test uses `torch.cuda.set_sync_debug_mode('warn')`
  to check that there is exactly one synchronizing op. A blocking index copy would
  make that count 2, which I checked by hand.
* **In-place snapshots.** The first update clones every input tensor with
  `detach().clone()`. After that:
  * if every member improved, the snapshots are updated with `copy_`;
  * if only some improved, they are updated with `dst.index_copy_(0, idx, src.index_select(0, idx))`;
  * if none improved, nothing on the device changes.

  Storage pointers stay the same across updates, and a test checks this. The work
  runs under `torch.no_grad()`.
* **The first evaluation always improves.** A member improves when
  `(best_step < 0) | (score > best_val_score)`, which is a strict comparison. So a
  first score of `-inf` or NaN still counts as an improvement. As in the official
  code, a NaN best is never beaten later (`x > nan` is False), so that member only
  collects bad updates and is stopped by patience. The contract says `best_step` is
  -1 before the first evaluation. The official code uses `int64.min` here instead.
* **Checks before mutation.** After the first update, the keys and shapes of
  `predictions` and `model_state` must match the stored snapshots. This is checked
  before any field changes, so a rejected call leaves the state intact.
* **Return value of `update`.** It is a **CPU** bool `(K,)` tensor made with
  `torch.from_numpy`, so returning it costs no extra transfer. You can index CUDA
  tensors with it, but `torch.where(mask, cuda_a, cuda_b)` needs `mask.to(device)`
  first.
* **`select_(keep_idx)`.**
  * `keep_idx` may be a CPU tensor, a device tensor or a numpy array. It must be a
    1-D integer index; bool masks and floats are rejected.
  * It must be unique and in range. The order is kept, and an empty index is allowed.
  * It costs at most one device-to-host transfer. The host arrays use fancy
    indexing, which copies. Device tensors use `index_select` with the index already
    on their device, so the new tensors hold no views of the old storage.
  * It works before the first update too.
* **`validate()`.** It asserts shape `(K,)` and the dtype of every host array,
  `ids >= 0` and unique, `steps >= 0`, `n_bad_updates >= 0`,
  `-1 <= best_step <= steps`, and `best_val_score == -inf` with `n_bad_updates == 0`
  for members never evaluated. It also asserts that every tensor has `shape[0] == K`.
  The official code has one more check (`steps > 0` at `update`/`remove`) that I
  left out, because the contract does not require it.

## Tests

```
tools/dev/py -m pytest tests/training/test_state.py -q
24 passed, 1 skipped (gpu)
TABPACK_GPU=1 tools/dev/py -m pytest tests/training/test_state.py -q
25 passed
tools/dev/py -m ruff check / ruff format --check (owned files): clean
```

The tests cover:
* init;
* `step`;
* the first update improving every member, including -inf and NaN scores;
* the first update cloning its inputs;
* strict improvement;
* NaN stickiness;
* `n_bad_updates` counting and reset over a 6-update history;
* a 20-update randomized check that the best slices and scores follow improvements
  only, with in-place storage;
* no aliasing of later inputs;
* shape and key checks, and that a rejected update leaves the state intact;
* `select_`: all fields and order across 5 index sets × (tensor, numpy), continued
  tracking after a selection, selection before the first update, invalid indices;
* `validate` failure modes;
* the empty pack;
* CUDA: device placement, the sync count, and `select_` with a CUDA index.

## Coordination

* Board: posted `#28 status started`, then `done`.
* No messages were addressed to a20.
* No peer branches were merged, because a20 has no dependencies.
* Workaround for the blocker the data agents posted (#6, #7, #12, #20: `.gitignore`
  `data/` hides `src/tabpack_repro/data`, so `tests/conftest.py` cannot import). I
  copied the skeleton `src/tabpack_repro/data/*.py` from the main checkout into my
  worktree as ignored, untracked files, only so that pytest can run. I did not commit
  them.

## Open issues

* `update` returns its mask on the CPU. The contract does not name a device, and
  a23 and a53 should know this.
