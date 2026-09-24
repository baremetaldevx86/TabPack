# a22-train-stopping

## Summary

Implemented `compute_stop_idx` (early stopping plus the epoch budget) and
`FinishedPool.extend` (the pool of finished members) in
`src/tabpack_repro/training/stopping.py`. Both follow the official
`compute_stop_pack_idx` and `FinalStatePack.extend` in
`.reference/tabpack/src/project/tabpack.py`. The task brief calls the second one
`FinishedStatePack`; in the official code it is named `FinalStatePack`.

## Files

- `src/tabpack_repro/training/stopping.py`: implementation (the contract is unchanged).
- `tests/training/test_stopping.py`: 24 CPU tests and 1 GPU test.
- `evidence/agents/a22-train-stopping.md`: this report.

## Design decisions

- **Stopping rule.** Stop member `i` if `(patience >= 0 and n_bad_updates[i] > patience)`
  or `(max_epochs >= 0 and steps[i] // epoch_size >= max_epochs)`. A negative value
  turns off that criterion. When both are off, or when no member meets either
  criterion, the function returns `None`. Otherwise it returns the stopping members'
  positions as sorted `int64` values from `np.flatnonzero`. I checked this against
  the official function outside the repo: I ran 5000 random cases and the results
  were identical, including every `None` result. The script is not committed,
  because parity tests belong to `tests/parity/**`.
- **Input checks (added beyond the official code).** `n_bad_updates` and `steps` must
  be 1-D arrays of the same shape. `epoch_size` must be greater than 0 when
  `max_epochs >= 0`, so the code never divides by zero. Either violation raises
  `ValueError`.
- **FinishedPool.extend.**
  - `ids` and `steps` are cast to `int64` and appended with `np.concatenate`.
  - The first extend (on an empty pool) stores the caller's tensors as-is in a new
    dict, so the caller's dict is not shared. Device and dtype stay unchanged.
  - Later extends join each part with `torch.cat(dim=0)`. The result stays on the
    device the tensors are on.
  - Finishing order is preserved.
- **Mismatch errors.** An extend fails if it passes a different set of parts than
  the pool already holds (checked once the pool is non-empty or has parts). It also
  fails if `len(ids) != len(steps)`, or if any prediction's first dimension differs
  from `len(ids)`. These errors are raised as `ValueError`, not with a bare `assert`,
  so they still fire under `python -O`. They are raised before anything changes, so
  a failed extend leaves the pool untouched.

## Tests

```
tools/dev/py -m pytest tests/training/test_stopping.py -q
24 passed, 1 skipped (CUDA test) in 0.40s
TABPACK_GPU=1 tools/dev/py -m pytest tests/training/test_stopping.py -q -k cuda
1 passed
tools/dev/py -m ruff check / ruff format --check (owned files): clean
```

The tests cover:

- **Patience:** stopping requires strictly more bad updates than `patience`
  (`>`, not `>=`); `patience=0` stops a member on its first bad update; `-1` turns
  it off.
- **Epoch budget:** `max_epochs=-1` turns it off. The budget uses
  `steps // epoch_size`, tested with epoch sizes that do and do not divide the step
  counts evenly, and with `max_epochs=0`.
- **Stopping results:** both criteria combined with OR; `None` when no member stops
  and for an empty pack; a comparison with the documented rule on 300 random cases;
  inputs are not modified; shape and `epoch_size` errors.
- **FinishedPool:** empty defaults; separate pools do not share state; finishing
  order is kept across three extends; ids and steps are cast to int64; multiclass
  `(M, N, C)` predictions; extending with zero members; errors for missing or extra
  parts and for row-count mismatches, with the pool unchanged afterwards; the first
  extend keeps the original tensors; predictions stay on CUDA (GPU test).

## Coordination

- Merged `checkpoint/00b-data-fix` at the start, as instructed.
- Board: posted `status` (#71) and `done`. No messages were addressed to a22.
- No peer branches merged (no dependencies).

## Open issues

- None for this module. Note for a23: `compute_stop_idx` takes arrays, not a state
  object. Pass `state.n_bad_updates` and `state.steps` (the `PackState` fields).
  The returned positions refer to the current pack, so they can go straight to
  member removal.
