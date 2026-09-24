# a18-train-batches

## Summary

Implemented `generate_member_batches` and `epoch_size` in
`src/tabpack_repro/training/batches.py`. Each epoch draws one
`torch.rand((K, N), generator=g, device=g.device)`, argsorts every row, and splits
along dim 1 into chunks of `batch_size`, returned as a `list` of int64 `(K, b)` tensors.
This matches the RNG consumption of the official `generate_training_batches`
(`.reference/tabpack/src/project/tabpack.py`, lines 846-861) exactly, so a seeded
generator gives the same batch sequence as the official code.

## Files

- `src/tabpack_repro/training/batches.py` (implementation)
- `tests/training/test_batches.py` (36 tests)
- `evidence/agents/a18-train-batches.md` (this report)

## Design decisions

- Contract kept as is (keyword-only args, `list[Tensor]` return).
- Validation: `train_size`, `batch_size`, `pack_size` must be positive `int`s
  (`bool` rejected), raising `TypeError` for the wrong type and `ValueError` for values
  <= 0. `generator` must be a `torch.Generator`. `epoch_size` validates the same way.
- `batch_size > train_size` is allowed and yields a single batch of size `N`, the same
  as the official code.
- Batches live on `generator.device`. The trainer should create the batch generator on
  the same device as the data, as the official code does.
- `epoch_size` uses integer ceiling division (`-(-n // b)`), with no float rounding.

## Tests

`tools/dev/py -m pytest tests/training/test_batches.py -q` gives **35 passed,
1 skipped** (the CUDA test when CUDA is hidden) in 3.8 s.
`TABPACK_GPU=1 tools/dev/py -m pytest tests/training/test_batches.py -q -m gpu` gives
**1 passed**.
`ruff check` and `ruff format --check` on the owned files are clean.

Coverage:
- each member's concatenated batches form a permutation of `range(N)`
- members get different orders
- a seeded generator is deterministic, and consecutive epochs differ
- last-batch sizes
- `epoch_size == ceil`
- `K = 1`
- argument validation
- equality with a direct re-statement of the official formula over two consecutive
  epochs, including the final generator state
- a CUDA-generator variant

## Coordination

- Posted `status` (#22) at start and `done` at the end. No messages were addressed to
  a18. I merged no peer branches.
- To run the conftest, I copied the git-ignored skeleton package
  `src/tabpack_repro/data` from the main checkout into the worktree without tracking it
  (same `.gitignore` `data/` blocker as board #9/#14). It is not committed.

## Open issues

- I could not import the official `generate_training_batches` directly, because the
  reference package needs `rtdl_num_embeddings`, which is not installed. The parity
  test re-states the official formula instead. A true import-based check belongs in
  `tests/parity/`.
