# a11-nn-mlp: MLPBlockPack and MLPBackbonePack

## Summary

Implemented `MLPBlockPack` (LinearPack -> activation -> DropoutPack) and
`MLPBackbonePack` (K MLPs with per-member depth and per-member dropout) per the
frozen contract in `src/tabpack_repro/nn/mlp_pack.py`. Each member k applies only
its first `n_blocks[k]` blocks; members that skip a block keep their activations
bit-for-bit and get exactly zero gradient from it. Nothing derived from `n_blocks`
is cached, so in-place member removal (slicing params/buffers along dim 0) works
without any extra hook.

## Files

* `src/tabpack_repro/nn/mlp_pack.py`: implementation.
* `tests/nn/test_mlp_pack.py`: 42 tests.
* `evidence/agents/a11-nn-mlp.md`: this report.

## Design decisions

* **Depth buffer.** `n_blocks` is always a registered, persistent int64 `(K,)`
  buffer, also when the constructor gets an int (broadcast to K). The official code
  keeps `None` in that case; a uniform buffer keeps the pack invariant simple for
  `pack_ops` and `pack_state_dict`. `len(blocks) == max(n_blocks)`. Invalid input
  (wrong length, any value < 1, pack_size < 1) raises `ValueError`.
* **Per-forward schedule, no cache.** Each forward builds `active[i, k] = i <
  n_blocks[k]` on the buffer's device and reads the per-block counts with one
  `.tolist()` (the only host-device sync). For block i: count == K -> the block runs
  on the whole pack with `member_idx=None` (so the all-same-depth case never touches
  `index_select`/`index_copy`); 0 < count < K -> indices via
  `torch.nonzero_static(active[i], size=count)` (sorted, no extra sync), then
  `x.index_copy(0, idx, block(x.index_select(0, idx), idx))`; count == 0 -> stop
  (depth is monotone, so every later block is unused too). The official code caches
  the index lists keyed by the buffer object; the contract forbids caching, and the
  cost is one tiny sync per forward.
* **Exact zero gradients.** Skipping members never enter the block's graph:
  `LinearPack` indexes `weight[member_idx]`, whose backward scatters into zeros, and
  `index_copy` routes the gradient of copied-over rows only through the block.
* **Activation.** One plain module from `make_activation` per block (element-wise
  and stateless, so it serves any member subset). Dropout rates are per member and
  shared by all blocks of that member (same as the official code).
* `MLPBlockPack` also exposes a read-only `pack_size` property (derived from
  `linear.weight`), an additive convenience that matches the other pack modules.
* **Official semantics check (scratch, not committed; a39 owns parity tests).** With
  `torch.manual_seed(s)` our constructor and the official
  `project.nn.MLPBackbonePack(..., max_n_blocks=max(n_blocks), dropout=[p]*K)` give
  identical weights (ours = official transposed) for list and int `n_blocks`; with
  copied weights and dropout 0, forward output, input gradient and every weight/bias
  gradient are bit-identical (max abs diff 0.0, CPU fp32, `n_blocks=[1,3,2,3,1,2]`).

## Tests

`tools/dev/py -m pytest tests/nn/test_mlp_pack.py -q` -> `42 passed in 1.67s`
(with the real a09 LinearPack and a10 DropoutPack merged; `tools/dev/py -m pytest
tests/nn -q` -> `124 passed`; ruff check and format --check clean on owned files).

Coverage:
* `MLPBlockPack` vs plain `nn.Linear` + activation, with and without `member_idx`.
* Structure: block shapes (block 0 `d_in -> d_block`), buffer dtype, registration and
  presence in `state_dict`, pack invariant, int broadcast, per-member dropout buffers,
  `ValueError` on invalid `n_blocks`.
* Equivalence with K independent plain-PyTorch MLPs of different depths built from
  copied weights (ReLU/GELU/SiLU x 5 depth patterns, dropout 0, fp32, atol 1e-5),
  including an expanded shared input `(B, d) -> (K, B, d)` as ModelPack passes it.
* Gradients: input, weight and bias gradients match the independent MLPs; members
  that skip a block have exactly zero weight/bias gradient (`torch.equal`), and their
  outputs equal block 0's output bit-for-bit.
* Schedule: forward pre-hooks record `member_idx` per block: all-same-depth -> every
  block gets `None` (fast path); mixed depths -> block i gets exactly the sorted
  members with `n_blocks > i`; K=1.
* In-place removal (params sliced via `param.data = param.data[keep]`, buffers
  replaced): outputs equal the pre-removal outputs of the kept members for 5 keep
  patterns (incl. removing all deepest members, removing all shallow members,
  reordering, a single member); unused trailing blocks are not called; gradient
  masking still holds; repeated removal and an in-place edit of `n_blocks` take
  effect on the next forward. `.double()` keeps `n_blocks` int64.
* Dropout: eval mode is the identity (deterministic, equals the reference); in train
  mode members with p=0 are bit-for-bit unchanged and members with p>0 change and
  vary between calls; the empirical drop rate (~0.3) and the 1/(1-p) scaling hold.

## Coordination

* Posted `status` (#15) at start.
* Merged `feat/a09-nn-linear` after a09 posted `done` (#32); merged
  `feat/a10-nn-dropout` after a10 posted `done` (#58). a13 answered the finding
  below (#53) and documented the caveat in `pack_select_`.
* Before the peers landed, the tests were run against throwaway contract-conforming
  stand-ins injected by a scratch pytest plugin (never committed).
* `finding` to a13-nn-packops (#40) and a23-train-trainer (#41): `param.data =
  param.data[keep_idx]` with a new dim-0 size breaks the next backward if an autograd
  graph that used the param is still alive (the cached AccumulateGrad node remembers
  the old shape). Removal must happen when no such graph is alive (after
  backward/step; evaluate under `no_grad`). My removal tests run the pre-removal
  forward under `no_grad` for this reason.
* `status` to a39-parity-nn (#55): how to line up our backbone with the official one
  and the bit-exact results above.
* The worktree lacks `src/tabpack_repro/data/` (git-ignored skeleton bug, board #6,
  #7, #12); I copied the skeleton package from the main checkout as untracked,
  ignored files only so `tests/conftest.py` imports. Nothing of it is committed.

## Open issues

* None in MLPBackbonePack. The forward does one host-device sync per call (reading
  per-block member counts), as the contract forbids caching; negligible next to a
  training step.
