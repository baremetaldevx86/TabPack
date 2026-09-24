# a39-parity-nn report

## Summary

Wrote the numerical parity tests between `tabpack_repro.nn` and the official pack
modules (`project.nn` and `project.tabpack.ModelPack` at reference commit `05a89e2`).
The tests run in float32 on CPU. **The two implementations match bit for bit on every
compared path:** same-seed initialization, forward outputs, input and parameter
gradients, train-mode dropout under equal seeds, and member removal, selection and
partial checkpoint loading. No mismatch was found, so no `finding` was posted.

## Files

* `tests/parity/test_parity_nn.py`: 74 tests plus a small weight-copy and comparison
  helper. The module docstring documents the layout mapping.
* `evidence/agents/a39-parity-nn.md`: this report.

## Design decisions

* **Explicit layout mapping, checked for completeness.** `linear_pairs(ours, official)`
  returns (our `LinearPack`, official `LinearPack`) pairs. A `LinearPack` maps to
  itself, `backbone.blocks[i].linear` maps to `blocks[i].linear`, and our `head` maps
  to the official `output`. The helper asserts that the pairs cover every parameter on
  both sides, so a new or renamed parameter cannot slip through uncompared.
  `copy_official_to_ours_` sets `ours.weight = official.weight.transpose(-2, -1)`
  (official `(K, out, in)`, ours `(K, in, out)`) and copies the bias `(K, out)`
  unchanged. `assert_params_same(..., grad=True)` compares the gradients through the
  same mapping. It also requires unused trailing blocks to have `None` gradients on
  both sides.
* **Parity with a distinct source, not only same-seed init.** Most tests fill the
  official parameters with `randn` values and then copy them, so parity does not
  depend on the initialization matching. Separate tests check that initialization is
  identical under the same `torch.manual_seed` (LinearPack, the backbone with list or
  int `n_blocks`, and the full ModelPack). The LinearPack init test also checks that
  both constructors leave the global RNG in the same state.
* **Bit-exact comparisons.** `assert_same` uses `torch.equal` and reports the maximum
  absolute difference on failure. It also checks that `None` matches `None`, and that
  shape and dtype agree. The module constant `EXACT = False` switches every check to
  `rtol = atol = 1e-6`, in case a different BLAS backend ever breaks bit identity for
  bmm on transposed versus contiguous weights.
* **Official quirks that the tests handle (not bugs):**
  * Official `MLPBackbonePack` needs `max_n_blocks` for list `n_blocks`, and its
    `n_blocks` is `None` for an int. Ours is always an int64 `(K,)` buffer.
  * With a float `p`, the official `DropoutPack` uses `F.dropout`, which draws from the
    RNG in a different pattern. So train-mode RNG parity uses the list form, which is
    the form TabPack uses.
  * Official logits are `(K, B, 1)` for binclass and regression, and its
    `apply_model_impl` squeezes them. The tests squeeze too.
  * Official one-hot output is int64, and its ModelPack casts it to float32.
* **Documented divergence.** For a code greater than the cardinality, the official
  `OneHotEncoding` raises (`one_hot` rejects it), while ours encodes all zeros (a12
  clamps). A test pins both behaviors. Codes equal to the cardinality (the official
  unknown code) give identical results.
* **Removal, selection and loading are the official training paths.**
  * `module_pack_remove(m, remove_idx)` is compared with
    `pack_select_(m, make_keep_idx(K, remove_idx))`. The comparison includes a removal
    that leaves the last block unused by every member (`[1, 3]`), a removal down to one
    member, and a following training step (dropout on) with its gradients.
  * `module_pack_select` (temporary, eval only) is compared with `pack_select_` on a
    deepcopy, with sorted and unsorted indices.
  * `module_pack_load_state_dict(m, sd, pack_idx=stop)` is compared with
    `pack_load_members_(m, pack_state_dict(src), stop)`. This is the official "evaluate
    the best checkpoints of stopped members" step.
* After the coordinator's note (#133) that `rtdl_num_embeddings` and
  `rtdl_revisiting_models` are locked in the parity group, the `otp` fixture imports
  `project.tabpack` directly, without `sys.modules` stubs.

## Tests

```
tools/dev/py -m pytest tests/parity/test_parity_nn.py -q   -> 74 passed in ~2-5 s
tools/dev/py -m ruff check tests/parity/test_parity_nn.py  -> All checks passed!
tools/dev/py -m ruff format --check tests/parity/test_parity_nn.py -> already formatted
```

Coverage:

* LinearPack: init, 6 cases.
* LinearPack forward and backward: `member_idx` None, unsorted, single and duplicate
  indices, with and without bias; 8 cases.
* DropoutPack: eval with list and float p, p=0 in train mode, and train mode with
  equal seeds (with and without `member_idx`, including the gradient and the RNG draw
  count).
* MLPBackbonePack: init, 2 cases. Eval forward and all gradients: 4 `n_blocks`
  settings times ReLU/GELU/SiLU. One train-mode case with dropout.
* OneHot: `(B, c)` and `(K, B, c)` inputs with unknown codes.
* ModelPack:
  * init for list/int `n_blocks` times n_classes None/2/3;
  * eval on shared rows for num+cat, num-only and cat-only times n_classes None/2/3;
  * a training step for shared and per-member rows.
* Member removal (5 cases), selection (3 cases) and partial loading (1 case).

A mutation check (a scratch copy that perturbs one copied weight by 1e-6) makes 36
tests fail, so the comparisons are not vacuous.

## Coordination

* Read: a09 #33 (weight layout, RNG-exact init), a10 #59 (dropout RNG parity with
  list p), a11 #55 (backbone mapping, `max_n_blocks`), and integrator #133 (parity
  deps installed).
* Posted: #123 (status: started), and `done` at the end.
* Merges: `main` twice, at setup and after #133. Main includes a09-a13.
* No findings: all compared behavior is bit-identical.

## Open issues

* None blocking.
* The bit-exact matmul comparisons assume the same BLAS kernels give identical results
  for transposed and contiguous weights. They do on this machine. If CI ever disagrees,
  set `EXACT = False` (tolerance 1e-6).
