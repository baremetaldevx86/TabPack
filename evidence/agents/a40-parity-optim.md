# a40-parity-optim: optimizer parity with the official code

## Summary

`tests/parity/test_parity_optim.py` has 84 tests. They compare our optimizers
(a14-a17) with the official clone at `05a89e2`, using the `official` fixture.

| Component | Official counterpart | Result |
| :-- | :-- | :-- |
| `zeropower_via_newtonschulz5` | `vendor.muon.zeropower_via_newtonschulz5` | **bit-identical** (`torch.equal`): wide/tall/square/2-D/4-D/rank-1/zero, fp32 and bf16 input, 0/1/5 steps. Our batched result also equals the official one applied per matrix. |
| NS in our (K, in, out) layout | official (K, out, in) | **bit-identical** for non-square matrices (including 384x20). Square matrices are close: about 4% relative Frobenius, bound 10%. |
| `AdamWPack`, float lr/wd, `shared_step=True` | `project.optim.AdamWPack` | **bit-identical** params and moments after 10 steps |
| `AdamWPack`, per-member lr/wd and/or per-param step | same | moments bit-identical; params within the float32 bound (measured <= 10 float32 eps absolute) |
| `MuonAdamWPack`, non-square, per-member lr/wd/muon_lr, `shared_step=True`, scale default or (K,) tensor, Nesterov on/off | `project.optim.MuonAdamWPack` | momentum buffers and Adam moments bit-identical. AdamW params are bit-identical with float hparams. Muon weights are within the bf16 bound, or within the float32 bound when the official code does not round. |
| `MuonAdamWPack`, square weights | same | same orientation: tight bound. Real layouts: displacement within 10% (measured 2-3%). |
| `make_param_groups(model, muon=True/False)` | `tabpack.py` Muon groups + `lib.optim.utils.make_parameter_groups(_default_zero_weight_decay_condition)` | same groups in the same order (default, wd=0.0 biases, one Muon group per block), same keys; `muon_scale` bit-identical to `_make_muon_scale` |
| End to end: ModelPack + groups + MuonAdamWPack, 4 steps, remove 2 members, 4 steps | official ModelPack + `module_pack_remove` + `optimizer_pack_remove` | hyperparameter tensors and all optimizer states bit-identical before and after removal; params within the bounds above |

Mutation checks confirm that the tests catch real errors. Each of these makes tests
fail: a wrong default Muon scale orientation (12 tests fail), flipped Nesterov (28),
Muon using `lr` instead of `muon_lr` (28), and AdamW eps x10 (the exact test fails).

## Files

* `tests/parity/test_parity_optim.py` (new, owned).
* `evidence/agents/a40-parity-optim.md` (this report).

## Design decisions

* **Same inputs, fixed gradients.** Each parameter exists twice, as ours and as an
  official `ParameterPack`. 3-D weights are transposed between the layouts. Each
  step, both copies get the same fresh random gradient (the official Nesterov step
  overwrites `p.grad`, so the gradient is a new tensor every step). The gradients
  do not depend on the parameters, so:
  * optimizer states must be, and are, bit-identical;
  * parameter rounding differences add up linearly without feedback, so an
    "N steps x per-step bound" tolerance is sound.
* **Tolerances** (derivations are in the module docstring and in the helper
  docstrings):
  * `_fp32_atol = 8 * N * eps32 * max(1, max|p|)`. The official code does the (K,)
    scalar math in float32 and applies `(m/d)*step`. Ours uses float64 and applies
    `(m*step)/d`.
  * `_muon_atol = N * lr_k * scale_k * 1.5 * rel + fp32`, with `rel = 2 * 2^-8` when
    the official code rounds the scaled update to bf16. That happens with a tensor
    scale, a Python scale != 1, or a per-member lr (bf16 in-place `mul_`). a16 casts
    to fp32 first, as documented in its report. Otherwise `rel = 2 * eps32` (fused
    `alpha=` vs a separate multiply). The factor 1.5 bounds the entries of the NS
    output, whose measured spectral norm is < 1.21.
  * Square Muon weights. NS(G^T)^T differs from NS(G) by bf16 rounding that the NS
    polynomial amplifies. This cannot be fixed by any choice of layout. Two tests
    cover it:
    * a tight test that feeds the official optimizer our orientation, so NS sees
      the same matrix, which shows that orientation is the only square-specific
      difference;
    * a loose test on the real layouts: per-member displacement error <= 10%.
* **The `muon_scale` key quirk.** Official `tabpack.py` puts `'muon_scale'` into the
  groups, but the official optimizer reads `'muon_update_scale'`, which defaults to
  None. The official run therefore uses the default
  `max(1, out/in)**0.5` of the weight shape. Our optimizer does use `muon_scale`. A
  test (`test_official_ignores_muon_scale_but_default_is_equal`) pins this down and
  checks that the two values agree (rtol eps32) for our fixed-width MLPs. The quirk
  would only matter for zero-padded heterogeneous widths, which we do not use.
* **Optional dependencies.** The `tabpack_official` fixture stubs
  `rtdl_num_embeddings` / `rtdl_revisiting_models` in `sys.modules` (via
  monkeypatch) only if they are missing. Since integrator #133 they are installed,
  so the real packages are used. The clone is never edited.
* `_make_models` seeds inside `torch.random.fork_rng()`, so the global RNG is not
  changed.

## Tests

```
tools/dev/py -m pytest tests/parity/test_parity_optim.py -q
84 passed in 7.89s
tools/dev/py -m ruff check tests/parity/test_parity_optim.py      # All checks passed
tools/dev/py -m ruff format --check tests/parity/test_parity_optim.py  # already formatted
```

## Coordination

* Merged `main` (a14 and a15, then again with a16 and a17 integrated), plus
  `feat/a16-optim-muon` and `feat/a17-optim-groups`, all posted done.
* Read a14 #47: bf16 `(c*A)@A` op order, eps hard-coded. Confirmed bit-identical.
* Used a16's report on the square-weight and bf16 precision notes. My measurements
  agree with them.
* Integrator #133: the rtdl packages are installed, so no stubs are needed.
* Posted status #120, and `done` at the end.
* No findings. No mismatch beyond the documented and intended precision
  differences.

## Open issues

* None blocking. For information only:
  * The official Muon rounds the scaled update to bf16. Ours keeps fp32 (a16's
    choice). The difference is at most about 1% of one Muon step per step.
  * Official AdamW with `shared_step=True` keeps a per-param int step. Ours keeps
    one step for the whole optimizer. They are identical whenever every param has
    a gradient on every step, which is what the tests use. Blocks that stop being
    used never get a gradient again, so the practical results do not change.
