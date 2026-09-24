# a16-optim-muon: MuonAdamWPack

## Summary

I implemented `MuonAdamWPack` in `src/tabpack_repro/optim/muon_adamw_pack.py`. It is a
pack of K Muon+AdamW optimizers, and each member can have its own `lr`,
`weight_decay`, `muon_lr` and `muon_scale`.

* **Muon groups** (`group['muon'] is True`) follow the official
  `MuonAdamWPack._step_muon`. The steps are:
  1. decoupled weight decay `1 - muon_lr * wd`, skipped when either value is a
     float 0;
  2. momentum `lerp`;
  3. optional Nesterov `lerp`;
  4. batched bf16 Newton-Schulz (a14);
  5. spectral scale;
  6. per-member `muon_lr`. When `muon_lr` is None, `lr` is used instead.
* **All other groups** call a15's `adamw_update_` and use the same state layout as
  `AdamWPack`.

The class inherits from a15's `_PackOptimizer` base. It passes 30 tests.

## Files

* `src/tabpack_repro/optim/muon_adamw_pack.py`: `MuonAdamWPack`, plus the private
  helpers `_bcast`, `_default_muon_scale` and `_check_unit_interval`.
* `tests/optim/test_muon_adamw_pack.py`: 30 tests, all on CPU and fast (about 5 s).
* `evidence/agents/a16-optim-muon.md`: this report.

## Design decisions

* **Layout: the Newton-Schulz orthogonalization.** Our `LinearPack.weight` is
  `(K, in, out)`, the transpose of the official `(K, out, in)`. Newton-Schulz does
  not depend on the layout:
  * NS(G^T) = NS(G)^T. Each quintic iteration `aX + (bA + cA^2)X`, with
    `A = X X^T`, equals the transpose of the same iteration on `X^T`, because
    `(X X^T)^k X = X (X^T X)^k`. The Frobenius normalization does not change under
    a transpose.
  * NS itself turns tall matrices into wide ones. For non-square weights, both
    layouts therefore orthogonalize the same wide matrix. The tests confirm that the
    results are **bit-identical** over 5 steps.
  * Square weights iterate on `X^T X` where the official code uses `X X^T`. The two
    differ only by bf16 rounding. The NS polynomial amplifies it to about 5% of the
    total parameter movement (about 3e-3 to 8e-3, measured for 8x8, 16x16 and
    32x32). No transpose trick removes this.
* **Layout: `muon_scale`.** The scale is not symmetric. The official
  `max(1, size(-2)/size(-1))**0.5` means sqrt(max(1, out/in)) for `(out, in)`. For
  `(K, in, out)` the same quantity is `size(-1)/size(-2)`, and that is what the
  default reads. It agrees with a17's explicit `muon_scale`. A test uses in=3,
  out=12, where the correct scale is 2 and a layout mistake would give 1.
* **Precision.** The NS output (bf16) is cast back to `p.dtype`. The scale and the
  learning rate are then applied in fp32 as one per-member coefficient. The official
  code applies them in bf16, with `update *= scale` and `update.mul_(lr)`. Our
  version is slightly more accurate and differs from the official one by up to one
  bf16 ulp of the update per step.
* **`p.grad` is not changed.** The official code runs `grad.lerp_(buf, momentum)`,
  which overwrites `p.grad`. We compute the Nesterov mix out of place. The numbers
  are the same.
* **State layout.** The class inherits from `_PackOptimizer`, so it matches
  `AdamWPack`, and a17's `optimizer_select_` sees a single layout:
  * Muon params keep only `muon_momentum_buffer`. They have no step.
  * AdamW params keep `exp_avg` and `exp_avg_sq`.
  * With `shared_step=False`, each AdamW param also keeps its own `step`, an int64
    `(K,)` tensor.
  * With `shared_step=True`, there is one step for the whole optimizer, in
    `state['__shared__']`. It advances only on a `step()` call where at least one
    **non-Muon** param has a gradient, because only the AdamW bias correction uses
    it. a15 agreed to this.
  * This differs from the official per-param int only when some param has no
    gradient on some steps. That choice belongs to a15.
* **Hyperparameters.** These go through the base class:
  * Floats stay floats.
  * Sequences and tensors become float32 `(K,)` tensors on the params' device, with
    one copy per group. The defaults are kept on the CPU, so `optimizer_select_` can
    slice them.
  * `muon_lr` and `muon_scale` are declared nullable (`_nullable_keys`).
  * Muon groups must contain 3-D params. `_normalize_group` checks this, and an
    invalid group added later is dropped rather than left half-registered.
  * `muon_momentum`, `beta1` and `beta2` must be in [0, 1), `eps` must be > 0 and
    `muon_ns_steps` must be >= 0.

## Tests

`tools/dev/py -m pytest tests/optim/test_muon_adamw_pack.py -q` gives **30 passed**.
`tools/dev/py -m pytest tests/optim -q` gives 101 passed and 3 skipped (CUDA only).
`ruff check` and `ruff format --check` are clean on the owned files.

What the tests cover:

* **Naive per-member reference.** The test contains its own reference: a 2-D Muon
  in the nn.Linear `(out, in)` layout, with Keller Jordan's NS, run member by
  member.
  * Shapes 4x12, 12x4 and 5x7, with and without Nesterov, over 5 steps: the packed
    optimizer matches with atol 1e-5. It is bit-identical in practice.
  * `ns_steps` in {1, 3} with momentum 0.8: same result.
  * Square 8x8: exact against a reference that runs NS on the transpose, and within
    10% of the movement against the official orientation.
* **`muon_scale`.** Leaving it out, setting None, a float and an explicit `(K,)`
  tensor of the default value all give the same result. The default reads
  out/in of our layout. Explicit per-member scales match the reference.
* **Per-member hyperparameters.**
  * With a zero gradient, each member becomes `p_k * (1 - muon_lr_k * wd_k)`, which
    shows that weight decay uses `muon_lr` and not `lr`.
  * A member with `muon_lr = 0` stays frozen.
  * Members are independent of each other.
  * `muon_lr=None` gives the same result as `muon_lr=lr`.
  * Float values match per-member tensors that hold the same value.
  * Values are stored as float32 `(K,)` tensors, with separate storage per group.
* **Non-Muon groups against `AdamWPack`.** Parameters, `exp_avg`, `exp_avg_sq` and
  the step layout are **bit-equal** (`torch.equal`), for `shared_step` True and
  False. The test includes a param that has no gradient on one step.
* **Other behaviour.**
  * The shared step counts only AdamW updates.
  * Params and groups whose gradient is None are skipped: no update and no state.
    The Muon buffer stays untouched.
  * `p.grad` is not modified.
  * A closure with real gradients lowers the loss.
  * `state_dict` round trip and `copy.deepcopy` both work.
  * Invalid hyperparameters and shapes raise errors, and an invalid group added
    later leaves the optimizer unchanged.
* **Mutation checks.** These were run by hand and not committed. Each mutation below
  makes tests fail:
  * swapping the default scale to in/out (11 failures);
  * using the wrong Nesterov weight (7);
  * dropping the weight decay (12);
  * always using `muon_lr` (1);
  * counting the shared step on Muon-only steps (1).

## Coordination

* Sent:
  * #24 status, when I started.
  * #25 question to a15 about the state layout.
  * #31 answer to a17 (#27) on the state layout and the `muon`/`muon_scale`
    defaults.
  * #83 question to a15 about reusing `_PackOptimizer`.
  * #96 answer to a15.
  * #109 done.
* Received:
  * a17 #27 on the group layout from `make_param_groups`.
  * a09 #34, confirming the `(K, in, out)` layout.
  * a14 #48 and #51 on the NS semantics: it returns bf16 and the caller casts.
  * a15 #84, #92, #100 and #103 on the state layout, the reusable helpers and
    `_nullable_keys`.
  * integrator #60 on the data-package fix.
* Merges (all posted `done` or were tagged by the integrator):
  * `feat/a14-optim-ns`
  * `checkpoint/00b-data-fix`
  * `feat/a15-optim-adamw`, merged twice: at ee9c2d7, and at e72130c for
    `_nullable_keys`.

## Open issues

* **The class relies on a15's private helpers.** They are `_PackOptimizer`,
  `_normalize_group`, `_nullable_keys`, `_advance_shared_step`,
  `_init_param_state`, `_next_param_step` and `_group_value`. a15 said in #100 that
  they will stay stable. If `adamw_pack.py` changes, check this class too.
* **Parity with the official code (a40) needs a tolerance.** Newton-Schulz on square
  hidden weights differs from the official code by bf16 rounding (about 5% of the
  per-step Muon movement). Scale and learning rate are applied in fp32 here but in
  bf16 officially.
* **The `muon_lr` annotation does not mention None.** It is `PerMember`, but the
  code accepts None at runtime, following the official semantics. The frozen
  signature was not widened.
