# a15-optim-adamw: AdamWPack

## Summary

Implemented `adamw_update_` and `AdamWPack` in `src/tabpack_repro/optim/adamw_pack.py`.
The update matches K independent `torch.optim.AdamW(foreach=False)` optimizers
member by member: within 1e-6 with per-member lr / weight decay, and bit-exact with
float hyperparameters and a shared step. It works with 1D, 2D and 3D packed params,
per-group float overrides, a shared or per-param step, `grad is None`, member
removal, device moves, closures, `state_dict`/`load_state_dict` (also through
`torch.save`/`torch.load(weights_only=True)`) and `copy.deepcopy`.

## Files

* `src/tabpack_repro/optim/adamw_pack.py`: `adamw_update_`, `AdamWPack`, and the private
  helpers `_PackOptimizer` (a base class a16 can reuse), `_to_per_member` and
  `_SHARED_STATE_KEY`.
* `tests/optim/test_adamw_pack.py`: 38 tests (2 of them need CUDA).
* `evidence/agents/a15-optim-adamw.md`: this report.

## Design decisions

* **Hyperparameter storage.** Every param group is normalized in `add_param_group`,
  so the rules cover the constructor defaults, per-group overrides and groups added
  later:
  * a Python number stays a `float` and applies to all members. For example, the
    bias group uses `'weight_decay': 0.0`.
  * a sequence, array or tensor becomes a new float32 `(K,)` tensor. Each group has
    its own storage, so it never aliases the input or another group. The tensor sits
    on the device of the group's params.
  * `step()` moves a `(K,)` tensor that sits on another device than its param, for
    example after `model.to(device)` or a checkpoint loaded with `map_location`. It
    stores the moved tensor back in the group. Values must be finite and >= 0.
    `beta1` and `beta2` must be in [0, 1) and `eps` >= 0.
* **Shared step (`shared_step=True`, the default).** One Python int lives at optimizer
  level in `optimizer.state['__shared__'] = {'step': t}`.
  * `Optimizer.state_dict()` and `load_state_dict()` keep non-param keys as they are,
    so the step survives a checkpoint.
  * Member removal does not touch it, because it is an int and not a `(K,)` tensor.
  * The dict is replaced on every step and never changed in place. A `state_dict()`
    taken earlier therefore stays a true snapshot.
  * The step is incremented once per `step()` call in which at least one param has a
    gradient, before any update (so the first update uses t=1).
  * Per-param states then hold only `exp_avg` and `exp_avg_sq`.
* **Per-param step (`shared_step=False`).** `state[p]['step']` is an int64 `(K,)` tensor
  on `p.device`. It is incremented only when `p.grad` exists, which is PyTorch's
  semantics, and a17's `optimizer_select_` slices it with the moments.
* **Missing gradients.** A param whose grad is None gets no update, no weight decay and
  no state.
* **Numerics.** With a float lr and weight decay and an int step, `adamw_update_` runs
  the same kernels and Python-double scalars as `torch.optim.AdamW`
  (`lerp_`, `mul_().addcmul_()`, `addcdiv_(value=-step_size)`). With `(K,)` values,
  the per-member scalars (the `1 - lr*wd` factor, bias corrections and step size) are
  computed in float64. They are then cast to `p.dtype` and viewed as
  `(K, 1, ..., 1)`. MPS falls back to float32. Weight decay is skipped only when it
  is statically a no-op, meaning lr or wd is the float 0.
* **`adamw_update_`** accepts `step` as an int, a 0-d tensor or a `(K,)` tensor, and
  does not increment it. It validates the lr, wd and step shapes and `step >= 1`
  before it mutates anything.
* **Pickling.** `__getstate__` also saves `_pack_size` and `_shared_step`. The torch
  base class would otherwise drop them in `copy.deepcopy`.

### Differences from the official `AdamWPack` (`.reference/tabpack/src/project/optim.py`)

| Aspect | Official | Ours |
| :-- | :-- | :-- |
| `shared_step=True` | a Python int per param, incremented when that param has a grad | one int per optimizer (`state['__shared__']`), incremented once per `step()` that has any grad |
| `shared_step` default | `False` (the TabPack experiments pass `True` for MuonAdamWPack) | `True` (the contract) |
| per-member `beta1`/`beta2`/`eps` | lists allowed | floats only (the contract) |
| `follow_pytorch` | a flag; `False` switches to `lerp` for v and a tensor `div_` | always the PyTorch formulas (the same as `follow_pytorch=True`) |
| bias correction with a `(K,)` step | int64 tensor to float32 `pow` | float64 (closer to PyTorch's Python doubles) |
| group values | lists become tensors; floats stay floats | the same, plus a copy per group, a float32 dtype, validation and a lazy device move |
| param type | asserts `ParameterPack` | any tensor with the pack dim first (validated against `pack_size`) |

The two shared-step designs differ only for a param whose grad is None on some steps
but not others. Its bias correction then uses the optimizer step instead of its own
count. In TabPack, a param either always gets a gradient or never does (an unused
backbone block), so the two agree.

I also compared the implementations informally with a scratch script (not committed;
a40 owns parity tests). The setup was 50 random-gradient steps, K=3, mixed 2D and 3D
params, a zero-wd bias group, and lr/wd as lists or floats:

| Configuration | Max difference vs official |
| :-- | :-- |
| `shared_step=True`, float lr/wd | 0 (bit-exact) |
| `shared_step=True`, list lr/wd | 3e-8 |
| `shared_step=False` | up to 1e-6 (official computes its bias corrections in float32) |

## Tests

```
tools/dev/py -m pytest tests/optim/test_adamw_pack.py -q
36 passed, 2 skipped (CUDA hidden) in ~4 s
TABPACK_GPU=1 tools/dev/py -m pytest tests/optim/test_adamw_pack.py -q -k cuda
2 passed
tools/dev/py -m ruff check / ruff format --check (owned files): clean
```

The tests cover:

* **Equivalence with `torch.optim.AdamW`.** The reference is K independent
  `torch.optim.AdamW(foreach=False)` optimizers, run for 10 steps. lr and wd are given
  as a list, a tuple or a float64 tensor, with default and custom betas and eps and
  both step modes. The params mix (K,4,5), (K,5), (K,5,2) and (K,2) shapes, and the
  biases sit in a zero-wd group. Params match within atol 1e-6 (rtol 0), and
  `exp_avg`/`exp_avg_sq` also match.
* **Bit-exactness** with float hyperparameters.
* **Hyperparameter storage.** The group values are normalized: dtype, shape, device,
  floats kept as floats, and no aliasing. With zero grads, the zero-wd group override
  leaves the biases unchanged and scales the weights by exactly `1 - lr*wd`.
* **Validation errors.**
* **Step counters.** The shared step counts only `step()` calls that have gradients.
  The per-param step is an int64 `(K,)` tensor with correct counts when some grads
  are None. Shared and per-param steps agree when all grads exist.
* **Member removal after step 4.** A local stand-in slices the params, the state and
  the group tensors. The remaining members stay equal to their torch references.
* **Missing gradients.** When grads are None, the result matches torch AdamW, which
  also skips those params. A param without a grad is left untouched and gets no
  state.
* **Devices.** Hyperparameters move to the params' device eagerly and lazily (tested
  with `meta` on CPU). On CUDA, the tensors sit on the right device and the results
  match CPU.
* **Checkpointing.** A closure's loss is returned. The `state_dict` round trip goes
  through `torch.save`/`torch.load(weights_only=True)` and continues bit-identically.
  A snapshot of the shared step is not mutated. A shared/per-param mismatch on load
  raises a clear error. `deepcopy` keeps the configuration.
* **`adamw_update_` alone.** A `(K,)` step tensor [1, 4, 9] matches the float64
  formula. An int step, a constant `(K,)` step and a 0-d step all agree. Bad inputs
  are rejected.

## Coordination

* Merged `checkpoint/00b-data-fix` (fast-forward), as the integrator asked in #60.
* Posted `status` #19.
* Answered #25 from a16 (MuonAdamWPack state layout; the reusable `_PackOptimizer`
  helpers) with #84.
* Answered #26 from a17 (the state and group layout for `optimizer_select_`) with #85.
* Merged no peer branches, because I have no dependencies.

## Open issues

* **Shared and per-param states are not interchangeable.** A state saved with
  `shared_step=True` and loaded into an optimizer with `shared_step=False` raises a
  `RuntimeError` on `step()`. The reverse silently restarts the shared step at 1.
  Keep `shared_step` the same when saving and loading.
* **Load does not move tensors itself.** `load_state_dict` keeps per-param `step`
  tensors and group `(K,)` tensors on their saved devices, as torch does. `step()`
  moves them lazily. `exp_avg` and `exp_avg_sq` are moved by torch.
* **`add_param_group` after member removal.** A group without an explicit lr or wd
  takes the defaults. If a `(K,)` default was not sliced (a17 says it slices
  `optimizer.defaults`), the size mismatch raises a `ValueError`.
* **Unsupported features.** Sparse gradients raise an error, and there is no
  amsgrad, maximize or foreach/fused path. The contract does not require any of them.
