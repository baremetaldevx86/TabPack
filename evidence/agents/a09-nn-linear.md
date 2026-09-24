# a09-nn-linear: LinearPack

## Summary

I implemented `LinearPack`, which holds K independent `nn.Linear(in, out)` layers and
runs them with one batched matmul. The weight has shape `(K, in, out)` and the bias
`(K, out)`, so the forward pass is `torch.baddbmm(bias[:, None], x, weight)`, or
`torch.bmm(x, weight)` when there is no bias. An optional `member_idx` gathers
`weight[member_idx]` and `bias[member_idx]`. `pack_size` is read from
`weight.shape[0]` on every access, so in-place member removal works. Each member is
initialized like `nn.Linear`. The values match `nn.Linear` bit for bit when K=1, and
match the official `LinearPack` for any K, given the same CPU RNG state.

## Files

* `src/tabpack_repro/nn/linear_pack.py`: the implementation, with `extra_repr`.
* `tests/nn/test_linear_pack.py`: 27 tests.
* `evidence/agents/a09-nn-linear.md`: this report.

## Design decisions

* **Layout.** The contract fixes `(K, in, out)`, which is the transpose of both
  `nn.Linear` and the official `LinearPack`. With this layout the forward pass is a
  plain `x @ W`, with no transposed view.
* **Initialization.** `nn.Linear` draws its weight with `kaiming_uniform_(a=sqrt(5))`.
  That is the same as U(-b, b) with b = sqrt(1/3) * sqrt(3/in) = 1/sqrt(in), and the
  bias uses the same bound. `reset_parameters` draws one `(K, out, in)` tensor, copies
  its transpose into `weight`, and then draws the bias. This keeps the draw order and
  memory order of `nn.Linear` (K=1) and of the official pack (weight first, then bias,
  bound `in**-0.5`). As a result, a seeded run produces the same numbers, which makes
  parity checks easy. The drawn values are i.i.d., so each member follows the
  `nn.Linear` distribution independently. `reset_parameters` runs under `no_grad` and
  modifies the parameters in place, so parameter identity is preserved.
* **Forward.** A `(K', B, in)` input with the wrong rank, pack size or feature count
  raises `ValueError`. Inputs expanded with stride 0 work, for example a `(B, in)`
  batch broadcast to all members. Autocast needs no special handling: `bmm` and
  `baddbmm` are on the autocast lower-precision list, so a bf16 context gives bf16
  outputs while the parameters and their gradients stay fp32.
* **Attributes.** `in_features` and `out_features` are plain ints, as in `nn.Linear`.
  `bias=False` registers `bias` as `None`, so the state dict contains only `weight`.
* **Differences from the official `project.nn.LinearPack`** (for a39):
  * `ours.weight == official.weight.transpose(-2, -1)`. The bias is the same, and the
    official `pack_idx` argument is our `member_idx`.
  * We do not support per-member `in_features`/`out_features` lists,
    `max_in_features`/`max_out_features` output masking, the debugging `loop=` mode,
    or `dtype`/`device` constructor kwargs. The contract does not require any of them:
    TabPack on Churn uses one `d_block` for every member.
  * The official pack stores its tensors as `ParameterPack` subclasses. Ours are
    plain `nn.Parameter`s, and the pack invariant (dim 0 == K) holds for them.
  * I checked locally, with a scratch script that is not committed: for the same seed,
    the initial weights and biases are bit-identical to the official ones after the
    transpose, and so are the forward outputs, with and without `member_idx`.
* **Muon orientation.** The official spectral scale is `max(1, out/in)**0.5`,
  computed from `size(-2)/size(-1)` in its `(K, out, in)` layout. In our layout the
  same value is `W.shape[2]/W.shape[1]`. I sent this to a16 (board #34).

## Tests

```
tools/dev/py -m pytest tests/nn/test_linear_pack.py -q
27 passed in 0.38s
tools/dev/py -m ruff check / ruff format --check (owned files): clean
```

The tests cover:

* Forward output and gradients equal K independent `nn.Linear` layers with copied
  weights (fp32, atol 1e-6 for outputs and 1e-5 for gradients), with and without bias.
* A `member_idx` subset (distinct, duplicate and full index sets) equals a pack built
  from the sliced parameters, exactly, and equals the sliced output of the full pack.
* Unselected members get exactly zero gradient, and selected members get non-zero
  gradients.
* Init statistics: mean close to 0, std close to b/sqrt(3), max |w| <= b = 1/sqrt(in),
  and members differ from each other. For K=1 the values equal `nn.Linear` under the
  same seed, and `reset_parameters` redraws in place.
* `bias=False`, K=1, and pack size derived after in-place slicing.
* Expanded input, bf16 autocast (output dtype, closeness, fp32 gradients) and float64.
* `extra_repr` and argument validation.

## Coordination

* Read the board before starting. The `.gitignore` `data/` rule is reported in
  #6/#7/#12/#20: the skeleton omitted `src/tabpack_repro/data`, so `tests/conftest.py`
  cannot be imported. As a workaround, I copied the skeleton `data` package from the
  main checkout into my worktree. The copy is gitignored and never committed. I did
  not modify any file outside my owned paths.
* Posted: #10 (started), #32 (done, `feat/a09-nn-linear`), #33 to a39 (layout mapping
  and RNG-exact parity), #34 to a16 (weight orientation for the Muon scale).
* Merged no peer branches. LinearPack has no dependencies.

## Open issues

* None in LinearPack. The data-package `.gitignore` blocker still needs an integrator
  fix before anyone can run pytest in a clean worktree.
