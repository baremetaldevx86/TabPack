# a14-optim-ns: batched Newton-Schulz orthogonalization

## Summary

Implemented `zeropower_via_newtonschulz5(G, steps=5, eps=1e-7)` for Muon. It works
batched over any leading dims (a `(K, m, n)` parameter pack or deeper) and computes in
bfloat16. It transposes when rows > cols and normalizes each matrix by its own
Frobenius norm (last two dims, keepdim, +eps). It then runs `steps` quintic iterations
with coefficients `(3.4445, -4.7750, 2.0315)`, transposes back and returns bf16 with the
input's shape. The output is **bit-identical** (`torch.equal`) to the official
`vendor/muon.py` on CPU and on CUDA for wide, tall, square and 4D inputs. I checked this
with a scratch script, not a committed test, because parity tests belong to a40.

## Files

* `src/tabpack_repro/optim/newton_schulz.py`: the implementation. The contract
  (name, signature, `NS_COEFFICIENTS`) is unchanged.
* `tests/optim/test_newton_schulz.py`: 34 tests (33 CPU, 1 `gpu`).
* `evidence/agents/a14-optim-ns.md`: this report.

## Design decisions

* **Per-matrix normalization.** `torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)`
  on the bf16 tensor. This is the same kernel `Tensor.norm(dim=(-2,-1))` dispatches to,
  so the rounding matches the reference.
* **Arithmetic order.** The reference writes `b * A + c * A @ A`. Python parses that as
  `(c * A) @ A`, and in bf16 it differs from `c * (A @ A)` (max abs difference of 32 on a
  16x32 example). I compute `(c * A) @ A` explicitly to keep bit parity. `a * X + B @ X`
  also stays unfused (no `baddbmm`) for the same reason.
* **Orientation.** The rows > cols test uses `G`'s last two dims. Iterating on the wide
  orientation keeps the Gram matrix `min(m,n) x min(m,n)`, so
  `f(G.mT) == f(G).mT` holds exactly.
* **Validation.** `ValueError` for `G.ndim < 2` or `steps < 0`. The reference asserts on
  the first case and does not check the second. The function does not modify its input,
  and it does not wrap itself in `no_grad`, because the optimizer's `step` already does.
* **Zero matrices.** `0 / (0 + eps)` gives an exact zero with no NaN, and a zero member
  does not affect the other members.

## Tests

`tools/dev/py -m pytest tests/optim/test_newton_schulz.py -q` gives **33 passed,
1 skipped** (the CUDA test skips because the runner hides CUDA).
`TABPACK_GPU=1 tools/dev/py -m pytest tests/optim/test_newton_schulz.py -q -m gpu` gives
**1 passed**. `ruff check` and `ruff format --check` pass on both files.

Coverage:

* A batched result equals the per-matrix results for 3D and 4D inputs, with member
  scales spanning 1e-3 to 1e3. The comparison uses bf16 tolerance; on this machine the
  results are bit-identical.
* Scaling members by different powers of two gives bit-identical output. A batch-wide
  norm fails this test (max difference 0.55).
* Singular values lie in [0.5, 1.5] for rectangular and small square Gaussian matrices,
  including 1xn and nx1, and also with 10 steps.
* The output has cosine > 0.95 with the polar factor `U V^T`.
* Wide and tall inputs, `f(G.mT) == f(G).mT`, and a near-orthonormal Gram matrix.
* K=1 matches the 2D call.
* Zero matrices, alone and as one member of a batch.
* The output is bf16 for fp32, fp64, fp16 and bf16 inputs, and the input is left
  unchanged.
* `steps=0` returns the Frobenius-normalized input.
* Invalid arguments raise `ValueError`.
* CUDA: results on the device, close to the CPU result, batched equal to per-matrix, and
  singular values within [0.5, 1.5].

## Coordination

* Read the board before starting and before finishing. Nothing was addressed to a14.
* #47 to a40-parity-optim: the reference normalizes per matrix for batched input, so it
  matches the contract and there is no finding. The message also covered bit-exact
  parity, the `(c*A)@A` order, and the reference signature `(G, steps)`, which has
  `eps=1e-7` hard-coded.
* #48 to a16-optim-muon: the API, shape, dtype and error behavior. The caller casts the
  result back and applies the `max(1, rows/cols)**0.5` scale.
* No peer branches merged. To run pytest locally I copied the skeleton
  `src/tabpack_repro/data` package from the main checkout into the worktree. It is
  git-ignored, untracked and not committed (see blocker #6/#7/#12).

## Open issues

* Large square Gaussian matrices (for example 256x256) have near-zero singular values
  that 5 steps cannot lift into [0.5, 1.5]. The minimum was about 0.04. This is inherent
  to Muon's iteration and identical in the reference, so the range test only uses
  rectangular or small square shapes.
* Worktree test runs depend on the `.gitignore` `data/` fix (blocker #6), because
  `tests/conftest.py` cannot import `tabpack_repro.data` otherwise.
