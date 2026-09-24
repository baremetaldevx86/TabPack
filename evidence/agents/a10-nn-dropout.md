# a10-nn-dropout: DropoutPack and make_activation

## Summary

I implemented `DropoutPack`, which applies dropout with a separate rate for each pack
member, and `make_activation`, following the frozen contract in
`src/tabpack_repro/nn/dropout_pack.py`. With the same seed, the float32 output is
bit-for-bit the same as the official `project.nn.DropoutPack` when both are given a
list of rates. This holds with and without `member_idx` (checked with a scratch
script, not committed). Unlike the official code, this version keeps bf16 inputs in
bf16.

## Files

* `src/tabpack_repro/nn/dropout_pack.py`: implementation.
* `tests/nn/test_dropout_pack.py`: 55 tests.
* `evidence/agents/a10-nn-dropout.md`: this report.

## Design decisions

* **Buffer `p`**: a persistent float32 `(K,)` buffer built with
  `torch.as_tensor(p, dtype=float32)`. A float is broadcast to all K members. A
  sequence or tensor must have length K. The constructor copies the value, so later
  changes to the caller's tensor do not affect it. The check `0 <= p < 1` runs
  after the float32 cast, because a value such as `1 - 1e-9` rounds to `1.0` and
  would make the scale infinite. NaN fails the check. The official code allows
  `p = 1`, which divides by zero. The constructor raises `ValueError` when
  `pack_size < 1`.
* **`pack_size`** is read from `p.shape[0]` on every access and never cached. When
  `pack_ops.pack_select_` replaces the buffer with a slice, the value follows
  automatically.
* **Forward pass**: `p = self.p[member_idx]`, or `self.p` when `member_idx` is
  None. It raises `ValueError` when `x.ndim == 0` or `x.shape[0] != len(p)`. This
  check uses only Python shapes, so it never forces a host sync.
  * In eval mode it returns `x` itself, the same object and not a copy.
  * In train mode it computes
    `keep = 1 - p`, reshaped to broadcast over the trailing dimensions, then
    `scale = torch.bernoulli(keep.expand(x.shape)).div_(keep)` and finally
    `(x * scale).to(x.dtype)`. For the fp32 parity above, this draws random numbers
    in the same pattern as the official code.
  * The mask and scale use `promote_types(x.dtype, float32)`, so the precision is at
    least float32. For bf16 or fp16 inputs, the `1/(1-p)` scale is applied in
    float32 and the result is rounded once. Rounding the scale itself to bf16 would
    bias the mean by up to about 0.2 %. Autocast does not cover `bernoulli`, `div`
    or `mul`, so the final `.to(x.dtype)` keeps the output dtype equal to the input
    dtype.
  * **Identity for p = 0**: `bernoulli(1.0)` is always 1 and `1/1` is exactly 1, so
    `x * 1.0` returns `x` bit-for-bit, including `-0.0` and `±inf`. The same holds
    for the gradient. There is no data-dependent branch, so there is no GPU sync. The
    random generator is still used for every member, which gives the same draw
    pattern whatever the values of `p`.
  * There is no shortcut that skips all work when every `p` is 0. It would need a
    host sync, or a cached flag that could go stale after in-place edits to the
    buffer.
* **`make_activation`** looks up the name in a dictionary (`ReLU`, `GELU`,
  `SiLU`). The lookup is case-sensitive, like the official `getattr(nn, type)`. Each
  call returns a new module. An unknown name or a non-string raises `ValueError`
  and lists the allowed names.

## Tests

`tools/dev/py -m pytest tests/nn/test_dropout_pack.py -q` gave `55 passed`, and
`ruff check` and `ruff format --check` pass on both owned source files. The tests
cover these areas:

* **Constructor and buffer**: dtype and shape of the buffer for float, int, list,
  tuple and float64-tensor inputs; the buffer is a copy of the input; `state_dict`
  holds only `p`, with shape `(K,)` and dtype float32, and survives a
  `load_state_dict` round trip; `pack_size` is correct after the buffer is sliced in
  place.
* **Invalid input**: `ValueError` for invalid `p`, a wrong length or a bad
  `pack_size`.
* **Eval mode**: the module returns the same object, with and without `member_idx`.
* **Exact identity**: in train mode, p = 0 gives bit-exact outputs in fp32, fp64 and
  bf16. A pack that mixes zero and non-zero rates is also exact for its zero
  members. The gradient of a p = 0 member is exact, and the gradient of a p = 0.5
  member follows the mask and the ×2 scale.
* **Statistics**: for each member with p in {0, .1, .3, .6}, the keep rate and the
  mean are measured on 131k elements. The tolerances are 0.01 and 0.03, which is at
  least 7 standard errors, and a guard test enforces this. Kept values equal
  `x / (1 - p)`. Masks are independent across members and across calls, and the same
  seed gives the same output. Inputs with any number of trailing dimensions and 1-D
  inputs work.
* **`member_idx`**: selecting only the p = 0 members gives the exact identity.
  Reordered and duplicated indices follow `p[member_idx]`. `member_idx = arange(K)`
  equals the full pack, and a subset equals a standalone sub-pack under the same
  seed. Shape mismatches raise errors.
* **Dtypes**: bf16, fp16 and fp64 are preserved in train and eval mode. The bf16
  statistics hold, with the float32 scale rounded once. Under
  `torch.autocast('cpu', bfloat16)`, a `Linear` output stays bf16. After
  `.to(float64)`, the module still works.
* **`make_activation`**: the three names give the right type and a new module each
  time. Unknown names raise `ValueError`.

The first `backward()` in a process takes about 4 to 12 s on this machine, even for
a plain `x * c`. This happens in any process under the current load, so it is not
caused by `DropoutPack`. The rest of the file runs in under 1 s.

## Coordination

* Posted `status` (#13) and `done` (#58) on the board.
* Sent #59 to a39-parity-nn with the fp32 bitwise parity result, how the official
  scalar-p path differs (it uses `F.dropout`, which draws random numbers
  differently), and the bf16 dtype difference.
* Received no questions addressed to a10. Merged no peer branches, since a10 has no
  dependencies.
* To run the tests, I copied the gitignored skeleton package `src/tabpack_repro/data`
  from the main checkout into the worktree, untracked. `tests/conftest.py` imports it
  (see the `.gitignore` `data/` blocker reported on the board in #6, #7 and #9). I did
  not commit it.

## Open issues

* **Contract note**: when every `p` is 0, the train-mode path still draws random
  numbers. The official code, given a scalar `p = 0`, calls `F.dropout`, which skips
  the draw. As a result, the global torch random state after a forward pass can
  differ from the official code in the homogeneous `dropout = 0` case. Outputs are
  identical either way. This matters only for parity tests that compare random
  streams downstream.
* `member_idx` may also be a boolean mask, since plain tensor indexing supports it,
  but no test covers that. The tested form is int64 indices.
