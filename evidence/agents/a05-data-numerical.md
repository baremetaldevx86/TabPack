# a05-data-numerical

## Summary

I implemented the numerical feature preprocessing that the official
`lib/data.py::transform_num` does:

* `noisy_quantile_transform(x_num, *, seed, noise_std=1e-5)` handles policy `noisy-quantile`, the one Churn uses.
* `standard_transform(x_num)` handles policy `standard`.
* `drop_constant_columns(x_num)` is the shared post-step that removes columns constant on train.

On every input I checked, the outputs are bit-identical to the official function,
including the real Churn split after binary-column extraction.

## Files

* `src/tabpack_repro/data/numerical.py`: the implementation. I force-added it, see *Open issues*.
* `tests/data/test_numerical.py`: 30 tests (force-added).
* `evidence/agents/a05-data-numerical.md`: this report.

## Design decisions

* **Recipe order** (the same as the official code):
  1. Fit the normalizer on train. For the quantile transform, fit on `train + RandomState(seed).normal(0, noise_std, train.shape).astype(train.dtype)`.
  2. Transform the *clean* arrays of every part.
  3. Apply `np.nan_to_num`.
  4. Drop the columns with one unique value on the transformed train part.
  5. Cast with `astype(float32)`.
* **Quantile settings:** `n_quantiles = max(min(n_train // 30, 1000), 10)` (the private
  helper `_n_quantiles`), `output_distribution='normal'`,
  `subsample=1_000_000_000` and `random_state=seed`.
* **NaN and inf:** I use plain `np.nan_to_num`, the same call as the official code.
  NaN becomes 0 and ±inf becomes the largest finite value.
* **Keys:** the output keeps the input dict's keys and their order. Any subset of parts
  works as long as it includes `'train'`.
* **No mutation:** every step builds new arrays (`+`, `transform`, `nan_to_num`,
  boolean indexing, `astype`), so the inputs are never modified.
* **`drop_constant_columns`** only removes columns. It keeps the dtype and does not
  replace NaNs. Its mask has an explicit `dtype=bool`, so zero-column inputs also work.
  The official code would raise an IndexError on those.
* **Private helpers:** `_fit_transform_parts` and `_finalize` are shared by both
  policies. The public signatures are unchanged from the skeleton contract.

## Tests

`tools/dev/py -m pytest tests/data/test_numerical.py -q` gives **30 passed** (about 5 s,
mostly the torch import in conftest). `ruff check` and `ruff format --check` pass on
both files.

The tests cover:

* Exact equality with the documented sklearn/numpy recipe:
  * float32 and float64 inputs;
  * `n_train` of 120 (clipped to 10 quantiles) and 3000 (100 quantiles);
  * seeds 0 and 3;
  * a custom `noise_std`.
* The `n_quantiles` formula.
* Determinism for a fixed seed, and a different seed giving a different but close fit.
* Finite float32 outputs for float32, float64 and int64 inputs.
* Standard-normal train marginals (mean, std and the 2.5/50/97.5% quantiles) and ranks
  preserved on train.
* Removal of train-constant and all-NaN-on-train columns applied to every part.
* NaN becoming 0 in every part.
* Identical clean train rows mapping to identical outputs, which shows that the clean
  train (not the noisy copy) is transformed.
* No input mutation.
* `drop_constant_columns` edge cases: train-only mask, dtype kept, copies, all columns
  constant, zero columns.
* `standard_transform` equality and statistics.

I also ran two throwaway scripts in the session scratchpad (not committed, since
parity tests belong to a42) that import the official `transform_num`:

* On synthetic data with `n_train` in {50, 700, 5000, 40000}, float32 and float64,
  seeds {0, 5}, plus the `standard` policy: all outputs were bit-equal.
* On real Churn (`x_num` split with `splits/default`, then the official
  `_extract_bin_from_num`, 6400×7 float32 train, seed 0): bit-equal.

## Coordination

* I read `docs/AGENTS_PROTOCOL.md`, the roster and the board before starting and
  after committing. No messages were addressed to me.
* I posted #9 (`status`): started, hit the same `.gitignore` blocker as #6/#7.
* I merged no peer branches, because I have no dependencies.
* I posted `done` with `--branch feat/a05-data-numerical`.

## Open issues

* **The `.gitignore` rule `data/`** (board #6, #7, #20) also matches
  `src/tabpack_repro/data/` and `tests/data/`. I force-added only my owned files.
  To run the tests locally I copied the skeleton `src/tabpack_repro/data/*.py` from the
  main checkout into my worktree. Those copies are ignored and not committed. The
  integrator's skeleton fix should un-ignore that path (for example with `/data/`) and
  commit `__init__.py` and the other contract files.
* A plain `ruff check src tests` skips `src/tabpack_repro/data/**` and
  `tests/data/**`. I confirmed this with `--show-files`, which lists no `/data/` file.
  Both `extend-exclude = ["data", ...]` in `pyproject.toml` and ruff's
  respect-gitignore cause it. Passing the files explicitly checks them, and mine pass.
  The integrator probably wants the fix to anchor both rules to the repo root.
