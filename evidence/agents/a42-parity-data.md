# a42-parity-data

## Summary

I wrote parity tests that compare our data pipeline (a04 to a07) with the official
`lib/data.py` (commit `05a89e2`). They cover the full `build_dataset` path and each
preprocessing step on its own. The inputs are the real Churn dataset and synthetic
edge cases. Every comparison is **bitwise**: same dtype, same shape and the same bytes,
so even `-0.0` vs `0.0` would count as a mismatch. All 129 cases match exactly. No
tolerance is needed anywhere.

## Files

* `tests/parity/test_parity_data.py`: the parity tests.
* `evidence/agents/a42-parity-data.md`: this report.

## Design decisions

* **The official module needs no environment setup.** `lib.data` imports cleanly
  once the official `src/` is on `sys.path` (the shared `official` fixture handles that).
  `lib.env` is only used for the cache, and the tests always pass `cache=False`. As a
  guard, the `od` fixture monkeypatches `lib.env.get_project_dir`, `get_cache_dir` and
  `get_data_dir` to raise. No test can therefore write a cache or project directory
  inside the clone, and the clone is never edited.
* **The official config is pinned.** `OFFICIAL_CHURN_KWARGS` (`noisy-quantile`,
  `extract_bin_from_num=True`, `convert-to-cat`, `ordinal`, `seed=0`, `cache=False`)
  is checked against the `data` section of the official
  `experiments/tabpack/churn/main/config.json`, plus the default `seed` of
  `build_dataset`. The same test checks that `DataConfig()` defaults equal these
  kwargs. `build_dataset(DataConfig(path=<churn>))` is therefore the paper's Churn
  preprocessing.
* **What the end-to-end comparison covers:** x_num, x_cat and y (values, dtype and
  column order); `cat_cardinalities` vs `Dataset.compute_cat_cardinalities()`;
  `n_num_features` and `n_cat_features`; the part sizes; the task type, score and
  `n_classes` (vs `try_compute_n_classes`). It also checks that no `x_bin` is left over
  and that our tensors are CPU and contiguous.
* **End-to-end cases:**
  * real Churn with the official config;
  * Churn with other settings: seeds 1 and 2026, `standard`, `None`, and
    `extract_bin_from_num=False`;
  * a Churn copy whose 3 binary features are interleaved into `x_num`. Both pipelines
    re-extract them to `bool`, which becomes the categories `'False'`/`'True'`. The
    result must equal the build from the original files;
  * 6 synthetic dataset directories in the official on-disk format (int32 split files,
    shuffled rows), each with 2 seeds:
    * NaN, constant, train-constant and all-NaN-train columns;
    * binary and all-binary numerical columns;
    * x_bin with NaN;
    * unknown str and int categories;
    * no x_cat, and a multiclass task;
    * a dataset where every numerical column is dropped.
  * The richest synthetic directory is also run under every num policy, with and without
    binary extraction.
* **Per-step cases:**
  * `noisy_quantile_transform` and `standard_transform` vs `transform_num`, with
    n_train = 7, 64, 450 and 30030. These cover fewer rows than quantiles, the minimum of
    10 quantiles and the 1000-quantile cap. Both float32 and float64 are tested, and the
    inputs must not be mutated.
  * `drop_constant_columns` vs `transform_num(x, None, None)`.
  * `extract_bin_from_num` vs `_extract_bin_from_num`: 50 random column mixes plus 4
    fixed edge cases. The mixes include:
    * two values, with and without NaN;
    * one value plus NaN;
    * three values;
    * a second value that occurs only outside train;
    * `-0.0`/`0.0`/`1.0`.
  * `bin_to_cat` vs `Dataset.convert_bin_features_to_cat_`. The bins are float, float
    with NaN, or bool, combined with no x_cat, `<U7`, `<U1` (truncation) or int64 x_cat.
    The result is then chained into `ordinal_encode` vs `transform_cat` and the
    cardinalities.
  * `ordinal_encode` on unknown categories in val, test or both.
  * The same steps on the real Churn arrays, loaded by the official loader.
  * `load_raw_dataset` vs `Dataset.from_dir` on Churn.
* **Known differences are stated, not hidden:**
  * When every numerical column is constant, the official code keeps an `(N, 0)`
    x_num, while our `x_num` is `None` (a07's documented decision). The helper asserts
    exactly this equivalence.
  * NaN bins cast to int64 x_cat are undefined behavior in numpy, so that one
    combination is skipped.

## Tests

```
tools/dev/py -m pytest tests/parity/test_parity_data.py -q
129 passed, 1 skipped in ~8s
```

`ruff check` and `ruff format --check` pass on the test file.

To check that the tests can actually fail, I ran them with mutations injected through a
scratch pytest plugin, without touching any `src` file:

| Mutation | Failed tests |
| :-- | :-- |
| `n_quantiles + 1` | 30 |
| float64 output | 20 |
| unknown categories mapped to 0 | 34 |
| NaN bin code 2 changed to 3 | 1 |

The NaN-code mutation fails only in the per-step test. That is correct: ordinal
encoding makes the end-to-end codes identical either way.

## Coordination

* Read a05's note (#44) and a06's note (#46). a06's hints are covered: bool bins cast to
  `'True'`/`'False'` and float bins to `'0.0'`/`'1.0'`, and unknown categories get
  code `train_max + 1`.
* Posted the start `status` (#114) and a low-severity `finding` to a06 (#143).
* Merges: `main`, which already contains a03 to a07. No peer branches were merged.

## Open issues

* **Finding #143 to a06 (FYI, cannot occur on Churn).** A two-valued numerical column
  that contains `inf` behaves differently. The official `_extract_bin_from_num` raises
  (the OrdinalEncoder rejects infinity), while ours extracts the column as binary. A
  non-binary column with `inf` fails in both implementations. The tests leave `inf` out.
* There are no parity gaps on Churn: the processed dataset is bit-identical to the
  official one.
