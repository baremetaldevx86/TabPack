# a06-data-categorical

## Summary

Implemented the binary and categorical preprocessing steps of the official Churn pipeline
in `src/tabpack_repro/data/categorical.py`:

* `extract_bin_from_num` moves binary numeric columns out of x_num.
* `merge_bin` puts the extracted columns in front of the original x_bin.
* `bin_to_cat` is the official 'convert-to-cat' step.
* `ordinal_encode` is the official 'ordinal' step plus `compute_cat_cardinalities`.

Outputs match the official code exactly. That includes the dtypes, and it was checked
on the real Churn data and on random inputs.

## Files

* `src/tabpack_repro/data/categorical.py`: implementation. The contract text comes
  from the main checkout's skeleton, with more detail added to the docstrings.
* `tests/data/test_categorical.py`: 24 tests.
* `evidence/agents/a06-data-categorical.md`: this report.

## Design decisions

* **Binary extraction.** A column is binary when it has exactly two unique values
  over all parts together and no NaN. Its encoding is `x == hi`, where `hi` is the
  larger of the two sorted values. This matches the official result:
  `OrdinalEncoder(categories=np.unique over all parts)` followed by `.astype(bool)`.
  Column order is kept, and the remaining columns keep their dtype.
  * When nothing is extracted, `remaining` is a copy of x_num, not the same object.
* **Merge.** `np.concatenate([extracted, x_bin], axis=-1)`. A bool column combined
  with float32 becomes float32 0.0/1.0. When only one side exists, it is returned
  unchanged as a copy, so bool-only bins stay bool.
* **Bin to cat.** Uses a plain `astype(x_cat dtype)`, so numpy's casting rules apply
  exactly:

  | Input | Output |
  | :-- | :-- |
  | float32 `1.0` | `'1.0'` |
  | bool `True` | `'True'` |
  | `NaN` | `'nan'` |
  | any value, cast to `'<U1'` | truncated (e.g. `'1'`, `'T'`) |

  Bool and float bins therefore give different strings, but they sort the same way,
  so their codes are the same. The result is `[x_cat, bins]`. Without x_cat, NaN
  becomes 2 and the output is int64.
* **Ordinal encoding.** Uses sklearn `OrdinalEncoder(handle_unknown='use_encoded_value',
  unknown_value=-1, dtype=int64)` fitted on train.
  * The -1 sentinel is then replaced, in every non-train part, by that column's
    train max + 1. The official code uses the sentinel `int64 max - 3`, and the
    result is the same.
  * The cardinality of a column is `len(np.unique(train codes))`.
* **Input checks.** When the part keys differ, `ValueError` is raised (the official
  code uses asserts). Inputs are never mutated.
* **Churn result.** No numeric column is binary (Tenure has 11 values, NumOfProducts
  has 4), so nothing is extracted. x_cat ends up as `[Geography, 3 flags]` in
  `'<U7'`, with cardinalities `[3, 2, 2, 2]`.

## Tests

`tools/dev/py -m pytest tests/data/test_categorical.py -q`: **24 passed** (3.8 s).
This includes the `data`-marked test on the real Churn arrays, which loads them with
`np.load` and applies the split indices.

`ruff check` and `ruff format --check` on both files: clean.

I also ran two scratch parity scripts (not committed, because a42 owns the parity
tests). Both were bit-identical to the official code in values and dtypes:

* **Random inputs.** 262 random cases compared against the official
  `_extract_bin_from_num`, `convert_bin_features_to_cat_`, `transform_cat('ordinal')`
  and `compute_cat_cardinalities`.
* **Real Churn.** The official `build_dataset(churn, extract_bin_from_num=True,
  bin_policy='convert-to-cat', cat_policy='ordinal')` against my chain.

## Coordination

* Read the board before starting and after each commit.
* Posted #14 (status). It reported the same `.gitignore` blocker as #6/#7, plus the
  ruff note below.
* Posted #45 to a07-data-pipeline: how to call these functions in the official order.
* Posted #46 to a42-parity-data: hints for the parity tests.
* No peer branches merged.

## Open issues

* **Data package missing from git (for the integrator).** The `.gitignore` rule
  `data/` also matches `src/tabpack_repro/data/` and `tests/data/`, so the skeleton's
  data package is not in `checkpoint/00-skeleton`.
  * I force-added only my two owned files.
  * To test, I copied the other skeleton data files, including `__init__.py`, into
    my worktree. They are still untracked and ignored.
  * Suggested fix: change the rule to `/data/`.
* **Ruff skips the data package (for the integrator).** `pyproject.toml` has
  `[tool.ruff] extend-exclude = [..., "data", ...]`, which also excludes
  `src/tabpack_repro/data` and `tests/data` from `ruff check src tests`. Suggested
  fix: `/data` or `./data`.
