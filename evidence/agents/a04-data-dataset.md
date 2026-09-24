# a04-data-dataset report

## Summary

Implemented `load_raw_dataset(path, split='default') -> RawDataset` in
`src/tabpack_repro/data/dataset.py`. It reads a dataset directory in the official
TabPack format, validates everything it reads, applies the split and infers
`n_classes`. The `TaskInfo`/`RawDataset` contract is unchanged (only private helpers
were added). On the real Churn data the arrays are identical, with the same dtypes, to
the official `lib.data.Dataset.from_dir(churn, ['default'])`. I checked this with an
ad-hoc script. The parity test belongs to a42.

## Files

- `src/tabpack_repro/data/dataset.py`: implementation (force-added, see Open issues)
- `tests/data/test_dataset.py`: 65 tests (force-added)
- `evidence/agents/a04-data-dataset.md`: this report

## Design decisions

- **Files.** `info.json` and `y.npy` are required. `x_num.npy`, `x_bin.npy` and
  `x_cat.npy` are optional, and a missing one gives `None`. Other `.npy` files in the
  directory are ignored. The official loader loads every `.npy` file, but only these
  four keys exist in `RawDataset`. Files are read with `allow_pickle=False`.
- **Dtypes are preserved, and only the dtype kind is checked:** `x_num` must be
  floating. `x_bin` may be floating, bool or integer. `x_cat` must be str or integer
  (the official loader also allows both). `y` must be integer for classification and
  floating for regression. Features must be 2D with the same number of rows as `y`,
  and `y` must be 1D. The official loader asserts exact float32/int64/int32 dtypes;
  checking only the kind is looser but still catches the real errors.
- **Splits.** `split` is a directory name under `splits/`. A nested id such as
  `'a/0'` collects parts from every level, like the official `load_split`. A part
  defined twice raises an error, and `._*` AppleDouble files are skipped. The split
  must have exactly the parts `train`, `val` and `test`, because `PartKey` is that
  literal. Index arrays may use any integer dtype (the official format uses int32).
  Each part must be 1D, non-empty, within `[0, n_rows)` and free of duplicates, and
  the parts must be pairwise disjoint. Otherwise the loader raises `ValueError` with
  the names of the parts involved.
- **`n_classes`** is the number of unique train labels, as the contract says. The
  official `Task.compute_n_classes` counts unique labels over all parts. The two agree
  on valid data because the loader also requires:
  - the train labels to be exactly `0..C-1` (they are used as class indices);
  - val/test to contain no class that is missing from train;
  - `C == 2` for binclass and `C >= 2` for multiclass.
- **`score`** is passed through as a string (for example `'accuracy'`). Only a
  non-empty string is required, because the metric names belong to a08.
- **Errors.** A missing directory, `info.json` or `y.npy` raises `FileNotFoundError`
  with the hint ``run `tabpack-repro download --name <dir name> --data-dir <parent>`
  to fetch it``. A missing split raises `FileNotFoundError` that lists the available
  splits. Malformed content raises `ValueError`.
- Integer-array indexing gives each part an independent copy.

## Tests

`tools/dev/py -m pytest tests/data/test_dataset.py -q` gives **65 passed** in about
3 s, including the 2 `@pytest.mark.data` Churn tests, which run (they are not
skipped). `ruff check` and `ruff format --check` pass on both files.

The tests cover:

- all 8 combinations of present/absent `x_num`/`x_bin`/`x_cat` for regression,
  binclass and multiclass (values equal to `array[idx]`, dtypes kept);
- str and int `x_cat`, independent copies, str paths, named splits, nested splits and
  AppleDouble files;
- missing directory, `info.json`, `y.npy` or split directory, and invalid split names;
- bad splits: overlaps, out-of-range or negative indices, duplicates, empty, missing
  or extra parts, float indices, and a part defined twice;
- bad labels: 3 or 1 classes for binclass, labels that are not `0..C-1`, a label
  unseen in train, float labels for classification and int labels for regression;
- bad `info.json` and bad feature shapes or dtypes;
- Churn: parts 6400/1600/2000; `x_num` `(.,7)` float32, `x_bin` `(.,3)` with values
  in {0,1}, `x_cat` `(.,1)` str with values {France, Germany, Spain}; y int64;
  binclass, `accuracy`, 2 classes; and equality with the raw files.

## Coordination

- Posted #4 (`status`: started).
- Posted #7 (`blocker` to the integrator) about the `.gitignore` `data/` rule. a03,
  a05, a06 and a07 reported the same problem (#6, #9, #12, #14).
- No questions were addressed to me, and I merged no peer branches (a04 has no
  dependencies).
- To run the tests locally I copied the skeleton `data/*.py` files from the main
  checkout into my worktree. The copies are untracked and ignored, and I did not
  commit them.

## Open issues

- **Blocker for the integrator: the skeleton is missing the data package.** The
  `.gitignore` rule `data/` also ignores `src/tabpack_repro/data/` and `tests/data/`,
  so `checkpoint/00-skeleton` has no `tabpack_repro.data` package. I committed my two
  files with `git add -f`.
  - The integrator still has to commit `src/tabpack_repro/data/__init__.py` and the
    other data contracts.
  - The suggested fix is to change the rule to `/data/`.
  - The `[tool.ruff] extend-exclude = ["data"]` setting also excludes these
    directories from `ruff check src tests`. Explicit file paths are still linted.
- A feature file with 0 columns is returned as a `(n, 0)` array rather than `None`,
  the same as the official loader.
