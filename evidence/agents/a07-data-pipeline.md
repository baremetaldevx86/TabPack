# a07-data-pipeline report

## Summary

I implemented `PreparedDataset.to(device)` and `build_dataset(config: DataConfig)` in
`src/tabpack_repro/data/pipeline.py`. `build_dataset` calls the functions written by
a03-a06 in the order of the official `lib/data.py::build_dataset`:

1. `load_raw_dataset`
2. `extract_bin_from_num` + `merge_bin`
3. the numerical transform
4. `bin_to_cat`
5. `ordinal_encode`

It returns CPU tensors: `x_num` is float32, and `x_cat` and `y` are int64. The public
signatures did not change. I only added private helpers and constants.

On real Churn, with the default `DataConfig()` and `seed=0`, the output is
**bit-identical** to the official
`lib.data.build_dataset(churn, extract_bin_from_num=True, num_policy='noisy-quantile', bin_policy='convert-to-cat', cat_policy='ordinal', seed=0)`.
All nine arrays (`x_num`, `x_cat` and `y` for each of the three parts) have the same
dtypes and are `np.array_equal`. Both give cardinalities `[3, 2, 2, 2]`. I checked this
with an ad-hoc script, because the committed parity test belongs to a42.

## Files

- `src/tabpack_repro/data/pipeline.py`: the implementation.
- `tests/data/test_pipeline.py`: 36 tests. The CUDA test is skipped without a GPU.
- `evidence/agents/a07-data-pipeline.md`: this report.

## Design decisions

- **Policy validation comes first.** Before reading any data, `build_dataset` checks
  each policy against the values it can run:
  - `num_policy` ∈ {None, `noisy-quantile`, `standard`};
  - `bin_policy` ∈ {None, `convert-to-cat`};
  - `cat_policy` ∈ {None, `ordinal`}.

  Any other value raises `ValueError` naming the field. The official code also has
  `cat_policy='one-hot'`, but `PreparedDataset.x_cat` holds ordinal codes by
  contract (the model does the one-hot encoding), so `one-hot` is rejected too.
- **Path.** `Path(config.path).expanduser()` is used as given when it is absolute.
  Otherwise it becomes `get_data_dir() / path`. The split is always `'default'`,
  because `DataConfig` has no split field.
- **Regression** raises `NotImplementedError` right after loading, as the contract
  requires.
- **`num_policy=None`** runs the official `transform_num` post-steps and nothing else:
  NaN -> 0, then `drop_constant_columns`, then a cast to float32.
- **When every numerical column disappears**, `x_num` becomes `None` instead of an
  `(N, 0)` tensor. This happens when bin extraction takes all of them or when all
  are constant. It matches the contract ("x_num ... or None") and keeps
  `n_num_features == 0` consistent. The official code keeps an empty array in this
  case. Churn is not affected.
- **Binary features left unconverted** (`bin_policy=None` while there are binary
  features) raise `ValueError`. `PreparedDataset` has no `x_bin`, and the official
  trainer asserts `n_bin_features == 0`.
- **`cat_policy=None`** is accepted only for integer-valued `x_cat`. An example is
  binary features converted without an original `x_cat`. The cardinalities are then
  the number of unique train values per column, as in the official
  `compute_cat_cardinalities`. String categories raise `ValueError`.
- **Tensors** are built with `torch.from_numpy(np.ascontiguousarray(a, dtype))`.
  Read-only arrays are copied first, so there is no torch warning and no shared
  read-only memory. Labels are always cast to int64. Cardinalities are plain `int`.
- **`.to(device)`** returns a new `PreparedDataset` built with
  `dataclasses.replace`. It has new dicts for every tensor kind, `None` kinds stay
  `None`, `cat_cardinalities` is a new list, and the frozen `task` is shared.
- **Imports.** The dependency functions are imported by name into `pipeline`, so the
  fake-based tests monkeypatch `pipeline.<name>` and leave the other modules alone.

## Tests

`tools/dev/py -m pytest tests/data/test_pipeline.py -q` gives **35 passed, 1 skipped**
(the skip is the CUDA test when no GPU is visible). The 2 `@pytest.mark.data` Churn
tests run: they are not skipped. With `TABPACK_GPU=1` the CUDA round-trip test also
passes. The whole `tests/data` directory, with a03-a06 merged, gives 198 passed and
1 skipped. `ruff check` and `ruff format --check` pass on both owned files.

- **Fake-based orchestration tests (a03-a06 monkeypatched):**
  - the exact call order and the data flow between steps (checked by object
    identity);
  - the seed reaching `noisy_quantile_transform`;
  - extracted binary features merged before the original `x_bin`;
  - extraction turned off, nothing extracted, or all numerical columns extracted;
  - any feature kind missing;
  - the `standard` and `None` numerical policies;
  - unknown policies and `one-hot` rejected before loading, and unconverted binary
    features rejected;
  - `cat_policy=None` with integer codes and with strings;
  - regression raising `NotImplementedError`;
  - relative and absolute path resolution;
  - output dtypes, device and contiguity;
  - read-only inputs.
- **`.to` tests:** `cpu`, `torch.device('cpu')` and `meta` (a real device move
  without a GPU), value-preserving round trips, `None` kinds kept, and a CUDA round
  trip (`@pytest.mark.gpu`).
- **Real modules on a tiny dataset in the official on-disk format:**
  - bin extraction and a constant column dropped;
  - NaN -> 0;
  - an unseen test category mapped to `train_max + 1`;
  - ordinal codes checked column by column;
  - monotone quantile outputs;
  - determinism and seed sensitivity;
  - relative path through `TABPACK_DATA_DIR`;
  - `FileNotFoundError` for a missing directory.
- **Real Churn (`@pytest.mark.data`):**
  - 6400/1600/2000 rows;
  - `x_num` is `(N, 7)` float32 with no NaN;
  - `x_cat` is `(N, 4)` int64 with 0 <= code < cardinality + 1 and cardinalities
    `[3, 2, 2, 2]`;
  - train codes cover exactly `0..card-1`;
  - `y` is int64 with values in {0, 1};
  - the relative and absolute paths give equal results;
  - `.to('cpu')` round trip.

## Coordination

- Posted `status` #11 (started).
- Posted `blocker` #12 and #20 to integrator. The `.gitignore` rule `data/` also
  swallowed `src/tabpack_repro/data/` and `tests/data/`, so `tests/conftest.py` failed
  to import in every worktree. a03 (#6) and a04 (#7) reported the same problem. The
  integrator fixed it with tag `checkpoint/00b-data-fix` (#60).
- Received #45 from a06, with notes on how to use `categorical.py`. The pipeline
  follows it.
- Merges, each done after the owner posted `done`: `feat/a04-data-dataset` (#29),
  `feat/a05-data-numerical` (#43), `feat/a06-data-categorical` (#49),
  `checkpoint/00b-data-fix`, and `feat/a03-data-download` (#69). The add/add
  conflicts from the skeleton fix were resolved with `--ours`. The resulting
  `dataset.py`, `numerical.py` and `categorical.py` are identical to their owners'
  branches.

## Open issues

- The contract behaviour for the edge cases is my own reading, because the contract
  leaves these cases open:
  - `x_num` with no remaining columns becomes `None`;
  - `bin_policy=None` with binary features, and `cat_policy='one-hot'`, raise
    `ValueError`.

  Churn does not reach any of these paths.
- My branch contains the a03-a06 branches and `checkpoint/00b-data-fix`, so when it
  is merged, those branches come along.
