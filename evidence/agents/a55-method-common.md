# a55-method-common: agent report

## Summary

I implemented the shared plumbing in `src/tabpack_repro/methods/common.py` that the
three methods (a30 MLP, a31 homogeneous, a32 TabPack) and a33 (conservative) call:
`setup_run`, `base_report`, `best_member` and `write_run`. The frozen contract is
unchanged. The report fields follow the schema in `methods/report.py`. I posted `done`
early (#97) with the core tested against monkeypatched dependencies, so the method agents
could merge. I then added tests with the real a07 pipeline, a29 utils and the real
Churn data, and (after a28) an artifact round trip.

## Files

- `src/tabpack_repro/methods/common.py`: implementation. Added private helpers
  `_val_score`, `_to_float32_numpy` and `_save_npz_atomic`, plus module constants
  `REPORT_FILENAME`, `PREDICTIONS_FILENAME`, `CONFIG_FILENAME` and `PREDICTION_PARTS`.
- `tests/methods/test_common.py`: unit tests with recording fakes, and `data`-marked
  tests on real Churn.
- `evidence/agents/a55-method-common.md`: this report.

## Design decisions

- **setup_run order**: `seed_everything(seed)` -> `resolve_device(training.device)` ->
  `build_dataset(data).to(device)` -> `make_autocast(training.amp_dtype, device)` ->
  `make_score_fn(dataset.y[part], dataset.task)` for each of `PARTS` (labels are already
  on the device, so scoring never copies labels) -> `y_true` -> `env`. The seed comes
  first so that everything after `setup_run` is reproducible. The data transform has
  its own `data.seed` and does not touch the global RNG.
- **y_true** holds owned numpy copies (`.cpu().numpy().copy()`). On CPU, `.numpy()`
  would share memory with the dataset's label tensors, so changing the report labels
  could corrupt the training labels. The copy is cheap (10k labels).
- **env** = `describe_device(device) | {'git_commit': git_commit()}`. This keeps a29's
  extra `cuda` key next to the schema keys `device/gpu/torch/git_commit`.
- **base_report** returns only the common fields, in schema order: `schema_version`,
  `method`, `dataset`, `seed` (cast to a plain `int`), `config` (`config_to_dict`) and
  `env` (a copy, so editing the report does not change `ctx.env`). The caller adds
  `metrics`, `members`, `ensemble`, `best_member`, `n_epochs`, `n_steps`, `time_sec`
  and `history`. `dataset` is `Path(config.data.path).name`, so `churn`, `churn/`
  and `/abs/.../churn` all give `churn`. A config without a `data` section
  (`ConservativeEvalConfig`) gives `dataset=None` instead of crashing. Its per-seed
  runs are expected to pass the `TabPackConfig` they train.
- **best_member** ranks by `member['metrics']['val']['score']`. A strict `>` keeps the
  first member on ties. NaN never wins (an undefined ROC-AUC, for example). It returns
  `None` for an empty list or when every score is NaN. A member without a val score
  raises `ValueError`. It returns `{'id': int(id), 'metrics': deepcopy(metrics)}`,
  with the member's full train/val/test metrics. Numpy and 0-d tensor scores and ids
  are accepted.
- **write_run** checks and converts before it writes anything. `predictions` must have
  exactly the keys `val` and `test`, otherwise `ValueError` and nothing is written.
  Values may be numpy arrays or tensors of any float dtype, including bf16 and CUDA
  tensors that require grad, and are stored as float32. The output directory is
  created with parents. `report.json` goes through a29's `dump_json` (atomic, strict
  JSON). `predictions.npz` is written with `np.savez` into a temporary file in the
  same directory, then `os.replace`d; it is written through a file object so numpy
  cannot append a second `.npz`. `config.toml` goes through a28's `dump_config`.
  Existing files are overwritten.

## Tests

`tools/dev/py -m pytest tests/methods/test_common.py -q` -> see the final count in the
Coordination section (35 passed at the last run, all on CPU, under 2 s).

- Fakes (the dependencies were not implemented yet): call order and arguments of
  `setup_run`, the device passed to `.to()`, `env` merging, `y_true` owned copies,
  score functions for every part (checked against `score_pack`), the file layout and
  dependency file names of `write_run`, float32/tensor/bf16 conversion, overwriting,
  no leftover temporary files, and key validation before writing (5 bad key sets).
- Direct: `base_report` fields, key order and types for mlp/homogeneous/tabpack
  configs, the dataset basename for 4 path forms, a numpy seed, the config without a
  data section, JSON-serializability. `best_member`: empty, highest, first on ties,
  negative and `-inf` scores, NaN skipping, numpy/tensor scores, returned copy,
  missing val score.
- Real (`@pytest.mark.data`, `churn_dir`, device `cpu`, `amp_dtype=None`): the Churn
  sizes 6400/1600/2000, 7 numerical features, categorical cardinalities `[3, 2, 2, 2]`,
  dtypes, labels in {0, 1}; score functions on dummy predictions (perfect = 1.0,
  all-zero = majority rate, random = `compute_metrics` score); a real env with a
  40-hex commit; reproducible global torch/numpy RNG after `setup_run`; and a full
  `setup_run -> base_report -> best_member -> write_run` round trip read back with
  `load_json`, `load_config` and `np.load`.

## Coordination

- Merged `checkpoint/00b-data-fix`, then `feat/a08-metrics`, `feat/a03-data-download`,
  `feat/a04-data-dataset`, `feat/a05-data-numerical` and `feat/a06-data-categorical`
  one at a time (the octopus merge failed). a04/a05/a06 hit add/add conflicts with the
  checkpoint stubs; I resolved each with the peer's version (`git checkout --theirs`)
  and checked that the result matches their branch.
- Merged `feat/a07-data-pipeline` (after #91) and `feat/a29-utils` (after #99).
- Posted #76 (status) and #97 (`done`, with the full behavior summary for
  a30/a31/a32/a33). a31 (#102) and a32 (#101) reported that they were merging this
  branch. No questions were addressed to me.

## Open issues

- None in the contract. `base_report` for a config without `data` returns
  `dataset=None`. That is a documented fallback, not a schema value.
