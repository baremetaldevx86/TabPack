# a34-cli: command-line interface

## Summary

`src/tabpack_repro/cli.py` implements `main(argv) -> int`. It is the entry point
for `tabpack-repro` (`[project.scripts]`) and `python -m tabpack_repro`. It has five
argparse subcommands, as the module contract specifies:

| Command | What it does |
| :-- | :-- |
| `download [--name churn] [--data-dir DIR] [--cache-dir DIR] [--force]` | `data.download.download_dataset`, then prints the dataset path |
| `run --config PATH --output DIR [--seed S] [--device D]` | `config.load_config`, `--seed` / `--device` overrides (`dataclasses.replace`; the device goes to `training.device`), dispatches on `config.method` to `methods.<method>.run`, then prints one result line and the output path |
| `conservative --source-run DIR [--n-seeds 5] --output DIR` | builds `ConservativeEvalConfig(source_run, n_seeds)` and runs `methods.conservative.run` |
| `summarize --runs-dir DIR --output DIR` | `reporting.summarize.collect_runs` / `summarize` / `write_summary`, then prints `to_markdown` |
| `reference --output PATH [--reference-dir DIR]` | `reporting.reference.get_reference_dir` / `extract_churn_reference`, written with `utils.io.dump_json` |

Example output of `run`:

```
mlp  seed=1  val_score=0.83062  test_score=0.82250  time=1.2s
output: runs/churn/mlp/seed-1
```

Exit codes: 0 on success, 1 on runtime errors, 2 on usage errors and 130 on Ctrl-C.
Runtime errors include a missing or invalid config, a failed download, a missing
source run, no reports to summarize, no official clone, and any exception from a
method. Usage errors are argparse errors plus `--seed`/`--device` on a
tabpack-conservative config. `main` never raises `SystemExit`: `--help` returns 0.

## Files

* `src/tabpack_repro/cli.py`: the implementation. The public API is only `main`.
  Every helper is private (`_build_parser`, `_execute`, `_cmd_*`, ...).
* `tests/test_cli.py`: 67 tests.
* `evidence/agents/a34-cli.md`: this report.

## Design decisions

* **Lazy imports.** The module imports only the standard library. Each handler
  imports what it needs (config, methods, reporting). A subprocess test checks that
  `python -m tabpack_repro --help` and every `COMMAND --help` load none of
  torch, numpy, optuna or pandas.
* **Monkeypatch-friendly lookups.** Handlers call `module.function` when they run.
  For example, `importlib.import_module('tabpack_repro.methods.mlp').run` and
  `summarize_lib.collect_runs`. Patching the module attribute is therefore enough
  in tests, and a35/a49 can use the same approach.
* **Validation at parse time (exit 2).** `--seed` must be an integer in `[0, 2**32)`,
  which is the range `seed_everything` accepts. `--device` must be `auto`, `cpu`,
  `cuda`, `cuda:<i>` or `mps`, checked by a regex so that torch is not imported.
  `--n-seeds` must be at least 1.
* **Overrides never mutate the loaded config.** The CLI calls `dataclasses.replace`
  on the config and on its `training` section. The tests check that the result
  equals the loaded config with only `seed` and `training.device` replaced.
* **`run` also accepts a `tabpack-conservative` config** (`configs/churn/tabpack-conservative.toml`).
  `config.method` has four values, and all four dispatch to a method.
  `ConservativeEvalConfig` has no seed and no training section, so `--seed` and
  `--device` are usage errors for it.
* **Early, clear runtime errors.** A missing config file gives
  `config file not found: P`. A loader error gives
  `invalid config file P: ValueError: unknown config key 'training.patiance' (did you mean ...)`.
  A missing `<source_run>/report.json` gives an error that includes the command
  that creates it. Other checks cover a missing runs dir, zero reports, and a missing
  clone or `--reference-dir`. Errors take one line on stderr, prefixed with
  `tabpack-repro <cmd>: error:`. `-v` adds the traceback.
* **Logging.** `logging.basicConfig` sends output to stderr. It does nothing when the
  root logger already has handlers (pytest, or an embedding script). The CLI always
  sets the level of the `tabpack_repro` package logger: INFO by default, DEBUG with
  `-v`, WARNING with `-q`. This makes download progress and method summaries (for
  example a31's INFO line) visible. Results go to stdout.
* **Result line.** The line shows method, seed, val/test score (5 decimals) and time.
  Values come from `report.json` fields (`metrics.*.score`, `time_sec`). A missing
  field prints `n/a`, and a missing `time_sec` falls back to the wall-clock time
  measured by the CLI. For the conservative aggregate, the line shows the mean ± sample
  std of `scores.val` and `scores.test` over seeds.
* The `reference` subcommand writes `extract_churn_reference(...)` with `dump_json`,
  which is how a38 recommended it (#148). The parity test confirms that the output is
  byte-identical to the committed `results/reference/churn_official.json`.

## Tests

`tools/dev/py -m pytest tests/test_cli.py -q` gives **67 passed** in about 15 s (CPU).
`tools/dev/py -m ruff check` and `ruff format --check` on both owned files pass.

* Help: top-level and each subcommand's `--help` exit 0, and each lists the contract
  options. `python -m tabpack_repro --help` works in a subprocess without importing
  torch.
* Usage errors exit 2 in these cases: no command, an unknown command or option, a
  missing required option on each subcommand, an invalid seed or device,
  `--n-seeds 0`, `-v -q`, and `--seed`/`--device` with a conservative config.
* `run` with fake method `run`s: dispatch per method (only the right one is called,
  with the right output dir); `--seed`/`--device` reach the config, alone and
  together; the result line and the output path are printed.
* Runtime errors exit 1 with a one-line message in these cases: a bad config path,
  a loader error, a method exception (the traceback only with `-v`), an unknown
  method, a missing source run (for both `conservative` and `run`), a download
  failure, a missing or empty runs dir, and a missing clone or reference dir.
  Ctrl-C returns 130.
* After merging the real modules:
  * a28: the shipped `configs/churn/*.toml` load and dispatch correctly, with the
    overrides applied. Invalid TOML, unknown keys, wrong types and unknown methods
    exit 1.
  * a30, a31, a32 (`data` marker): tiny one-epoch MLP, homogeneous and TabPack
    runs on real Churn through `run --seed --device cpu`. `report.json` holds the
    overridden seed and device.
  * a33 (`data` marker): a tiny TabPack source run, then `conservative --n-seeds 2`
    through the CLI. The test checks the aggregate report, the per-seed reports and
    the printed mean ± std.
  * a36 (`data` marker): two MLP seeds run through the CLI and are then summarized
    by the real `summarize`.
  * a38 (`parity` marker): `reference` reproduces the committed JSON byte for byte.

## Coordination

* Posted `status` #111 (started).
* Received the FYI from a33 (#147) about conservative error types. They map to
  exit 1 through the generic handler, and the CLI pre-checks the source report
  itself. Also received the answer from a38 (#148) on how to write the reference
  JSON, which is what the CLI does.
* Merged: `main` (setup), `feat/a28-config` (done #134),
  `feat/a30-method-mlp` (#138), `feat/a36-report-summarize` (#145),
  `feat/a38-report-reference` (#151), `feat/a31-method-homogeneous` (#158),
  `feat/a32-method-tabpack` (#159) and `feat/a33-method-conservative` (#167).

## Open issues

* `run` has no override for `training.max_epochs` or other fields. Smoke runs use a
  small TOML file, and every omitted key takes its default.
* `--device` for `conservative` is not supported, because the frozen
  `ConservativeEvalConfig` has no device field. The retraining runs use the device
  recorded in the source run's config.
