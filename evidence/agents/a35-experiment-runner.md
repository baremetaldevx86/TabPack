# a35-experiment-runner: the one-command Churn experiment

## Summary

`scripts/run_churn.py` runs the whole experiment in-process through the library:
(1) download Churn, (2) `mlp`, `homogeneous` and `tabpack` for seeds 0..4 (seed-major
order) into `runs/churn/<method>/seed-<s>/`, (3) the conservative TabPack evaluation of
`runs/churn/tabpack/seed-0` into `runs/churn/tabpack-conservative/`, (4) `summarize`
into `results/churn/`, (5) figures into `results/churn/figures/`, (6) the official
reference numbers (refreshed from the clone if it exists, else the saved file). Every
step is a function, so the orchestration is tested with fakes. The script resumes
(finished runs are skipped), keeps going after a failing step, logs with timestamps,
prints a summary table (plus the `summarize` markdown table and the official numbers)
and exits with 1 if any step failed (130 on Ctrl-C). `scripts/run_churn.sh` is the thin
wrapper that `make experiment` calls.

## Files

* `scripts/run_churn.py`: argparse CLI, plan builder, step functions, executor,
  summary printing.
* `scripts/run_churn.sh`: `cd` to the repository root, `uv run python -u
  scripts/run_churn.py "$@"`, appends the output to `<runs-dir>/run.log`.
* `tests/scripts/test_run_churn.py`: 48 tests (no real training).

## Design decisions

* **Flags**: `--seeds`, `--methods` (subset of mlp, homogeneous, tabpack,
  tabpack-conservative; always run in canonical order), `--runs-dir`, `--results-dir`,
  `--configs-dir`, `--device`, `--force`, `--dry-run`, `--skip-download`, plus
  `--reference-file` (default `results/reference/churn_official.json`, so tests and
  alternative result dirs do not touch the committed file) and `--conservative-seeds N`
  (the conservative protocol's `n_seeds`; default from its config).
* **Default paths** are anchored at the repository root but stored relative to the cwd,
  so reports keep short relative paths such as `source_run = "runs/churn/tabpack/seed-0"`
  when run from the root (which the wrapper always does).
* **Config overrides**: `dataclasses.replace` of `seed` and, with `--device`, of
  `training.device`. `DataConfig.seed` stays 0 (official semantics). The conservative
  config gets `source_run = <runs-dir>/tabpack/seed-0`. `--device` cannot reach the
  conservative runs (`ConservativeEvalConfig` has no device; `conservative.run` copies
  the source run's config, which already has the override if the source run was made by
  this script with `--device`); the script logs this.
* **Resume**: a single run is skipped when its `report.json` exists; the summary table
  still shows its scores. The conservative step is always called because
  `conservative.run` resumes per seed by itself (a33, board #152), which also picks up a
  larger `n_seeds`; the dry run shows it as `resume`. `--force` removes the old
  `report.json` before rerunning (so a failed forced run cannot leave a stale report
  that looks finished); for the conservative step it removes the aggregate and
  `seed-<s>/report.json` for `s < n_seeds` only.
* **Failures**: every step runs inside `try/except Exception`; the traceback is logged,
  the other steps continue, failures are listed at the end and the exit code is 1.
  `StepSkipped` (nothing to summarize or plot) is a skip, not a failure. The conservative
  step fails with a clear message when `runs/churn/tabpack/seed-0/report.json` is
  missing. Plots of TabPack seed 0 are skipped with a warning when its report is missing.
  A broken official clone falls back to the saved reference file.
* **Dry run** prints the numbered plan (run/skip/resume/rerun, config, seed, device,
  output), loads and checks every needed config (exit 1 on a config error), and notes
  whether the conservative source run will be produced by the plan. Nothing is written.
* Between steps: `gc.collect()` and `torch.cuda.empty_cache()` (runs are in-process).
* Logging goes to stderr (`%(asctime)s` timestamps, INFO, so library loggers such as the
  download progress show up); the summary goes to stdout; the wrapper uses `python -u`
  and `2>&1 | tee -a` with `pipefail`, so the exit status is the script's. `--dry-run` and
  `--help` are not logged. `$UV` overrides the uv binary and `$RUN_CHURN_LOG` the log path.

## Tests

`tools/dev/py -m pytest tests/scripts/test_run_churn.py -q` -> **48 passed** (~6 s).
Covered: defaults and argument validation, plan order and `--methods`/`--seeds`/
`--skip-download` filtering, full execution order, resume-skip (with scores of skipped
runs in the table), `--force`, failure accounting (steps continue, exit 1, failures
listed), `StepSkipped`, Ctrl-C (exit 130, "not run"), dry-run output/config errors/source
note, the real step functions with fake method runners and fake reporting functions
(seed/device/source_run/n_seeds overrides, conservative `--force` removal, summarize +
plots paths, reference refresh vs saved file vs broken clone), the real configs of a28,
`format_reference` on the committed reference file, and the shell wrapper with a fake
`uv` (runs from the repo root, passes arguments, appends the log, propagates the exit
code, no log for dry runs). `ruff check` and `ruff format --check` pass on the owned files.

Real smoke run (a28, a30, a36, a38 merged; CPU): `run_churn.py --methods mlp --seeds 0 1
--device cpu` into a scratch dir: download ok, MLP seed 0 test 0.8495, seed 1 test
0.8595, summarize ok, plots failed with `NotImplementedError` (a37 not done yet),
reference refreshed from the clone; exit 1 as designed. Re-running through
`run_churn.sh` (with a `uv run` stand-in) skipped both runs, appended to `run.log` and
propagated exit 1.

## Coordination

* Posted status #112.
* Answered a45 (#86 -> #124): in-process calls instead of CLI subprocesses, the step
  order, figure file names under `results/churn/figures/`, resume/`--force`, all flags,
  the log file. The reference step comes last; it is independent of the others.
* Answered a01 (#61 -> #125): `make experiment` (`uv run bash scripts/run_churn.sh
  $(ARGS)`) is compatible; no Makefile change needed.
* Read a33's #152 (conservative.run resumes itself) and changed the conservative step to
  always call it.
* Merged `main`, `feat/a28-config`, `feat/a30-method-mlp`, `feat/a36-report-summarize`,
  `feat/a38-report-reference` (all posted `done`) for the real-config tests and the smoke
  run.

## Open issues

* Not yet exercised for real: homogeneous, tabpack, conservative and plots (a31, a32,
  a33, a37 were not done); their calls follow the frozen contracts and are tested with
  fakes.
* Resume trusts an existing `report.json`: it does not check that the run's stored
  config matches the current TOML. Use `--force` after changing a config.
* `runs/churn/run.log` is not git-ignored; the integrator decides whether to commit it.
* `run_churn.sh` uses `uv run`, which syncs a `.venv` in the directory it runs from; in an
  agent worktree use `tools/dev/py scripts/run_churn.py ...` instead.
