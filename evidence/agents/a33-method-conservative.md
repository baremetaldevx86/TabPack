# a33-method-conservative: agent report

## Summary

I implemented the paper's conservative evaluation protocol,
`tabpack_repro.methods.conservative.run(config, output_dir)`. It reads a finished
TabPack main run, selects the sorted unique ids of its final online ensemble, looks up
their member configs, and retrains only those configs with `methods.tabpack.run` for
seeds `0..n_seeds-1` into `<output_dir>/seed-<s>/`. It marks each per-seed report with
method `tabpack-conservative-seed` and writes the aggregate `<output_dir>/report.json`
(plus `config.toml`). The run can be resumed. The frozen contract is unchanged; only
private helpers were added.

## Files

- `src/tabpack_repro/methods/conservative.py`: implementation (`run` plus private
  helpers `_check_output_dir`, `_load_source`, `_selected_ids`, `_member_configs`,
  `_seed_config`, `_finished_report`, `_run_seed`, `_score`, `_is_int`, `_is_real`).
- `tests/methods/test_conservative.py`: unit tests against a fake `tabpack.run`, and
  one end-to-end test with the real a32 `tabpack.run` on a tiny on-disk dataset.
- `evidence/agents/a33-method-conservative.md`: this report.

## Design decisions

- **Official semantics** (`scripts/run_tabpack_experiment.py::_evaluate_ensemble` and
  `lib/tools/evaluate.py` at `05a89e2`). Selection is `sorted(set(ensemble ids))`. The
  per-seed config is the source config with `n_models = len(selected)`,
  `configs = selected configs` (in sorted-id order, so new member i is source member
  `selected_ids[i]`), `optimizer.shared_step = True` and `seed = s` for
  `s in range(n_seeds)`. The official code pops `sampler`. Here `space` is kept for
  provenance, because TabPack ignores it when `configs` is set. Everything else is
  copied unchanged: data (with data seed 0), d_block, activation, training and online
  ensemble. The source config is deep-copied for each seed, so no state is shared
  between seed runs.
- **Where the configs come from.** An online ensemble can contain members that never
  finished: training stops when the ensemble patience runs out, and the official
  Churn run finished 57 of 64 members. `report['members']` lists only finished
  members. a32 (#126, #132) therefore also writes `report['member_configs']` (all
  members, index = id), the counterpart of the official `experiments.json`. Configs
  are taken from `member_configs` when it is present and completed from `members`.
  If the two disagree for some id, `ValueError`. A selected id found in neither
  (or with a null config) also raises `ValueError`, naming the ids. Lookup is by id,
  never by list position, because `members` is in stopping order.
- **Source config.** It is read from `report['config']` via `config_from_dict`: that
  is lossless JSON (nulls) and lives in the same file as the members and the ensemble,
  so the three cannot disagree. `config.toml` (`load_config`) is only a fallback for
  reports without a config. The source must have `method == 'tabpack'` and a
  `TabPackConfig`.
- **Per-seed method field.** a32 confirmed (#132) that `run()` stays the only entry
  point and always writes `method='tabpack'`. After `tabpack.run` returns, the module
  reloads `seed-<s>/report.json`, sets `method = 'tabpack-conservative-seed'` and
  rewrites it with `dump_json` (atomic). `tabpack.run` is called through the module
  attribute, so tests can monkeypatch it.
- **Resuming** (the official code removes incomplete seed dirs and skips completed
  seeds). A seed counts as finished when `seed-<s>/report.json` has method
  `tabpack-conservative-seed`, even without `predictions.npz`, which is git-ignored
  and therefore missing from committed runs. A report with method `tabpack` means
  an interruption between `tabpack.run` and the rewrite. Because `write_run` writes
  report.json first, this counts as finished only when `predictions.npz` and
  `config.toml` exist too. The report is then marked; otherwise the seed is retrained
  and its files are overwritten. Finished seeds are still aggregated. Like the
  official `assert load_config(exp) == config`, every seed report's `config` must
  equal the expected per-seed config after a JSON round trip. Otherwise `ValueError`,
  which guards against resuming into a directory trained from another source run.
  Rerunning with fewer seeds aggregates only `range(n_seeds)`.
- **Safety.** The module refuses to overwrite a `seed-<s>/report.json` or
  `<output_dir>/report.json` that belongs to another method, and refuses
  `output_dir == source_run`. The source run is never written to.
- **Aggregate report**: `{schema_version, method: 'tabpack-conservative', dataset,
  config (config_to_dict of the ConservativeEvalConfig), source_run (as given),
  source_seed, selected_ids, n_seeds, seeds, scores: {val, test}, mean: {val, test},
  std: {val, test}}`. Scores are `metrics[part]['score']` of each seed's online
  ensemble. `mean` is `statistics.fmean`. `std` is `statistics.stdev` (ddof=1), and
  0.0 when n = 1. A missing or non-finite score raises `ValueError`. The fields beyond
  the contract (`schema_version`, `dataset`, `config`, `source_seed`) are additive.
- **Validation**: `TypeError` for a non-`ConservativeEvalConfig`. `ValueError` for
  `n_seeds` that is not a positive int, an empty `source_run`, a missing or empty
  ensemble, or non-int ids. `FileNotFoundError` when `<source_run>/report.json` is
  missing.

## Tests

`tools/dev/py -m pytest tests/methods/test_conservative.py -q` gives **38 passed**
(about 6 s; the end-to-end test takes most of it). `ruff check` and
`ruff format --check` on both owned code files are clean.

- 37 unit tests replace `methods.tabpack.run` with a fake that writes a run the way
  the real one does (report.json first, then predictions.npz and config.toml). They
  cover:
  - selection by id, where `members` is in stopping order;
  - the per-seed config (overrides plus everything else copied), with configs
    independent between seeds;
  - per-seed method marking, and the source run left untouched;
  - the aggregate schema: mean, ddof=1 std, std 0.0 for n=1, and a `config.toml`
    round trip;
  - resuming: skip finished seeds, add more seeds, aggregate fewer seeds, mark a
    finished but unmarked seed, retrain a seed without predictions.npz or
    config.toml, treat a marked seed without predictions.npz as finished, and raise
    when resuming from a different source run;
  - refusing foreign seed or aggregate reports, and `output_dir == source_run`;
  - the source config: report["config"] first, config.toml as fallback;
  - `member_configs`: it supplies unfinished members, is enough on its own, and a
    disagreement with `members` raises;
  - errors: a missing report, missing or duplicate members, a member without a
    config, a missing or empty ensemble, non-int ids, a non-tabpack source, a
    non-TabPack config, bad `n_seeds`, an empty `source_run`, a wrong config type,
    and a missing seed score.
- 1 end-to-end test (`slow` marker): a real `tabpack.run` source run with n_models=4
  on a tiny synthetic on-disk dataset (official format: info.json, x_num/x_cat/y,
  splits/default). The conservative evaluation then runs with n_seeds=2. The test
  checks:
  - the per-seed `config.toml` round trip to the expected TabPackConfig;
  - `member_configs` equal the selected configs;
  - the marked seed reports;
  - aggregate scores equal to the seed reports, with their mean and std;
  - a rerun retrains nothing and rebuilds the same report.
- Manual check (not committed): real `tabpack.run` followed by conservative, then
  a36's `collect_runs` + `summarize` + `to_markdown` on the output tree. It produces
  the 'TabPack (reduced, conservative)' row and the selected-ids note.

## Coordination

- Merged `main` (setup), then `feat/a55-method-common` (done #97), `feat/a28-config`
  (done #134) and `feat/a32-method-tabpack` (done #159), and `main` again after #162.
- #107 status (started). #115 question to a32 about a cleaner hook for the method
  field and about the member config/id semantics. a32 answered in #126/#132: keep
  the post-processing, and use `report['member_configs']` because ensemble ids may
  include unfinished members. The implementation was changed to do that.
- #162 (integrator contract: resolve ids via `member_configs`) was already
  implemented. `members` only completes ids that `member_configs` lacks and is
  cross-checked where both have a config. This is a strict superset of the
  requested fallback.
- #146 FYI to a36 (aggregate schema), #147 to a34 (errors and return value), #152
  to a35 (resumability).

## Open issues

- Committed runs keep only `report.json` and `config.toml`, so resuming treats a
  seed as finished from its marked report alone. A seed that was interrupted after
  report.json was written, and whose report was never marked, is retrained only if
  `predictions.npz` or `config.toml` is missing. This relies on the write order of
  `write_run` (a55): report.json first.
- The resume guard compares each seed report's `config` with the expected per-seed
  config after a JSON round trip. If a future TabPackConfig field changes, old seed
  dirs are rejected with a clear `ValueError` and are not silently mixed in. Use a
  fresh output directory in that case.
- `source_run` is stored as given (for example `runs/churn/tabpack/seed-0`), so the
  aggregate stays portable. It is resolved against the current working directory
  when the evaluation runs.
