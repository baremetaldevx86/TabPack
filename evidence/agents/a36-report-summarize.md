# a36-report-summarize: summaries across seeds and methods

## Summary

Implemented `collect_runs`, `summarize`, `to_markdown` and `write_summary` in
`src/tabpack_repro/reporting/summarize.py`. The input is the run-report schema in
`methods/report.py`. The output is a self-describing `summary.json` plus
`summary.md` and `summary.csv`. The conservative protocol is counted only through its
aggregate report, so no run is counted twice. All statistics use `statistics.stdev`
semantics (ddof = 1), and the std is 0.0 when there is only one seed.

## Files

| Path | Content |
| :-- | :-- |
| `src/tabpack_repro/reporting/summarize.py` | Implementation; the summary JSON schema is documented in the module docstring |
| `tests/reporting/test_summarize.py` | 37 tests on fabricated reports (stdlib JSON files in the documented `runs/churn` layout) |
| `evidence/agents/a36-report-summarize.md` | This report |

## Design decisions

* **No double counting.** `collect_runs` loads every `report.json` below `runs_dir`
  in path order and adds `"path"` (the report.json file path). A
  `tabpack-conservative-seed` report that lies below a `tabpack-conservative`
  aggregate is not returned at the top level. Instead it is nested, sorted by seed,
  under `aggregate["seed_runs"]`. If a per-seed report has no aggregate above it, it
  stays at the top level and a warning is logged. `summarize` also ignores any
  top-level per-seed report, so passing a flat list with both gives the same summary.
  The conservative scores always come from the aggregate's `seeds` and `scores`. The
  nested reports only supply the time, K, the ensemble size, and the member and
  best-member test scores. The aggregate's own `mean` and `std` are recomputed rather
  than trusted, so every row uses the same statistics.
* **Summary JSON shape** (posted to a37, board #110, and acknowledged in #118):
  `{schema_version, dataset, metric: 'score', std_ddof: 1, methods: [ROW...]}`. Rows
  come in canonical order (mlp, homogeneous, tabpack, tabpack-conservative, then any
  unknown methods sorted by name). Each ROW has `method`, `display_name`, `n`,
  `seeds` (sorted) and `paths`, plus these blocks: `test`, `val`, `time_sec`,
  `n_models`, `ensemble_size`, `ensemble_n_unique`, `member_test` and
  `best_member_test`. A BLOCK is `{n, mean, std, min, max, values}`, with `values`
  aligned to `seeds`. The statistics skip missing (None) values. An optional block is
  None when no value is available, for example the member blocks of the plain MLP.
  The conservative row also has `aggregate_path`, `source_run` and `selected_ids`.
  Scores stay raw fractions. Only the markdown converts them to percentages.
* **K and sizes come from the data.** K is read from `report["n_models"]` first (a32
  writes it at the top level, and its `members` lists only the finished members),
  then from `config["n_models"]`, then from the number of members. For the
  conservative protocol, K falls back to `len(selected_ids)`. When `ensemble` is null
  but there are several members, as in the homogeneous uniform average, the ensemble
  size is the number of members. The display names are `MLP`,
  `MLP ensemble (homogeneous, K=<K>)`, `TabPack (reduced, single run)` and
  `TabPack (reduced, conservative)`.
* **A run counts as an ensemble** when it has an `ensemble` dict, more than one
  member, or an `n_models` value. The plain MLP has one member and no `n_models`, so
  its ensemble statistics are None.
* **Validation.** `ValueError` for a duplicate (method, seed), for reports from
  different datasets, for more than one conservative aggregate, for a missing
  val/test score, or when the aggregate's seed and score lengths do not match.
  `TypeError` for a missing or non-integer seed, a missing method, or non-object JSON
  (ruff 0.16 TRY004). `collect_runs` rejects a `schema_version` other than 1 and
  raises `FileNotFoundError` when the directory is missing.
* **Markdown** has four parts. The main table has the columns Method, Test acc. (%),
  Val acc. (%), n and Mean time (s), with values such as `86.12 ± 0.35`. Next come an
  ensemble table (models, ensemble size, unique members, member test accuracy, best
  single member test accuracy) and a per-seed test accuracy table with `-` for seeds
  a method is missing. Last is a note on the conservative protocol: its source run,
  selected ids and seeds, and the fact that its std excludes config sampling. The
  markdown depends only on the summary dict, so it can be rebuilt from
  `summary.json`.
* **Files.** `summary.json` is written with `utils.io.dump_json` (a29). `summary.md`
  and `summary.csv` are written through a temporary file and a rename. The CSV has one
  row per method with raw fractions at full precision, and `;` joins the seeds and
  per-seed values.

## Tests

`tools/dev/py -m pytest tests/reporting/test_summarize.py -q` gives 37 passed in
0.3 s. `ruff check` and `ruff format --check` pass on the owned files.

The tests cover:

* the statistics compared with `statistics.mean` and `statistics.stdev`, including an
  exact ddof = 1 check and n = 1 giving std = 0;
* order independence under shuffling, seed sorting, and unknown methods placed last;
* nesting and orphan handling in `collect_runs`, and no double counting when the
  input is a flat list;
* conservative scores taken from the aggregate even when the per-seed reports
  disagree, with and without per-seed reports and with only some of them present;
* K taken from the data, including the top-level `n_models`;
* the validation errors;
* a full markdown snapshot, format details, missing seeds, and an empty summary;
* `write_summary` output: the JSON round trip, the markdown rebuilt from the JSON,
  the CSV content, and overwriting.

The fabricated aggregate uses the same keys as a33's `conservative.run`.

## Coordination

* Merged `main` twice (foundation branches and a29 utils) and `feat/a29-utils` after
  its `done` (#99).
* #105 from a37 asked for the summary shape. I answered with the exact shape in #110,
  and a37 confirmed in #118 that it codes against it.
* Read a32's #126 (top-level `n_models`, only finished members in `members`) and
  adapted how K is read. Checked a33's aggregate keys on `feat/a33-method-conservative`
  (`seeds`, `scores`, `selected_ids`, `source_run`, `dataset`, and a `config` without
  `n_models`). They are compatible.

## Open issues

* The markdown labels `score` as accuracy, which holds for Churn (binary
  classification). Other task types would need a different label.
* Paths in `summary.json` are stored as `collect_runs` found them. They are relative
  when the CLI passes a relative `--runs-dir`, such as `runs/churn`.
