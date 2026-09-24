# a24-sampler report

## Summary

Implemented `sample_config` and `sample_configs` in `src/tabpack_repro/sampler.py`.
With the same space and seed, the configs are identical to the official
`HyperparameterSampler(type='RandomSampler', seed=seed).ask(i)` outputs.
**Parity with the official Churn run: exact match.** `sample_configs(space, 64, seed=0)`
reproduces all 57 member configs recorded in
`.reference/tabpack/experiments/tabpack/churn/main/report.json` (matched by
`report.id`, with bitwise float equality). The 7 ids missing from the report
(10, 12, 24, 25, 28, 31, 44) are members the official run did not record, so there is
nothing to compare for them. So optuna 4.9.0's `RandomSampler` gives the same stream as
the version used for the official run.

## Files

* `src/tabpack_repro/sampler.py`: the implementation, plus the private helpers
  `_suggest` and `_sample_tuned`.
* `tests/test_sampler.py`: 37 tests.
* `evidence/agents/a24-sampler.md`: this report.

## Design decisions

* **Same optuna calls, in the same order, as the official code.** For
  `int`/`uniform`/`loguniform` the calls are `suggest_int`/`suggest_float`/
  `suggest_float(log=True)`. A third numeric argument becomes `step=`.
  `categorical` calls `suggest_categorical(label, choices)`. `?dist` flips
  `suggest_categorical('?<label>', [False, True])` and then samples under `<label>`.
  `$list` samples under `<label>.<i>`. Labels are dot-joined key paths, with list
  indices as path parts (`layers.0.width`). Call order sets the RandomSampler stream, so
  a test pins the exact call sequence for the official space.
* `sample_configs` creates
  `optuna.create_study(sampler=RandomSampler(seed=seed), direction='maximize')` and
  calls `study.ask()` once per member, in order. As a result, a smaller `n` gives a
  prefix of a larger `n`. The global `random` and `numpy` RNG state is not touched.
* **Logging.** Optuna's verbosity is set to WARNING only for the duration of the call
  and restored in `finally`, including when an error is raised. This silences the INFO
  line "A new study created in memory".
* **Stricter than the official code where it would fail silently:**
  * Unsupported value types (for example sets or tuples) raise `TypeError`. The official
    code returns `None` for them.
  * Unknown distributions and malformed `_tune_` lists raise a `ValueError` that names
    the label. The official code raises a bare `KeyError`.
  * A dict-level `'_tune_'` key raises `ValueError`, as in the official code.
  * `None` is treated as a constant.
  * `n` must be a non-negative int. `n=0` returns `[]`.
* **No shared objects.** A `?` default that is not sampled is deep-copied, and
  constant lists and dicts are rebuilt for each member, so members never share mutable
  objects with each other or with the space.

## Tests

`tools/dev/py -m pytest tests/test_sampler.py -q` returns **37 passed** (about 1 s).
With `TABPACK_REFERENCE_DIR=/nonexistent`, 35 pass and 2 are skipped (the two official
run checks). `ruff check` and `ruff format --check` are clean on both owned files.

The tests cover:

* ranges and types for the official space;
* the log-uniform and uniform distribution shapes;
* `step`, `categorical`, and `$list`;
* `?uniform`, which yields exactly the default for some members and sampled values for
  others (including a default outside the sampled range);
* `?` flags recorded in `trial.params`;
* determinism for the same seed, different results for different seeds, and the prefix
  property;
* constants, nested dicts and lists passing through unchanged;
* label paths;
* the default `TabPackConfig().space` with `n_models=32`, including a JSON round trip;
* error cases;
* logging silencing and restoration, with a control case showing that INFO is emitted
  without silencing;
* the official space: `config.json` equals `_official_space()`;
* the 57 configs of the official report.

An ad hoc cross-check (not committed, because importing official code belongs in
`tests/parity/**`, owned by a43) compared the output with the official
`lib.tools.tune._sample_config`. It covered 2 spaces × 4 seeds × 50 members, including
step, categorical, `$list`, nested lists and `?int`/`?loguniform`, and all were equal.

## Coordination

* Merged `checkpoint/00b-data-fix` as instructed.
* Board: posted `status` (#80) and `done`. No messages were addressed to a24.
* No peer branches were merged, since there are no dependencies.
* a32 and a43 can merge `feat/a24-sampler`.

## Open issues

* None blocking.
* For a43: the official report ids are member indices into `ask(0..63)`, and 57 of the
  64 are present.
