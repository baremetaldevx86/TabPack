# a43-parity-sampler report

## Summary

Added parity tests for `tabpack_repro.sampler` (a24) against the official
`project.tabpack.HyperparameterSampler`. **Everything matches exactly.** For seeds 0..4
and n in (1, 32, 64), on both the official Churn space and a richer space, the member
configs are identical: same types, same dict key order and bitwise-equal floats. The
recorded optuna trials are identical too (labels including the `?<label>` flags, values
and distributions). Seed 0 reproduces all 57 member configs recorded in the official
Churn `report.json`, and the official sampler under the installed optuna 4.9.0 does as
well (control). No findings for a24.

## Files

* `tests/parity/test_parity_sampler.py`: 47 tests.
* `evidence/agents/a43-parity-sampler.md`: this report.

## Design decisions

* **Official side.** `HyperparameterSampler(type='RandomSampler', space=..., seed=s,
  study_kwargs={'study_name': 'x', 'direction': 'maximize'})` with `ask(i)` for
  `i in range(n)`, as in the official `run()`. Our side is
  `sample_configs(space, n, seed=s)`. Each side gets its own deep copy of the space,
  and the test also asserts that `sample_configs` does not mutate the space.
* **Strict comparison.** `_assert_identical` checks the type (so `1` vs `1.0` and
  `True` vs `1` fail), dict key order, list length and `float.hex()` (so `-0.0` vs
  `0.0` fails). I checked by hand that it catches each of these cases and a seed
  mismatch.
* **Trial-level parity.** A separate test runs our public `sample_config` on the
  trials of our own seeded study and compares `FrozenTrial.params` and
  `FrozenTrial.distributions` with the official study's trials
  (`sampler._study.get_trials()`). This pins the labels and the optuna call sequence,
  not just the resulting values.
* **Rich space.** It covers what both implementations support:
  * constants: int, bool, str, bytes, float, `None`, `[]` and `{}`;
  * nested dicts, and plain lists that hold subspaces, `_tune_` leaves and constants;
  * `int` with step, and `uniform` with and without step;
  * `loguniform`;
  * `categorical`, including str, bool and int choices;
  * `?uniform`, `?loguniform`, `?int` with step, and `?categorical`;
  * `$list` over `loguniform`, over `int` with step, and over `categorical`.

  A guard test checks that, across 64 members, every `?` flag takes both values and
  every step or categorical leaf is really exercised. Not covered, because the
  official code either does not support them or behaves differently on purpose (a24
  is stricter): `loguniform` with step (optuna rejects it), `?$list` (official
  `KeyError`), tuples and sets (official returns `None`, a24 raises `TypeError`), and
  dict-level `_tune_`.
* **Churn regression** (kept from a24):
  * `config.json` pins seed 0, `n_models=64`, the `RandomSampler` type and the space,
    which must be identical to `TabPackConfig().space`;
  * the 7 unrecorded ids are pinned as `[10, 12, 24, 25, 28, 31, 44]`, so the check
    cannot pass vacuously;
  * for each n in (1, 32, 64), every recorded id below n must match.
* **Imports.** After the lockfile change on main, `project.tabpack` imports without
  stubs. The `tabpack` fixture still installs placeholder `rtdl_num_embeddings` and
  `rtdl_revisiting_models` modules if they are missing (for example, an install without
  the parity group), and removes them on teardown. I tested this fallback by blocking
  both packages: 41/41 selected tests passed and no stubs were left in `sys.modules`.
  `lib.env` needs no setup, because the sampler path never calls it. The clone is never
  modified.

## Tests

`tools/dev/py -m pytest tests/parity/test_parity_sampler.py -q` returns **47 passed**
(about 5-8 s). The tests are:

* 30 config-parity tests (2 spaces × 5 seeds × 3 n);
* 10 trial-parity tests;
* 1 rich-space guard;
* 3 on the Churn config and report coverage;
* 3 on the seed-0 report regression;
* 1 control running the official sampler against its own report.

`ruff check` and `ruff format --check` are clean on the test file.

## Coordination

* Merged `main` twice: at setup (it includes a24) and after the resume, for the
  lockfile change that adds the rtdl packages to the parity group.
* Board: posted `status` #129 and `done`. No messages were addressed to a43.
* No findings were needed, because a24 matches the official code everywhere.

## Open issues

None.
