# a28-config: config loaders and the Churn config files

## Summary

I implemented `config_from_dict`, `load_config` and `dump_config` in
`src/tabpack_repro/config.py`, together with private helpers and a small TOML 1.0
writer (the stdlib has no TOML writer). I also wrote the four commented Churn configs
and 78 tests. `load_config(dump_config(c)) == c` holds for every config type,
including `configs = None`, `[]` and a list, and explicit `None` in the optional
string settings. I did not change any dataclass field or default, or any public
signature.

## Files

* `src/tabpack_repro/config.py`: bodies of `config_from_dict`, `load_config` and
  `dump_config`, the private helpers below them, and the new stdlib imports.
* `configs/churn/mlp.toml`, `homogeneous.toml`, `tabpack.toml`,
  `tabpack-conservative.toml`.
* `tests/test_config.py`.
* `evidence/agents/a28-config.md` (this report).

## Design decisions

* **Dispatch and recursion.** `data['method']` is required and selects the class in
  `CONFIG_CLASSES`. A missing or unknown method raises `ValueError` that lists the
  valid methods. The fields are then built recursively from
  `typing.get_type_hints` (cached per class, because the module uses
  `from __future__ import annotations`). The converter handles nested dataclasses,
  `X | None`, `Literal`, `list[...]`, `tuple[...]` (fixed length and variadic),
  `dict[str, ...]`, `Any` and the scalars.
* **Unknown keys.** An unknown key raises `ValueError("unknown config key
  'training.patiance' (did you mean 'training.patience'?); valid keys in
  'training': ...")`, at every level. The path uses dots for tables and `[i]` for
  list items. Free-form fields (`space` and the entries of `configs`) accept any
  string keys.
* **Missing keys** take the dataclass defaults at every level. A `space` table
  *replaces* the default search space and is never merged into it. This is
  documented in the docstring and in `tabpack.toml`.
* **Types.** `int` fields reject `bool` and `float` (so `seed = 1.0` is an error).
  `float` fields also accept ints and convert them to `float`. `bool`, `str` and
  `Literal` fields must match exactly. Numpy scalars are accepted through
  `numbers.Integral` and `numbers.Real`. A type error raises a private
  `_ConfigTypeError(TypeError, ValueError)` whose message names the key path, so
  callers can catch it either way. Free-form values are deep-copied, tuples become
  lists, and nothing in the result aliases the input.
* **None in TOML (TOML has no null):**
  * A field whose default is `None` (only `TabPackConfig.configs`) is omitted when
    it is `None`. `configs = []` and `[[configs]]` tables are written as they are,
    so `None`, `[]` and a list stay distinct.
  * In a `str | None` field (`data.num_policy`, `data.bin_policy`,
    `data.cat_policy`, `training.amp_dtype`), the exact string `"none"` means
    `None`. `config_from_dict` also maps it, so a plain `tomllib.load` followed by
    `config_from_dict` works too. JSON `null` is accepted as well, for example from
    a `report.json` config. To keep this unambiguous, `dump_config` raises
    `ValueError` for a config that holds the literal string `'none'` in such a
    field. `'None'` and `'none'` in non-optional fields such as `device` stay
    plain strings.
  * `None` anywhere else (inside `space` or a member config, or in a non-optional
    field) raises `ValueError` with the key path.
* **Writer.** Plain keys come first, then `[sub.tables]`, then `[[arrays.of.tables]]`.
  A non-empty table without plain keys gets no header of its own, because its
  sub-headers imply it. Arrays can mix types (`["_tune_", "int", 1, 4]`). Dicts
  inside arrays, other than a key whose value is a list of tables, become inline
  tables. Floats are written with `repr`, which gives the shortest exact round trip
  (`0.0001`, `0.005`, `1e-08`, `-0.0`, `5e-324`), and `nan`/`inf`/`-inf` are
  written as such. Ints stay ints. Strings and keys use TOML basic-string escapes
  (control characters and DEL become `\uXXXX`). Keys that are not bare are quoted.
  `dump_config` creates parent directories, writes UTF-8, puts a 3-line header
  comment on top that explains the None rules, and rejects a config whose `method`
  field would load back as a different class. `load_config` adds the file path as
  an exception note (`add_note`) on any decode or validation error.
* **Config files.** Every file writes out every knob with its default value and a
  comment, and points to `docs/EXPERIMENT.md` (sections 2, 3.x, 4, 5.3 and 8, as in
  a45's draft). The test checks that each file loads to the dataclass defaults and
  has exactly the same keys and value types as a dump of the expected config, so
  the files cannot silently drift from `config.py`. In `tabpack-conservative.toml`,
  `source_run = "runs/churn/tabpack/seed-0"` and `n_seeds = 5`.
* I deliberately did **not** add semantic validation beyond types, such as
  `n_models == len(configs)`. That is the method's job (a32), and
  `config_from_dict` stays a faithful parser.

## Tests

`tools/dev/py -m pytest tests/test_config.py -q` gives **78 passed** in about
0.6 s. `ruff check` and `ruff format --check` on the owned Python files pass.

The tests cover:

* Round trips (file and `config_from_dict(config_to_dict(c))`) for default and
  non-default configs of all four types, compared with a type-strict comparison
  (1 vs 1.0, list vs tuple, -0.0).
* `configs` as `None`, `[]`, a list, and entries with nested or empty tables.
* Explicit `None` in each optional string, alone and all at once.
* The reserved `'none'` string, and `None` or unsupported types inside free-form
  values.
* Unknown keys at the top level, nested levels and inside the wrong section, with
  the full path and a suggestion.
* Wrong types (17 cases) and `Literal` checks.
* Int accepted for float fields; numpy scalars.
* No aliasing of the input; conversion of tuples to lists.
* Error notes naming the file path; creation of parent directories.
* Exact float round trips (1e-4, 0.0001, 5e-3, 1e-8, 1/3, subnormals, ±inf, -0.0,
  nan), and ints and bools keeping their types.
* Mixed-type arrays keeping their element types.
* String and key escaping (quotes, backslashes, control characters, DEL, non-ASCII,
  dotted, empty and quoted keys).
* Deeply nested free-form structures, and 300 randomly generated nested
  structures (seeded).
* The exact set of files in `configs/churn`, each loading to the expected defaults
  and pointing to `docs/EXPERIMENT.md`.

## Coordination

* Merged `checkpoint/00b-data-fix` first, as instructed.
* Posted `status` (#75) at the start.
* Answered a45's question #87 (#131): the file names are `mlp.toml`,
  `homogeneous.toml`, `tabpack.toml` and `tabpack-conservative.toml`.
* I merged no peer branches, because there are no runtime dependencies.
* a30 (#130) is waiting for this branch to commit its tests. a34, a35 and a55
  depend on it.

## Open issues

* `docs/EXPERIMENT.md` section 8 runs the conservative protocol through
  `tabpack-repro conservative --source-run ... --n-seeds 5`, not through
  `configs/churn/tabpack-conservative.toml`. The file mirrors those flags. a34 and
  a35 decide whether `run --config tabpack-conservative.toml` is also supported.
* A config built directly (not through `config_from_dict`) with tuples inside
  `space`/`configs` loads back with lists, so it is equal only after
  normalization. Every config produced by `config_from_dict`, `load_config` or the
  sampler (lists) round-trips exactly.
