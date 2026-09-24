# a02-devtools report

## Summary

Added the developer tooling around the lint/test setup: a pre-commit configuration, an
`.editorconfig`, and `tools/dev/check.sh`, a single "is this checkout healthy?" command
for the main checkout or any worktree. All three follow the settings already in
`pyproject.toml` (ruff 0.16.8 from `uv.lock`, 88 columns, single quotes, pytest markers)
and keep `evidence/` byte-exact.

## Files

* `.pre-commit-config.yaml`: `pre-commit-hooks` v6.0.0 (trailing-whitespace,
  end-of-file-fixer, check-yaml, check-toml, check-merge-conflict,
  check-added-large-files `--maxkb=1024`) and `ruff-pre-commit` v0.16.8 (`ruff-check
  --fix`, `ruff-format`). The whitespace fixers and the large-file check exclude `^evidence/`.
* `.editorconfig`: UTF-8, LF, final newline, trimmed whitespace; 4 spaces for Python
  (max 88 columns); 2 for YAML/TOML/JSON/ipynb, Markdown and shell (matching
  `tools/dev/py`); 4 for `pyproject.toml` and `uv.lock`, which already use 4-space
  arrays; tabs for `Makefile`/`*.mk`; `evidence/**` unsets every rewrite rule.
* `tools/dev/check.sh`: runs `ruff check`, `ruff format --check` and
  `pytest -q -m "not slow and not gpu"` through `tools/dev/py`. Flags: `--fast`,
  `--paths PATH...`, `-h/--help`.

## Design decisions

* **Ruff pinned via the official hook repo** at `rev: v0.16.8`, the exact version in
  `uv.lock`. Using a local hook through `tools/dev/py` would avoid the network, but the
  pinned remote hook also works in a fresh clone with no venv. Hook id `ruff-check`
  (current name; `ruff` is only a legacy alias in v0.16.8).
* **Large-file limit of 1 MiB** outside `evidence/`: well above `uv.lock` (95 KB) and
  typical plots/JSON results, low enough to catch checkpoints (`*.pt`) or `.npz`
  artifacts committed by mistake. Transcripts under `evidence/` are exempt, which the task requires.
* **`check-merge-conflict` and `check-yaml/toml` still see `evidence/`**. JSONL and diff
  lines never start with conflict markers, and there is no YAML/TOML there.
* **check.sh resolves the checkout from its own location** (`git -C <script dir>
  rev-parse --show-toplevel`), `cd`s there, and calls `tools/dev/py`, so it checks the
  checkout it belongs to from any current directory, and it takes a concurrency slot
  like every other Python run.
* **Every step runs even after a failure.** The summary lists PASS/FAIL/SKIP per step with
  duration (which includes any wait for a `tools/dev/py` slot). The exit status is 1 if a
  step failed and 2 on usage errors (unknown flag, `--paths` with no path, or a path
  outside the checkout). pytest exit 5 ("no tests collected", for example a directory
  whose tests are all deselected) counts as SKIP, not FAIL.
* **`--paths` routing.** Relative paths are resolved from the current directory, or else
  from the checkout root. This lets the integrator in the main checkout run a worktree's script with
  worktree-relative paths. Ruff receives directories plus files it understands (`*.py`,
  `*.pyi`, `*.ipynb`, `*.md`, `pyproject.toml`). Other files are dropped with a note,
  because ruff parses any explicitly passed file as Python (verified: `.gitignore` gives
  34 syntax errors). pytest receives directories under `tests/` and `tests/**/test_*.py`
  files, and is skipped when none are given. `--paths .` means the whole checkout.
* **No `--force-exclude` in check.sh** (unlike the pre-commit hooks, where it is the
  hook default): a path named explicitly is always checked, even if `pyproject.toml`
  or `.gitignore` excludes it. This matters today because both exclude every
  directory named `data` (see Open issues).
* Style: `set -euo pipefail`, 2-space indent like `tools/dev/py`, colors only on a TTY
  and when `NO_COLOR` is unset. `shellcheck` 0.11.0 is clean at the default level. With `-o all`
  it is also clean apart from SC2250 (brace every variable, which `tools/dev/py` does not do
  either) and SC2310 (the `resolve_path || die_usage` pattern is intended).

## Tests

* `shellcheck tools/dev/check.sh`: clean.
* `tools/dev/check.sh` on the clean worktree (branched from `checkpoint/00-skeleton`):
  **FAIL, as expected**. `ruff check` reports I001 in `tests/_helpers.py` and
  `tests/conftest.py`, and pytest exits 4 with `ModuleNotFoundError: tabpack_repro.data`.
  Both come from the skeleton blocker in board #6/#7/#12/#20: `src/tabpack_repro/data/`
  was never committed, so ruff classifies `tabpack_repro.data` as third-party. This also
  confirms that failures propagate (`exit=1`, `RESULT: FAIL (2 failed: ruff check, pytest)`).
* The same command with the main checkout's untracked `src/tabpack_repro/data/` copied in
  temporarily (git-ignored, deleted afterwards, never staged): **PASS**. `ruff check`
  passed, 52 files were already formatted, and pytest reported `1 passed`.
  `--fast` gives PASS with pytest SKIP. `--paths tests/test_package_imports.py tests/parity
  ../.pre-commit-config.yaml` (run from `tests/`) routes correctly and drops the YAML
  with a note. `--paths tests/parity` run from outside the checkout reports pytest SKIP (no tests
  collected). `--paths .editorconfig` skips all three steps. `--paths`, `--bogus`,
  `--paths nope.py` and `--paths /etc/hosts` each exit 2.
* After merging `checkpoint/00b-data-fix` (integrator #60), `tools/dev/check.sh` on this
  branch gives **PASS**: `ruff check` passed, 59 files were already formatted (the data
  package is now included), and pytest reported `1 passed`.
* Pre-commit, run with `uvx pre-commit` in a scratch `git init` copy of the skeleton
  plus these files (the worktree was untouched, and `PRE_COMMIT_HOME` pointed at the scratchpad):
  `validate-config` passed. `run --all-files` passed all pre-commit-hooks, including a
  2.6 MB `evidence/before/big.jsonl` and a trailing-whitespace file under `evidence/`.
  `ruff-format` passed. `ruff-check` fixed only the two skeleton I001s above. A 2.6 MB
  file outside `evidence/` was rejected (`exceeds 1024 KB`), and trailing whitespace
  outside `evidence/` was fixed while the file under `evidence/` was left byte-identical.

## Coordination

* Posted `status` #3 (start).
* Posted `finding` #56 to `integrator`: ruff's `extend-exclude = ["data", ...]` and
  `.gitignore` `data/` (ruff respects gitignore) both match `src/tabpack_repro/data` and
  `tests/data`, so `ruff check`/`format` on `.`/`src`/`tests` and the pre-commit hooks
  skip the data package. The skeleton data files in the main checkout already fail ruff:
  E501 in `download.py:35` and `numerical.py:25`, I001 in four files, and two files need
  `ruff format`. Suggested fix: anchor both patterns (`/data/`).
* Received integrator `contract` #60 (skeleton fix, crediting #56). As instructed, I
  merged `checkpoint/00b-data-fix` (`git merge --no-edit`) with no conflicts. That fix
  anchors `.gitignore` to `/data/` and ruff's excludes to `data/*` and similar patterns.
  a01-ci's #57 is right that ruff does not accept `/data`, so the `'/data'` I suggested for
  ruff in #56 was wrong and `./data` or `data/*` is the working form. I merged no peer
  branches because the task has no dependencies.

## Open issues

* Resolved by `checkpoint/00b-data-fix`: the skeleton data package and the unanchored
  `data` patterns (#56). Since that merge, `check.sh` and the pre-commit hooks lint
  `src/tabpack_repro/data` and `tests/data`.
* **Do not run `pre-commit install` while agents are running.** Worktrees share
  `.git/hooks`, so installing in one checkout turns the hooks on for all 55 branches, and
  `ruff-check --fix` would start rewriting their commits. pre-commit is not a project
  dependency. `uvx pre-commit ...` works without changing the venv. The hooks need
  network access on first use to fetch the two hook repos.
* Suggestion for a01-ci: `make check` / CI can call `tools/dev/check.sh` directly. In CI
  the main checkout is the repository root, so `tools/dev/py` finds `.venv` there.
