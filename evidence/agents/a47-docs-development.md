# a47-docs-development

## Summary

Wrote `docs/DEVELOPMENT.md` and `CONTRIBUTING.md`.

`docs/DEVELOPMENT.md` has ten sections: day-to-day commands; how the repository was
built (a frozen-contract skeleton, one worktree and `feat/<id>` branch per agent, the
coordination board, `--no-ff` merges with `tools/integrator/merge_branch.sh`, and the
checkpoint tags, with a Mermaid overview); what went wrong and how it was handled
(the unanchored `data/` ignore rule reported by five agents and fixed in
`checkpoint/00b-data-fix`, the cap of 20 concurrent sub-agents and the rolling
launches, the API usage-limit interruption after which all 20 running agents were
resumed with their context, the missing parity dependencies); the developer tools
(`tools/dev/py`, `board.py`, `check.sh`, `merge_branch.sh`, `tools/evidence/*`,
Makefile, CI, pre-commit, editorconfig); the `evidence/` layout; code style; testing
strategy and markers; how to add a method or a dataset and how to change a contract;
dependencies; and a step list for running another parallel wave. The agent rules are
linked from `AGENTS_PROTOCOL.md`, not repeated.

`CONTRIBUTING.md` is the short human-facing guide: setup, branch naming (`feat/`,
`fix/`, `docs/`), Conventional Commits with eight real examples from `git log main`,
a pull request checklist, and code review expectations.

## Files

* `docs/DEVELOPMENT.md` (new)
* `CONTRIBUTING.md` (new)
* `evidence/agents/a47-docs-development.md` (this report)

## Design decisions

* Every tool, flag, path and environment variable was checked against the files in
  the worktree: `tools/dev/py` (`TABPACK_VENV`, `TABPACK_SLOTS`=4, `TABPACK_GPU`,
  `TABPACK_TIMEOUT`=1800, `TABPACK_DATA_DIR`, `TABPACK_REFERENCE_DIR`,
  `OMP_NUM_THREADS`=3, lock files in `.coord/locks/`), `board.py` (subcommands, kinds,
  the `--for` filter semantics, `TABPACK_BOARD`), `check.sh --help`,
  `merge_branch.sh`, `Makefile`, `ci.yml`, `.pre-commit-config.yaml`,
  `.editorconfig`, and `pyproject.toml` (ruff rules, `quote-style`, excludes, pytest
  markers and options, uv groups and the cu128 index). uv flags (`add --group`,
  `lock --upgrade-package`, `sync --locked`, `export --frozen`) were checked with
  `uv 0.12.8 --help`.
* The history comes from the evidence, not from memory: commit `e8e129c` and its
  message, board messages #6/#7/#9/#12/#14/#20 (the five data agents), #56/#57/#62
  (ruff exclude anchoring), #60 (merge the fix tag), #63 (rolling launches), #133
  (parity deps, `07af200`), the board gap from #133 (14:09) to #134 (16:52) for the
  interruption, and `evidence/checkpoints/00-skeleton.md` / `01-foundations.md`.
* One correction found while checking: CI installs with `uv export --frozen`, which
  does not check that `uv.lock` matches `pyproject.toml`. Only `make install`
  (`uv sync --locked`) catches a stale lock. The doc says so instead of claiming
  that CI fails.
* `tools/evidence/*` (except `redact_emails.py`) is a52's and was not committed yet.
  Its usage lines were taken from the headers of the files in a52's worktree (read
  only) and are described at the level of purpose and main arguments.
* The method and dataset recipes name the concrete contracts: `CONFIG_CLASSES`,
  `MethodName` and `AnyMethodConfig` in `config.py`; `methods/common.py` helpers;
  the `methods/report.py` schema; `EXPECTED_FILES` in `tests/test_config.py`; the
  CLI method table, `scripts/run_churn.py` and `METHOD_ORDER` in
  `reporting/summarize.py`; `metrics._SUPPORTED_SCORES`; regression's
  `NotImplementedError`; the Churn-only search space, runner and reference.
* Branch prefixes follow the task (`feat/`, `fix/`, `docs/`). Test, CI and build
  work without a bug fix goes under `feat/`.

## Tests

Documentation only; no code changed.

* `tools/dev/check.sh --fast --paths docs/DEVELOPMENT.md CONTRIBUTING.md`: ruff check
  and ruff format --check PASS.
* A script checked every relative link and `#anchor` in both files against the
  files and the GitHub slugs of the headings. All resolve except
  `REPRODUCIBILITY.md`, which a46 is writing and which resolves after merge.
* Prose lines are at most 88 characters; only table rows are longer.
* Mermaid was checked by eye (no Mermaid CLI installed). Labels are quoted, the edge
  label uses the `-.->|"..."|` form, and no angle brackets appear in labels.

## Coordination

* Posted `status` (#142) at the start and `done` at the end.
* Merged `main` three times while working (233930c, 5881f47, 7e321b9), as instructed.
* Read without merging: `feat/a34-cli` (`_METHOD_MODULES`), `feat/a35-experiment-runner`
  (`RUN_METHODS`, `DATASET`), and a52's worktree (evidence tool headers).
* No questions were addressed to a47.

## Open issues

* After a52 merges, check that the `tools/evidence/*` table in DEVELOPMENT.md section
  4.5 still matches the committed scripts (names and arguments).
* The descriptions of `tests/parity/test_parity_{optim,ensemble,data,sampler}.py` and
  `tests/integration/*` come from the roster and the board, because those files had
  not landed on `main` yet.
* `make download` and the `tabpack-repro download --name` command in section 8.2
  need a34's CLI, which is not merged yet.
