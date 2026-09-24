# a48-docs-readme: agent report

## Summary

I replaced the README stub with the top-level landing page. It has these sections:

1. Title and a one-line summary.
2. The paper, with the arXiv and official-code links and the BibTeX copied verbatim
   from the official README.
3. A TL;DR: the question, the data, a table of the three methods, the protocol, and
   the reductions versus the official run.
4. The results placeholder, exactly `<!-- RESULTS:BEGIN -->` and
   `<!-- RESULTS:END -->` with a "Results pending" line between them.
5. "What is TabPack", in 8 sentences.
6. A quickstart built on the real Makefile targets and the CLI contract.
7. A repository-layout table that links to `docs/ARCHITECTURE.md`.
8. A documentation index.
9. How this repository was built.
10. The relationship to the official code.
11. The hardware.
12. A license note that invents no license.

## Files

- `README.md` (rewritten)
- `evidence/agents/a48-docs-readme.md` (this report)

## Design decisions

- **Facts come from files on main, not memory.**
  - The Makefile targets come from `Makefile`: `install` = `uv sync --locked`,
    `download`, `experiment` = `uv run bash scripts/run_churn.sh`, `report`, `test`
    (`-m 'not gpu and not slow'`), `test-parity` (clones the reference) and
    `install-cpu` together with `UV_NO_SYNC=1`.
  - The subcommands come from the `src/tabpack_repro/cli.py` docstring.
  - Method settings and the official number 85.75 ± 0.14 % come from
    `docs/EXPERIMENT.md` and `configs/churn/tabpack.toml`.
  - The hardware and the design decisions come from
    `evidence/checkpoints/00-skeleton.md`: compute capability (12, 0), shown as sm_120,
    and torch 2.11.0+cu128. The 8 GB figure comes from the assignment.
  - The A100 used for the official run is in the official Churn `report.json`.
- **`run_churn.sh` behavior.** The claims about resuming (existing `report.json`
  skipped), writing the summary and figures after the runs, and `ARGS="--dry-run"` come
  from a35's board answers #124 and #125. a35 had not posted `done` when I wrote the
  README, so I kept those claims minimal.
- **Every path was checked.** I checked each path and link in the README with a script.
  All exist on main, or are listed in the roster or `evidence/README.md`: REPRODUCIBILITY,
  DEVELOPMENT, CONTRIBUTING, CONTRACT_AUDIT, `scripts/`, `results/reference/...`,
  `tests/integration/`, `evidence/after/`. A few paths are produced at run time
  (`runs/churn/...`, `results/churn/...`), which is how the Makefile defines them.
- **The concurrent documents are linked with no "in progress" note.** The README should
  stay correct after everything is merged, so it does not say that documents are being
  written.
- **"How this repository was built" states only what the evidence supports.**
  - It says the environment capped concurrency at 20. The source is the integrator's
    board message #63.
  - It says every commit carries an `Agent:` trailer. I checked this on main: all 123
    non-merge commits have one, 13 of them `integrator`.
  - It mentions one interruption by an API usage limit, after which all agents were
    resumed. I did not add details I could not verify.
- **License.** The README states that there is no license file, that the code is
  provided for research reproduction, and that the datasets keep their original
  licenses. It paraphrases the official README's data-license statement.
- **Clone URL.** The clone URL is the repository's `origin` remote
  (`github.com/baremetaldevx86/TabPack`). I left out a CI badge, because I could not
  confirm that Actions runs on that remote.

## Tests

- `tools/dev/check.sh --fast --paths README.md`: ruff check PASS, ruff format --check
  PASS.
- There is no trailing whitespace, and the file ends with a newline.
- The path and link check script found no missing path that is not in the roster,
  except the git-ignored `.reference/tabpack`, the `checkpoint/01-foundations` tag, the
  `feat/<agent-id>` pattern and the run-time output directories.
- No Python changes, so there is no pytest run.

## Coordination

- Posted `status` #139 when I started. I read the board before the commit and before
  finishing. No messages were addressed to a48.
- I merged `main` (4000ad0) at setup. I merged no peer branches.

## Open issues

- These links resolve only after the owners' branches are merged: `docs/REPRODUCIBILITY.md`
  (a46), `docs/DEVELOPMENT.md` and `CONTRIBUTING.md` (a47), `docs/CONTRACT_AUDIT.md`
  (a54), `scripts/run_churn.sh` (a35). The same holds for `evidence/after/`,
  `evidence/diffs/`, `evidence/commits.log` and the archived board, which the
  integrator produces at the end.
- The integrator should check the quickstart bullet about `make experiment` against
  a35's final `run_churn.py`: the 20 runs, the summary and figures, resuming, and
  `--dry-run`.
- The integrator fills the block between `<!-- RESULTS:BEGIN -->` and
  `<!-- RESULTS:END -->` after the runs.
