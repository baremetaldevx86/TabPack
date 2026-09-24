# Contributing

Thanks for helping with this TabPack reproduction. This page is the short version.
The details are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), and the code is
explained in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

You need Linux (the project was developed on WSL2), git, and
[uv](https://docs.astral.sh/uv/) (CI uses 0.12.8). uv installs Python 3.12 itself if
it is missing. An NVIDIA GPU is optional; the locked PyTorch build targets CUDA 12.8.

```bash
git clone https://github.com/baremetaldevx86/TabPack.git
cd TabPack
make install          # uv sync --locked: .venv exactly as pinned by uv.lock
make download         # the Churn dataset into data/churn/
make reference        # optional: the official code, needed only by parity tests
tools/dev/check.sh    # ruff + fast tests; should end with "RESULT: PASS"
```

Without a GPU, use `make install-cpu` and then `export UV_NO_SYNC=1`, so that
`uv run` keeps the CPU build of torch. To run the hooks on every commit, run
`uvx pre-commit install` once.

Useful commands: `make help`, `make test ARGS="-k linear -x"`, `make format`,
`make test-parity`, `tools/dev/check.sh --paths src/tabpack_repro/nn tests/nn`.

## Branches

Branch off an up-to-date `main` and name the branch by the kind of change, in
lowercase kebab-case:

| Prefix | For | Example |
| :-- | :-- | :-- |
| `feat/` | New behaviour, and tooling, tests or CI that do not fix a bug | `feat/regression-support` |
| `fix/` | Bug fixes | `fix/online-ensemble-ties` |
| `docs/` | Documentation only | `docs/development-guide` |

Branches written by the parallel agents are named `feat/<agent-id>`, for example
`feat/a09-nn-linear`. History is never rewritten on shared branches. Bring in new
`main` commits with `git merge main`, not with a rebase. Branches are merged into
`main` with `--no-ff` merge commits, never squashed, so every commit you make stays
in the history. Keep your commits clean.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org):
`<type>(<scope>): <summary>`. Write the summary in the imperative, starting in lower
case, with no final period, and keep it short. The scope is the package or area
(`data`, `nn`, `optim`, `training`, `ensembles`, `metrics`, `sampler`, `config`,
`configs`, `methods`, `reporting`, `utils`, `cli`, `parity`, `tools`, `evidence`,
`agents`), and you may leave it out for changes that span areas. Real examples from
`git log main`:

| Type | Example |
| :-- | :-- |
| `feat` | `feat(nn): add LinearPack with per-member nn.Linear init` |
| `fix` | `fix(reporting): take K from the TabPack report's top-level n_models` |
| `test` | `test(optim): check AdamWPack member-by-member against torch.optim.AdamW` |
| `refactor` | `refactor(optim): build MuonAdamWPack on AdamWPack's _PackOptimizer base` |
| `docs` | `docs: add the architecture guide for new contributors` |
| `build` | `build(parity): add the official code's embedding packages to the parity group` |
| `ci` | `ci: add GitHub Actions workflow for lint, fast tests and parity` |
| `chore` | `chore(tools): add the integrator's --no-ff merge helper` |

The body says what changed and why, in a short paragraph. Mark a change to a public
contract with `!` (for example `feat(config)!: ...`) and explain it in the body.
AI-assisted commits end with a `Co-Authored-By:` trailer. Agent commits also carry
`Agent: <agent-id>` ([docs/AGENTS_PROTOCOL.md](docs/AGENTS_PROTOCOL.md)).

Stage files by name (`git add <paths>`), not with `git add -A`. Never commit `data/`,
`runs/**/*.npz` / `*.pt`, `.reference/`, or code copied from the official repository.

## Pull request checklist

* [ ] `tools/dev/check.sh` passes: `ruff check`, `ruff format --check` and the fast
      tests (`-m "not slow and not gpu"`). CI runs the same checks plus parity.
* [ ] **Tests.** New or changed behaviour has unit tests in the matching
      `tests/<package>/test_<module>.py`, and they fail without your change.
* [ ] **Parity.** If you change numerics that the official code also computes, run
      `make test-parity` and add or update a test in `tests/parity/`.
* [ ] **Other suites.** If you touched GPU code paths, run
      `TABPACK_GPU=1 tools/dev/py -m pytest -m gpu -q`. If you touched real-data
      behaviour, run `tools/dev/py -m pytest -m data -q` (needs the data).
* [ ] **Docs.** Docstrings describe the new behaviour (they are the contract). The
      docs in `docs/` and the commented `configs/**/*.toml` still match the code,
      commands and defaults.
* [ ] **Dependencies** were changed with `uv add` / `uv lock`, and `pyproject.toml`
      and `uv.lock` are committed together (docs/DEVELOPMENT.md, section 9).
* [ ] **Evidence**, where relevant. Agent work includes
      `evidence/agents/<agent-id>.md`. Changes to results include the regenerated
      files under `results/`. Existing files in `evidence/` are never edited by hand.
* [ ] The description says what changed, why, and how you verified it (commands and
      their results), and links related issues or board messages.

## Code review

Reviewers look for, in order:

1. **Correctness against the contract**: signatures, shapes (`(K, B, d)` packs),
   dtypes, devices, error types and messages, as documented in the docstrings and in
   `src/tabpack_repro/types.py`.
2. **Faithfulness to the paper and the official code** at the pinned commit
   `05a89e2`: same semantics (RNG order, tie-breaking, early-stopping rule),
   bit-exact where the parity tests say so, and any deviation documented in
   [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) section 10.
3. **Reproducibility**: seeds, determinism and experiment defaults. `config.py` is
   the source of truth for defaults, and `docs/EXPERIMENT.md` must agree with it.
4. **Tests that prove it**: edge cases, errors, and comparisons with a naive
   reference.
5. **Style and clarity**: ruff-clean, typed, small functions, comments that explain
   why.

As an author, keep pull requests small and focused, answer every review comment, and
push follow-up commits rather than force-pushing. A pull request is merged with a
`--no-ff` merge after at least one approval and a green CI.

Found a bug? Open an issue with a minimal reproduction: the command, the seed, and
the expected versus the actual result.
