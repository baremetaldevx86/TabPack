# a01-ci: CI workflow and Makefile

## Summary

Added a GitHub Actions workflow with three jobs (`lint`, `test`, `parity`) and a
`Makefile` with documented developer targets. CI installs the locked dependencies with
CPU-only torch, using a `make install-cpu` target, so the jobs never download the CUDA
wheels (several GB). The Makefile is the single source of the commands, so running
`make lint`, `make test` or `make test-parity` locally runs exactly what CI runs.

## Files

| Path | Content |
| :-- | :-- |
| `.github/workflows/ci.yml` | CI on push/PR to `main` (+ `workflow_dispatch`): `lint`, `test`, `parity` |
| `Makefile` | `help` (default), `install`, `install-cpu`, `lint`, `format`, `test`, `test-parity`, `test-all`, `reference`, `download`, `experiment`, `report`, `clean` |
| `evidence/agents/a01-ci.md` | This report |

## Design decisions

* **CPU torch without touching the lockfile.** `uv.lock` pins `torch==2.11.0+cu128`
  from the cu128 index plus 15 `nvidia-*`, 3 `cuda-*` and `triton` wheels. `install-cpu`
  runs `uv export --frozen --all-groups --no-emit-project` to get the full pinned set,
  drops `torch|triton|nvidia-*|cuda-*`, installs the rest with `--no-deps`, then
  installs `torch==<locked version>+cpu` from `download.pytorch.org/whl/cpu`, then the
  project (`--no-deps --editable .`), and finally runs `uv pip check`. Every package
  except torch therefore has exactly its locked version, and the torch version is read
  from the lock, not hardcoded. Using `--no-deps` avoids the resolver quietly replacing
  the CPU torch with the CUDA build from PyPI (which `delu`'s torch requirement could
  otherwise trigger). I did not use `--torch-backend`, which is still a preview feature.
* **`UV_NO_SYNC=1` in the test jobs** so `uv run` (used by every Makefile target) keeps
  the CPU environment instead of syncing it back to the CUDA pins.
* **Lint without installing the project**: the job reads the ruff version from
  `uv.lock` (`uv export --only-group dev`, currently 0.16.8) and runs
  `make lint RUFF="uvx ruff@<version>"`. `RUFF_OUTPUT_FORMAT=github` turns violations
  into inline PR annotations.
* **Parity**: `make reference` runs the prescribed `git clone
  https://github.com/yandex-research/tabpack .reference/tabpack` + `checkout 05a89e2`.
  It is idempotent (it fetches only when the commit is missing) and fails if
  `src/project/tabpack.py` is absent, so a failed clone cannot make every parity test
  skip and the job pass anyway. `test-parity` depends on `reference` and passes an
  absolute `TABPACK_REFERENCE_DIR`. `REFERENCE_DIR` defaults to `$TABPACK_REFERENCE_DIR`
  when set.
* **Hygiene**: actions pinned by commit SHA (`actions/checkout` v7.0.1,
  `astral-sh/setup-uv` v10.2.0, the latest tags; setup-uv publishes no floating major
  tag after v7), uv 0.12.8 and Python 3.12 (both match checkpoint 00),
  `permissions: contents: read`, `persist-credentials: false`, per-job timeouts,
  `shell: bash` (pipefail). The uv cache is on, with `cache-suffix` `lint` or `cpu`, so
  the ruff-only lint cache never shadows the test jobs' cache. The concurrency group is
  `workflow + PR number or ref` with `cancel-in-progress: true`.
* **Makefile**: bash with `-eu -o pipefail`; the variables `RUN`, `RUFF`, `ARGS`,
  `FAST_MARKERS`, `REFERENCE_DIR`, `RUNS_DIR` and `RESULTS_DIR` can be overridden (e.g.
  `make test RUN="tools/dev/py -m"` in an agent worktree). `clean` only removes caches
  and build artifacts, and skips `.git`, `.venv`, `.worktrees` and `.reference`.
  `install-cpu` recreates `.venv` (`uv venv --clear`), so it must **not** be run in the
  shared main checkout while agents are using its venv. `make install` restores the
  CUDA environment.

## Tests

* YAML: `tools/dev/py -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
  parses (jobs `lint`, `parity`, `test`). `uvx --from actionlint-py actionlint -verbose
  .github/workflows/ci.yml` reports 0 errors (shellcheck included).
* `make -n <target>` for all 13 targets prints the expected commands, and `make` prints
  the help.
* `install-cpu` was verified without building a large venv, using an empty 72 KB scratch
  venv and `--dry-run`: the exported pins minus the CUDA packages install as listed,
  `torch==2.11.0+cpu` exists on the CPU index for Python 3.12, and the editable install
  of the project resolves. A full `--torch-backend cpu` dry-run resolution of the same
  pins plus torch 2.11.0 produced exactly the pinned set plus `torch==2.11.0+cpu`, so
  the CPU torch's dependencies are already covered by the lock and `uv pip check`
  should pass.
* `make reference REFERENCE_DIR=<scratch>` did a real clone to `05a89e2` in 3 s, and a
  second run was a no-op.
* Ran the CI lint step verbatim (`make lint RUFF="uvx ruff@0.16.8"`). Before the
  skeleton data fix it failed with I001 in `tests/_helpers.py` and `tests/conftest.py`,
  because `tabpack_repro.data` was missing from git. After merging
  `checkpoint/00b-data-fix` it passes: `All checks passed!` and `59 files already
  formatted`.
* `make test RUN="tools/dev/py -m"` (the fast CPU selection through the slot runner)
  gives `1 passed in 0.03s` after the merge.
* `make test-parity RUN="tools/dev/py -m" REFERENCE_DIR=<scratch clone>` clones, checks
  out and runs `pytest tests/parity -q`. It exits with 5 (`no tests ran`) because no
  parity tests exist yet (see Open issues).

## Coordination

* Posted #2 (`status`, started) and #57 (`finding` to integrator). #57 says that in ruff
  `extend-exclude`, `'/data'` excludes nothing, while `'./data'` or `'data/'` excludes
  only the top-level dir (checked on a scratch tree). It also says the CI lint I001
  failure is caused by the missing data package.
* Posted #61 to a35 describing how `make experiment` and `make report` call
  `scripts/run_churn.sh` and the `summarize` CLI.
* Read the data-package blocker reports #6, #7, #12, #20, #14 and #43. In #62 the
  integrator confirmed that `extend-exclude 'data/*'` fixes the ruff scope and asked me
  to merge `checkpoint/00b-data-fix`, which I did (`git merge --no-edit`). No peer
  branches were merged, since there are no runtime dependencies.

## Open issues

* The skeleton data-package problem (`.gitignore` `data/`) is fixed by
  `checkpoint/00b-data-fix`, and CI needs that fix: without it, lint (I001) and every
  test (conftest import) fail on a fresh checkout.
* `pytest tests/parity` exits with code 5 ("no tests collected") until at least one of
  a39-a43 is merged, so the `parity` job fails on a tree without parity tests.
* `make experiment` assumes `scripts/run_churn.sh` (a35) can be run with `bash` from the
  repo root and takes its options as arguments (`ARGS`). The `report` target assumes
  a34's `summarize --runs-dir/--output` CLI as documented in `cli.py`.
* The workflow has not run on GitHub yet: the branch is not pushed, and the protocol
  forbids pushing.
