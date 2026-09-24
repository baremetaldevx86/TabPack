# Developer entry points. Run `make` (or `make help`) to list the targets.
#
# Every Python command goes through `uv run`, which keeps .venv in sync with uv.lock.
# In a CPU-only environment created by `make install-cpu` (what CI does), export
# UV_NO_SYNC=1 so that `uv run` keeps the CPU build of torch instead of re-syncing
# the CUDA build pinned by uv.lock.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN ?= $(UV) run
RUFF ?= $(RUN) ruff
PYTHON_VERSION ?= 3.12
# Extra arguments for the underlying command, e.g. `make test ARGS="-k linear -x"`.
ARGS ?=

# The fast suite: CPU-only and quick (the same selection as the CI `test` job).
FAST_MARKERS ?= not gpu and not slow

# Official TabPack code, used only by tests/parity (never copied into src/).
REFERENCE_URL ?= https://github.com/yandex-research/tabpack
REFERENCE_COMMIT ?= 05a89e21b955f12de84889d662e15ca534019aaa
REFERENCE_DIR ?= $(or $(TABPACK_REFERENCE_DIR),.reference/tabpack)

# CPU-only install (`install-cpu`): locked dependencies, torch from this index.
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu
CPU_BUILD_DIR ?= build

RUNS_DIR ?= runs/churn
RESULTS_DIR ?= results/churn

.PHONY: help install install-cpu lint format test test-parity test-all reference \
	download experiment report clean

help: ## Show this help (default target)
	@echo 'Usage: make <target> [VAR=value ...]'
	@echo
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-13s %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
	@echo
	@echo 'Variables: ARGS (extra arguments), RUN (default: uv run), RUFF, FAST_MARKERS,'
	@echo '  REFERENCE_DIR (default: $$TABPACK_REFERENCE_DIR or .reference/tabpack),'
	@echo '  RUNS_DIR (default: runs/churn), RESULTS_DIR (default: results/churn).'
	@echo 'Typical flow: make install download experiment report'

install: ## Create or update .venv exactly as pinned by uv.lock (CUDA 12.8 torch on Linux)
	$(UV) sync --locked

install-cpu: ## Recreate .venv with the locked deps but CPU-only torch (used by CI)
	$(UV) venv --clear --python $(PYTHON_VERSION)
	@mkdir -p $(CPU_BUILD_DIR)
	$(UV) export --quiet --frozen --all-groups --no-emit-project --no-hashes \
		--no-annotate --no-header --output-file $(CPU_BUILD_DIR)/requirements-lock.txt
	grep -vE '^(torch|triton|nvidia-.*|cuda-.*)==' $(CPU_BUILD_DIR)/requirements-lock.txt \
		> $(CPU_BUILD_DIR)/requirements-cpu.txt
	$(UV) pip install --no-deps --requirements $(CPU_BUILD_DIR)/requirements-cpu.txt
	torch_version="$$(sed -n 's/^torch==\([^+;]*\)+cu[0-9]*.*/\1/p' \
		$(CPU_BUILD_DIR)/requirements-lock.txt)"; \
		test -n "$$torch_version"; \
		$(UV) pip install --no-deps --index-url $(TORCH_CPU_INDEX) "torch==$$torch_version+cpu"
	$(UV) pip install --no-deps --editable .
	$(UV) pip check
	@echo 'CPU environment ready: set UV_NO_SYNC=1 so that `uv run` keeps it.'

lint: ## Lint and check formatting with ruff (no changes)
	$(RUFF) check .
	$(RUFF) format --check .

format: ## Apply ruff's safe fixes and format the code
	$(RUFF) check --fix .
	$(RUFF) format .

test: ## Fast CPU tests: pytest -m "not gpu and not slow"
	$(RUN) pytest -m '$(FAST_MARKERS)' -q $(ARGS)

test-parity: reference ## Parity tests against the official code (clones it first)
	TABPACK_REFERENCE_DIR='$(abspath $(REFERENCE_DIR))' $(RUN) pytest tests/parity -q $(ARGS)

test-all: ## Every test (GPU tests skip without CUDA, parity tests without the clone)
	$(RUN) pytest -q $(ARGS)

reference: ## Clone the official TabPack repo at the pinned commit into REFERENCE_DIR
	if [ ! -d '$(REFERENCE_DIR)/.git' ]; then \
		git clone '$(REFERENCE_URL)' '$(REFERENCE_DIR)'; \
	fi
	git -C '$(REFERENCE_DIR)' cat-file -e '$(REFERENCE_COMMIT)^{commit}' 2>/dev/null \
		|| git -C '$(REFERENCE_DIR)' fetch origin
	git -C '$(REFERENCE_DIR)' -c advice.detachedHead=false checkout --quiet \
		'$(REFERENCE_COMMIT)'
	test -f '$(REFERENCE_DIR)/src/project/tabpack.py'

download: ## Download the Churn dataset (python -m tabpack_repro download)
	$(RUN) python -m tabpack_repro download $(ARGS)

experiment: ## Run the Churn experiment: all methods and seeds (scripts/run_churn.sh)
	$(RUN) bash scripts/run_churn.sh $(ARGS)

report: ## Summarize the runs in RUNS_DIR into RESULTS_DIR
	$(RUN) python -m tabpack_repro summarize --runs-dir '$(RUNS_DIR)' \
		--output '$(RESULTS_DIR)' $(ARGS)

clean: ## Remove caches and build artifacts (keeps .venv, data, runs, results)
	rm -rf .pytest_cache .ruff_cache build dist
	find . \( -path ./.git -o -path ./.venv -o -path ./.worktrees -o -path ./.reference \) \
		-prune -o \( -name __pycache__ -o -name '*.egg-info' \) -type d -prune \
		-exec rm -rf {} +
