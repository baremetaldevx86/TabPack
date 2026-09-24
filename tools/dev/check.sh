#!/usr/bin/env bash
# Is this checkout healthy? Lint, format check and the fast tests in one command.
#
#   tools/dev/check.sh                                   # everything
#   tools/dev/check.sh --fast                            # ruff only, no pytest
#   tools/dev/check.sh --paths src/tabpack_repro/nn tests/nn
#
# Checks the checkout this script lives in (the main checkout or a worktree), whatever
# the current directory. Every tool runs through tools/dev/py (shared venv, concurrency
# slots, CUDA hidden). All steps run even after a failure; the exit status is 0 only
# when every step passed.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/dev/check.sh [--fast] [--paths PATH...] [-h|--help]

Steps (each through tools/dev/py, from the checkout root):
  ruff check           ruff check [targets]
  ruff format --check  ruff format --check [targets]
  pytest               pytest -q -m "not slow and not gpu" [test targets]

Options:
  --fast            Skip pytest.
  --paths PATH...   Check only these files/directories. Relative paths are taken from
                    the current directory, or else from the checkout root. Ruff gets the
                    directories and the files it understands (*.py, *.pyi, *.ipynb, *.md,
                    pyproject.toml); pytest gets the directories and test_*.py files
                    under tests/ (it is skipped when there are none). Explicit paths are
                    checked even when pyproject.toml excludes them.
  -h, --help        Show this help.

Exit status: 0 if every step passed or was skipped, 1 if a step failed, 2 on a usage
error.
EOF
}

die_usage() {
  printf 'check.sh: %s\n' "$1" >&2
  printf 'Try: tools/dev/check.sh --help\n' >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROOT="$(cd -- "$ROOT" && pwd -P)"
PY=tools/dev/py # relative: the steps run from $ROOT
PYTEST_MARKERS='not slow and not gpu'

# --- arguments ---------------------------------------------------------------------
fast=0
raw_paths=()
while (($# > 0)); do
  case "$1" in
    --fast)
      fast=1
      shift
      ;;
    --paths)
      shift
      n_before=${#raw_paths[@]}
      while (($# > 0)) && [[ "$1" != -* ]]; do
        raw_paths+=("$1")
        shift
      done
      ((${#raw_paths[@]} > n_before)) || die_usage '--paths needs at least one path'
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die_usage "unknown argument: $1"
      ;;
  esac
done

# Print PATH relative to ROOT ("." for ROOT itself), or fail if it does not exist
# inside this checkout.
resolve_path() {
  local p="$1" candidate abs
  local candidates=("$p")
  [[ "$p" == /* ]] || candidates+=("$ROOT/$p")
  for candidate in "${candidates[@]}"; do
    abs="$(realpath -e -- "$candidate" 2>/dev/null)" || continue
    if [[ "$abs" == "$ROOT" ]]; then
      printf '.\n'
      return 0
    elif [[ "$abs" == "$ROOT"/* ]]; then
      printf '%s\n' "${abs#"$ROOT"/}"
      return 0
    fi
  done
  return 1
}

printf 'check.sh: checking %s\n' "$ROOT"

# --- targets -------------------------------------------------------------------------
# With no --paths, ruff and pytest use their configured defaults (the whole checkout).
restricted=0
ruff_targets=()
pytest_targets=()
pytest_all=0
if ((${#raw_paths[@]} > 0)); then
  restricted=1
  for p in "${raw_paths[@]}"; do
    rel="$(resolve_path "$p")" || die_usage "not a path inside $ROOT: $p"
    if [[ "$rel" == . ]]; then
      ruff_targets+=(.)
      pytest_all=1
      continue
    fi
    used=0
    if [[ -d "$ROOT/$rel" ]]; then
      ruff_targets+=("$rel")
      used=1
      if [[ "$rel" == tests || "$rel" == tests/* ]]; then
        pytest_targets+=("$rel")
      fi
    else
      case "$rel" in
        *.py | *.pyi | *.ipynb | *.md | pyproject.toml | */pyproject.toml)
          ruff_targets+=("$rel")
          used=1
          ;;
        *) ;;
      esac
      case "$rel" in
        tests/*/test_*.py | tests/test_*.py | tests/*_test.py)
          pytest_targets+=("$rel")
          used=1
          ;;
        *) ;;
      esac
    fi
    ((used)) || printf 'check.sh: note: %s is not a ruff or pytest target; ignored\n' "$rel"
  done
fi

# --- steps ---------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  c_pass=$'\e[32m' c_fail=$'\e[31m' c_skip=$'\e[33m' c_bold=$'\e[1m' c_off=$'\e[0m'
else
  c_pass='' c_fail='' c_skip='' c_bold='' c_off=''
fi

summary=()
failed=()

record() { # record STATUS NAME DETAIL
  local color
  case "$1" in
    PASS) color="$c_pass" ;;
    FAIL) color="$c_fail" ;;
    *) color="$c_skip" ;;
  esac
  summary+=("$(printf '%s%-4s%s  %-20s %s' "$color" "$1" "$c_off" "$2" "$3")")
}

# run_step NAME COMMAND... ; pytest's "no tests collected" (exit 5) counts as SKIP.
run_step() {
  local name="$1" start rc=0
  shift
  printf '\n%s==> %s%s\n$' "$c_bold" "$name" "$c_off"
  printf ' %q' "$@"
  printf '\n'
  start=$SECONDS
  "$@" || rc=$?
  local took="($((SECONDS - start))s)"
  if ((rc == 0)); then
    record PASS "$name" "$took"
  elif [[ "$name" == pytest ]] && ((rc == 5)); then
    record SKIP "$name" "no tests collected $took"
  else
    record FAIL "$name" "exit $rc $took"
    failed+=("$name")
  fi
}

cd "$ROOT"

if ((restricted && ${#ruff_targets[@]} == 0)); then
  record SKIP 'ruff check' 'no ruff targets in --paths'
  record SKIP 'ruff format --check' 'no ruff targets in --paths'
else
  run_step 'ruff check' "$PY" -m ruff check "${ruff_targets[@]}"
  run_step 'ruff format --check' "$PY" -m ruff format --check "${ruff_targets[@]}"
fi

if ((fast)); then
  record SKIP pytest '--fast'
elif ((restricted && !pytest_all && ${#pytest_targets[@]} == 0)); then
  record SKIP pytest 'no test paths in --paths'
else
  if ((pytest_all)); then
    pytest_targets=()
  fi
  run_step pytest "$PY" -m pytest -q -m "$PYTEST_MARKERS" "${pytest_targets[@]}"
fi

# --- summary -------------------------------------------------------------------------
printf '\n%s==== check.sh summary: %s ====%s\n' "$c_bold" "$ROOT" "$c_off"
printf '%s\n' "${summary[@]}"
if ((${#failed[@]} > 0)); then
  printf -v failed_list '%s, ' "${failed[@]}"
  printf '%sRESULT: FAIL%s (%d failed: %s)\n' "$c_fail" "$c_off" "${#failed[@]}" \
    "${failed_list%, }"
  exit 1
fi
printf '%sRESULT: PASS%s\n' "$c_pass" "$c_off"
