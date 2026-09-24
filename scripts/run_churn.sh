#!/usr/bin/env bash
# Run the whole Churn experiment (a35): a thin wrapper around scripts/run_churn.py.
#
#   bash scripts/run_churn.sh                      # everything (resumes finished runs)
#   bash scripts/run_churn.sh --dry-run            # print the plan only
#   bash scripts/run_churn.sh --seeds 0 --methods mlp --device cpu
#
# Runs from the repository root (so relative paths are relative to it) with `uv run`
# (override the binary with $UV), passes all arguments through, and appends the
# output to <runs-dir>/run.log (default runs/churn/run.log; $RUN_CHURN_LOG overrides
# it). --dry-run and --help are not logged. The exit status is the script's.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runs_dir="runs/churn"
log=1
prev=""
for arg in "$@"; do
  case "$arg" in
    --dry-run | -h | --help) log=0 ;;
    --runs-dir=*) runs_dir="${arg#--runs-dir=}" ;;
  esac
  if [[ "$prev" == "--runs-dir" ]]; then
    runs_dir="$arg"
  fi
  prev="$arg"
done

cmd=("${UV:-uv}" run python -u scripts/run_churn.py "$@")
if [[ "$log" == 0 ]]; then
  exec "${cmd[@]}"
fi

log_file="${RUN_CHURN_LOG:-$runs_dir/run.log}"
mkdir -p "$(dirname "$log_file")"
echo "==== $(date '+%Y-%m-%d %H:%M:%S') scripts/run_churn.sh $* (log: $log_file)" \
  | tee -a "$log_file"
"${cmd[@]}" 2>&1 | tee -a "$log_file"
