#!/usr/bin/env bash
# Integrator helper: merge one agent branch into the current branch with --no-ff.
#
#   tools/integrator/merge_branch.sh feat/a09-nn-linear "LinearPack with per-member init"
#
# Add/add conflicts on files the branch itself added (agents that had to force-add
# owned files before checkpoint/00b-data-fix) are resolved in favour of the branch.
# Any other conflict aborts the merge and exits 1 for manual resolution.
set -euo pipefail
branch="$1"; summary="$2"
agent="${branch#feat/}"
msg="Merge branch '$branch': $summary

Agent: $agent
Merged-by: integrator
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
if git merge --no-ff --no-edit -m "$msg" "$branch" >/dev/null 2>&1; then
  echo "merged $branch -> $(git rev-parse --short HEAD)"; exit 0
fi
conflicts=$(git diff --name-only --diff-filter=U)
for f in $conflicts; do
  status=$(git status --porcelain -- "$f" | cut -c1-2)
  if [[ "$status" == "AA" ]]; then
    git checkout --theirs -- "$f" && git add -- "$f"
    echo "  resolved add/add in favour of $branch: $f"
  else
    echo "  UNRESOLVED ($status): $f"; git merge --abort; exit 1
  fi
done
git commit --no-edit -q
echo "merged $branch -> $(git rev-parse --short HEAD) (with add/add resolutions)"
