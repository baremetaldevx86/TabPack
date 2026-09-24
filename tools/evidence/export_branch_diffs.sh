#!/usr/bin/env bash
# Export one diff per agent branch into evidence/diffs/ (idempotent).
#
#   tools/evidence/export_branch_diffs.sh                     # all feat/* merged into HEAD
#   tools/evidence/export_branch_diffs.sh feat/a09-nn-linear  # only these branches
#   tools/evidence/export_branch_diffs.sh --out DIR --head main feat/a09-nn-linear
#
# Options: --out DIR (default: <repo>/evidence/diffs), --head REV (default: HEAD).
# Without branch arguments: every local feat/* branch merged into --head, including
# branches with newer commits that are not merged yet.
#
# <out>/<branch with '/' replaced by '__'>.diff holds a '# ' header (branch, tip, merge
# commit, base, `git log --oneline` of the branch's commits, diffstat) followed by
# `git diff <base> <branch>`, where <base> = `git merge-base M^1 <branch>` and M is the
# first-parent merge commit of --head that brought the branch in. Details:
#  * A branch merged more than once (merge subjects naming it) is diffed from before its
#    first merge, so the file holds the branch's whole contribution.
#  * The diff ends at the newest merged commit of the branch; newer commits are listed
#    in the header as not merged yet. A never-merged branch is diffed up to its tip
#    against its merge base with --head.
#  * If the merge base is not unique (typical when the agent merged main or peer
#    branches that reached main before it), <base> is the virtual base `git merge`
#    itself uses (the merge of all merge bases), so the peers' changes stay out of the
#    diff. If those merge bases conflict with each other (e.g. add/add of files that
#    agents force-added), the diff is taken against M^1 instead, limited to the paths
#    the branch changed, which again leaves out what main already had.
#  * Branches without commits of their own (tip on --head's first-parent history) are
#    skipped. Files whose content would not change are not rewritten.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

out=""
head_rev="HEAD"
branches=()
while (($#)); do
  case "$1" in
    --out | --head)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      if [[ $1 == --out ]]; then out="$2"; else head_rev="$2"; fi
      shift 2 ;;
    --out=*) out="${1#--out=}"; shift ;;
    --head=*) head_rev="${1#--head=}"; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; branches+=("$@"); break ;;
    -*) echo "error: unknown option $1 (see --help)" >&2; exit 2 ;;
    *) branches+=("$1"); shift ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel)"
out="${out:-$ROOT/evidence/diffs}"
mkdir -p "$out"
head_c="$(git rev-parse --verify "$head_rev^{commit}")"

# Pinned so that user configuration cannot change the output.
DIFF_OPTS=(--no-color --no-ext-diff --no-textconv --find-renames
  --src-prefix=a/ --dst-prefix=b/)

# First-parent history of --head, newest first, and each commit's index in it.
mapfile -t fp < <(git rev-list --first-parent "$head_c")
declare -A fp_index=()
for i in "${!fp[@]}"; do fp_index[${fp[$i]}]=$i; done

# First-parent merges of --head (newest first) and the words of their subjects, e.g.
# "Merge branch 'feat/a09-nn-linear': LinearPack" -> Merge branch feat/a09-nn-linear ...
mapfile -t fp_merges < <(git log --first-parent --merges --format='%H%x09%s' "$head_c")
words=()
subject_words() { read -ra words <<<"${1//[\'\"\`,:;()]/ }"; } # sets $words

if ((${#branches[@]} == 0)); then
  declare -A named=()
  for line in "${fp_merges[@]}"; do
    subject_words "${line#*$'\t'}"
    for w in "${words[@]}"; do named[$w]=1; done
  done
  while read -r b; do
    if [[ -n ${named[$b]+x} ]] || git merge-base --is-ancestor "$b" "$head_c"; then
      branches+=("$b")
    fi
  done < <(git for-each-ref --format='%(refname:lstrip=2)' 'refs/heads/feat/')
fi

# Print the oldest commit of the first-parent history that contains $1 (which must be
# an ancestor of --head). Containment is monotone along that history: binary search.
oldest_containing() {
  local c=$1 lo=0 hi=${#fp[@]} mid
  while ((hi - lo > 1)); do
    mid=$(((lo + hi) / 2))
    if git merge-base --is-ancestor "$c" "${fp[$mid]}"; then lo=$mid; else hi=$mid; fi
  done
  echo "${fp[$lo]}"
}

# Print (newest first) the first-parent merges whose subject names branch $1 and whose
# merged parent is an ancestor of the branch tip $2.
named_merges() {
  local branch=$1 tip=$2 line w
  for line in "${fp_merges[@]}"; do
    subject_words "${line#*$'\t'}"
    for w in "${words[@]}"; do
      if [[ $w == "$branch" ]] && git merge-base --is-ancestor "${line%%$'\t'*}^2" "$tip"
      then
        echo "${line%%$'\t'*}"
        break
      fi
    done
  done
}

# Print the tree to diff against when there are several merge bases: the merge of all
# of them, folded pairwise as git's recursive/ort strategy does. Fails on conflicts.
virtual_base() {
  local v=$1 out tree="" i
  local -a rest=("${@:2}")
  for i in "${!rest[@]}"; do
    out=$(git merge-tree --write-tree --no-messages "$v" "${rest[$i]}") || return 1
    tree=${out%%$'\n'*}
    if ((i < ${#rest[@]} - 1)); then
      v=$(GIT_AUTHOR_NAME=export_branch_diffs GIT_AUTHOR_EMAIL=virtual@localhost \
        GIT_AUTHOR_DATE='2000-01-01T00:00:00+0000' \
        GIT_COMMITTER_NAME=export_branch_diffs GIT_COMMITTER_EMAIL=virtual@localhost \
        GIT_COMMITTER_DATE='2000-01-01T00:00:00+0000' \
        git commit-tree "$tree" -p "$v" -p "${rest[$i]}" -m 'virtual merge base') ||
        return 1
    fi
  done
  echo "$tree"
}

oneline() { git log -1 --no-decorate --no-color --format='%h %s' "$1"; }

# Export one branch. Runs in a subshell with errexit on; exit status: 0 written,
# 10 unchanged, 11 skipped, anything else failed.
export_one() {
  local branch=$1 tip end merge_first="" merge_last="" before before_desc merged_by
  local base base_desc dest
  local -a merges bases paths pathspec=()
  if ! tip=$(git rev-parse --verify --quiet "$branch^{commit}"); then
    echo "error   $branch: not a branch or commit" >&2
    return 1
  fi
  if [[ -n ${fp_index[$tip]+x} ]]; then
    echo "skip    $branch (no commits of its own: tip is on the first-parent history)"
    return 11
  fi

  # end = newest merged commit of the branch (its tip if fully merged or never merged)
  mapfile -t merges < <(named_merges "$branch" "$tip")
  if git merge-base --is-ancestor "$tip" "$head_c"; then
    end=$tip
    merge_last=$(oldest_containing "$tip")
  elif ((${#merges[@]})); then
    merge_last=${merges[0]}
    end=$(git rev-parse "$merge_last^2")
  else
    end=$tip
  fi
  if ((${#merges[@]})); then merge_first=${merges[-1]}; else merge_first=$merge_last; fi

  if [[ -n $merge_first ]]; then
    before=$(git rev-parse "$merge_first^1")
    before_desc="$(git rev-parse --short "$merge_first")^1"
    merged_by=$(oneline "$merge_first")
    if [[ $merge_last != "$merge_first" ]]; then
      merged_by+=$'\n'"#               $(oneline "$merge_last") (brought in the tip)"
    fi
  else
    before=$head_c
    before_desc=$head_rev
    merged_by="(not merged into $head_rev)"
  fi

  mapfile -t bases < <(git merge-base --all "$before" "$end" || true)
  if ((${#bases[@]} == 0)); then
    echo "error   $branch: no merge base with $before_desc" >&2
    return 1
  elif ((${#bases[@]} == 1)); then
    base=${bases[0]}
    base_desc="$base = git merge-base $before_desc $(git rev-parse --short "$end")"
  elif base=$(virtual_base "${bases[@]}"); then
    local b short=()
    for b in "${bases[@]}"; do short+=("$(git rev-parse --short "$b")"); done
    base_desc="tree $base = virtual merge base (merge of the ${#bases[@]} merge bases"
    base_desc+=" of $before_desc and $(git rev-parse --short "$end"): ${short[*]})"
  else
    # The merge bases conflict: compare with main before the merge (M^1) instead, on
    # the paths the branch changed relative to any of its merge bases.
    mapfile -t paths < <(for b in "${bases[@]}"; do
      git diff --name-only --no-renames "$b" "$end"
    done | sort -u)
    base=$before
    if ((${#paths[@]})); then pathspec=(-- "${paths[@]}"); else base=$end; fi
    base_desc="$before = $before_desc, on the paths the branch changed (its"
    base_desc+=" ${#bases[@]} merge bases conflict, so there is no virtual base)"
  fi

  dest="$out/${branch//\//__}.diff"
  tmp="$dest.tmp.$$"
  trap 'rm -f "$tmp"' EXIT # global on purpose: this runs in its own subshell
  {
    echo "# Branch diff: $branch"
    echo "#"
    echo "# branch:       $branch"
    echo "# tip:          $(oneline "$tip")"
    if [[ $end != "$tip" ]]; then
      echo "# diffed up to: $(oneline "$end") (newest merged commit)"
    fi
    echo "# merged by:    $merged_by"
    echo "# base:         $base_desc"
    if ((${#pathspec[@]})); then
      echo "# command:      git diff $base $end -- <the paths in the diffstat>"
    else
      echo "# command:      git diff $base $end"
    fi
    echo "#"
    echo "# Commits (git log --oneline $before_desc..$(git rev-parse --short "$end")):"
    git log --oneline --no-decorate --no-color "$before..$end" | sed 's/^/#   /'
    if [[ $end != "$tip" ]]; then
      echo "#"
      echo "# Not merged into $head_rev yet (not in this diff):"
      git log --oneline --no-decorate --no-color "$end..$tip" | sed 's/^/#   /'
    fi
    echo "#"
    echo "# Diffstat:"
    git diff --stat=100 "${DIFF_OPTS[@]}" "$base" "$end" "${pathspec[@]}" |
      sed 's/^/#  /'
    echo "#"
    git diff "${DIFF_OPTS[@]}" "$base" "$end" "${pathspec[@]}"
  } >"$tmp"
  if [[ -f $dest ]] && cmp -s "$tmp" "$dest"; then
    rm -f "$tmp"
    echo "same    $dest"
    return 10
  fi
  mv "$tmp" "$dest"
  echo "wrote   $dest"
}

n_wrote=0 n_same=0 n_skip=0 n_fail=0
for b in "${branches[@]}"; do
  set +e
  (set -e; export_one "$b")
  rc=$?
  set -e
  case $rc in
    0) n_wrote=$((n_wrote + 1)) ;;
    10) n_same=$((n_same + 1)) ;;
    11) n_skip=$((n_skip + 1)) ;;
    1) n_fail=$((n_fail + 1)) ;;
    *) n_fail=$((n_fail + 1)); echo "error   $b: failed (exit $rc)" >&2 ;;
  esac
done
echo "export_branch_diffs: $n_wrote written, $n_same unchanged, $n_skip skipped," \
  "$n_fail failed -> $out"
((n_fail == 0))
