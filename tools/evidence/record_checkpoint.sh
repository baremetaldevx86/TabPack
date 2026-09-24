#!/usr/bin/env bash
# Record a checkpoint of the current HEAD (run from the checkout being checkpointed).
#
#   tools/evidence/record_checkpoint.sh 02 wave2 [--notes FILE] [--no-tests]
#
# Writes evidence/checkpoints/<NN>-<name>.md: date, HEAD, the previous checkpoint (the
# checkpoint/* tag with the largest number below NN that is an ancestor of HEAD; e.g.
# 00 < 00b < 01), `git log --oneline --graph` since then, the branches merged since
# then, the diffstat, the fast test suite (tools/dev/py -m pytest -q -m "not slow and
# not gpu": summary line + exit code; --no-tests skips it), `tools/dev/board.py status`
# and the notes file, if given (markdown, included verbatim below the header table).
# Personal email addresses are redacted with tools/evidence/redact_emails.py.
# Also writes the cumulative diff since the previous checkpoint to
# evidence/diffs/checkpoint-<NN>-<name>.diff (not rewritten when unchanged).
# It never commits or tags: review the files, then commit and tag yourself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

nn="" name="" notes="" run_tests=1
while (($#)); do
  case "$1" in
    --notes)
      [[ $# -ge 2 ]] || { echo "error: --notes needs a file" >&2; exit 2; }
      notes="$2"; shift 2 ;;
    --notes=*) notes="${1#--notes=}"; shift ;;
    --no-tests) run_tests=0; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) echo "error: unknown option $1 (see --help)" >&2; exit 2 ;;
    *)
      if [[ -z $nn ]]; then nn="$1"
      elif [[ -z $name ]]; then name="$1"
      else echo "error: unexpected argument $1" >&2; exit 2
      fi
      shift ;;
  esac
done
if [[ ! $nn =~ ^[0-9]+[a-z]*$ || ! $name =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: record_checkpoint.sh <NN> <name> [--notes FILE] [--no-tests]" >&2
  echo "  NN: digits with an optional letter suffix (01, 02, 00b); name: [A-Za-z0-9._-]+" >&2
  exit 2
fi
if [[ -n $notes && ! -r $notes ]]; then
  echo "error: notes file not readable: $notes" >&2
  exit 2
fi
[[ -z $notes ]] || notes="$(cd "$(dirname "$notes")" && pwd)/$(basename "$notes")"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
id="$nn-$name"
tag="checkpoint/$id"
md="evidence/checkpoints/$id.md"
diff_file="evidence/diffs/checkpoint-$id.diff"
mkdir -p evidence/checkpoints evidence/diffs

# Sort key of a checkpoint number: 00 < 00b < 01 < 02 < 10.
order_key() {
  [[ $1 =~ ^([0-9]+)([a-z]*)$ ]]
  printf '%09d%s' "$((10#${BASH_REMATCH[1]}))" "${BASH_REMATCH[2]}"
}

key="$(order_key "$nn")"
prev_tag="" prev_key=""
while IFS= read -r t; do
  [[ $t != "$tag" && $t =~ ^checkpoint/([0-9]+[a-z]*)- ]] || continue
  k="$(order_key "${BASH_REMATCH[1]}")"
  [[ $k < $key && ($prev_key == "" || $k > $prev_key) ]] || continue
  git merge-base --is-ancestor "$t^{commit}" HEAD || continue
  prev_tag=$t prev_key=$k
done < <(git tag -l 'checkpoint/*')

head_c="$(git rev-parse HEAD)"
head_short="$(git rev-parse --short HEAD)"
empty_tree="$(git hash-object -t tree /dev/null)"
if [[ -n $prev_tag ]]; then
  prev_c="$(git rev-parse "$prev_tag^{commit}")"
  from=$prev_c
  since="\`$prev_tag\` (\`$(git rev-parse --short "$prev_c")\`)"
  log_range=("$prev_c..HEAD")
else
  prev_c=""
  from=$empty_tree
  since="the root commit (no earlier checkpoint tag)"
  log_range=(HEAD)
fi
DIFF_OPTS=(--no-color --no-ext-diff --no-textconv --find-renames
  --src-prefix=a/ --dst-prefix=b/)
GIT_LOG=(git log --no-decorate --no-color)

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

# --- tests -------------------------------------------------------------------------
test_cmd=(tools/dev/py -m pytest -q -m "not slow and not gpu")
test_cmd_str='tools/dev/py -m pytest -q -m "not slow and not gpu"'
test_rc="not run" test_summary="" test_failures=""
if ((run_tests)); then
  if [[ -x tools/dev/py ]]; then
    echo "record_checkpoint: running $test_cmd_str ..." >&2
    set +e
    "${test_cmd[@]}" 2>&1 | tee "$tmp_dir/pytest.log" >&2
    test_rc=${PIPESTATUS[0]}
    set -e
    test_summary="$(grep -E ' in [0-9.]+s' "$tmp_dir/pytest.log" | tail -n 1 || true)"
    if [[ -z $test_summary ]]; then
      test_summary="$(grep -v '^[[:space:]]*$' "$tmp_dir/pytest.log" | tail -n 1 || true)"
    fi
    test_summary="$(sed -E 's/^=+ //; s/ =+$//' <<<"$test_summary")"
    test_failures="$(grep -E '^(FAILED|ERROR) ' "$tmp_dir/pytest.log" | head -n 50 || true)"
  else
    test_summary="tools/dev/py not found: tests not run"
  fi
fi

# --- board -------------------------------------------------------------------------
if [[ -x tools/dev/board.py ]]; then
  board="$(tools/dev/board.py status 2>&1)" || board="(board.py status failed) $board"
  [[ -n $board ]] || board="(no status messages on the board)"
else
  board="(tools/dev/board.py not found)"
fi

# --- merged branches -----------------------------------------------------------------
merged_rows=""
while IFS=$'\t' read -r m subj; do
  [[ -n $m ]] || continue
  if [[ $subj =~ ^Merge\ branch\ \'([^\']+)\' ]]; then
    branch="${BASH_REMATCH[1]}"
  else
    branch="$(git for-each-ref --points-at "$m^2" --format='%(refname:lstrip=2)' \
      'refs/heads/' | paste -sd, -)"
  fi
  n_commits=$(($(git rev-list --count "$m^1..$m") - 1))
  subj="${subj//|/\\|}"
  merged_rows+="| \`$(git rev-parse --short "$m")\` | \`${branch:--}\` | $n_commits | $subj |"
  merged_rows+=$'\n'
done < <("${GIT_LOG[@]}" --first-parent --merges --reverse --format='%H%x09%s' \
  "${log_range[@]}")

# --- record --------------------------------------------------------------------------
{
  echo "# Checkpoint $nn: $name"
  echo
  echo "Date: $(date -Iseconds)"
  echo
  echo "| Item | Value |"
  echo "| :-- | :-- |"
  echo "| HEAD | \`$head_short\` $(git log -1 --format=%s HEAD | sed 's/|/\\|/g') |"
  echo "| Previous checkpoint | $since |"
  if tag_c="$(git rev-parse -q --verify "refs/tags/$tag^{commit}")"; then
    echo "| Tag | \`$tag\` (already exists: \`$(git rev-parse --short "$tag_c")\`) |"
  else
    echo "| Tag to create | \`$tag\` (not created by this script) |"
  fi
  echo "| Cumulative diff | [\`$diff_file\`](../diffs/checkpoint-$id.diff) |"
  if [[ -n $notes ]]; then
    echo
    cat "$notes"
  fi
  echo
  echo "## History since the previous checkpoint"
  echo
  echo '```'
  "${GIT_LOG[@]}" --oneline --graph "${log_range[@]}"
  echo '```'
  echo
  echo "## Branches merged since the previous checkpoint"
  echo
  if [[ -n $merged_rows ]]; then
    echo "| Merge | Branch | Commits | Subject |"
    echo "| :-- | :-- | --: | :-- |"
    printf '%s' "$merged_rows"
  else
    echo "No merge commits on the first-parent history since the previous checkpoint."
  fi
  echo
  echo "## Diffstat"
  echo
  echo '```'
  git diff --stat=100 "${DIFF_OPTS[@]}" "$from" HEAD
  echo '```'
  echo
  echo "## Tests"
  echo
  if ((run_tests)); then
    echo "Command: \`$test_cmd_str\`, exit code: $test_rc"
    echo
    echo '```'
    echo "$test_summary"
    [[ -z $test_failures ]] || echo "$test_failures"
    echo '```'
  else
    echo "Not run (--no-tests)."
  fi
  echo
  echo "## Board status"
  echo
  echo "Latest status/done/blocker message per agent (\`tools/dev/board.py status\`)."
  echo
  echo '```'
  echo "$board"
  echo '```'
} >"$tmp_dir/record.md"
mv "$tmp_dir/record.md" "$md"
python3 "$SCRIPT_DIR/redact_emails.py" "$md"

# --- cumulative diff -----------------------------------------------------------------
{
  echo "# Checkpoint diff: $id"
  echo "#"
  echo "# from:    ${prev_tag:-empty tree} ${prev_c:-$empty_tree}"
  echo "# to:      $(git log -1 --no-decorate --format='%h %s' HEAD)"
  echo "# command: git diff $from $head_c"
  echo "#"
  echo "# Diffstat:"
  git diff --stat=100 "${DIFF_OPTS[@]}" "$from" HEAD | sed 's/^/#  /'
  echo "#"
  git diff "${DIFF_OPTS[@]}" "$from" HEAD
} >"$tmp_dir/checkpoint.diff"
if [[ -f $diff_file ]] && cmp -s "$tmp_dir/checkpoint.diff" "$diff_file"; then
  diff_status="unchanged"
else
  mv "$tmp_dir/checkpoint.diff" "$diff_file"
  diff_status="written"
fi

if ((run_tests)); then
  echo "record_checkpoint: wrote $md (tests: exit $test_rc, $test_summary)"
else
  echo "record_checkpoint: wrote $md (tests not run)"
fi
echo "record_checkpoint: $diff_file $diff_status (since ${prev_tag:-the root})"
echo "Review, then commit and tag, e.g.:"
echo "  git add $md $diff_file"
echo "  git commit -m 'chore(evidence): record checkpoint $nn ($name)'"
echo "  git tag -a $tag -m 'checkpoint $nn: $name'"
