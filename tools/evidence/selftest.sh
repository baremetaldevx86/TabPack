#!/usr/bin/env bash
# shellcheck disable=SC2016 # backticks in the patterns are literal markdown
# Self-test for the evidence tools: builds a throwaway git repository in $(mktemp -d)
# with fake agent branches, merges, checkpoint tags, board messages, agent reports and
# session transcripts, runs every tool against it and checks the results. It never
# touches the repository it lives in.
#
#   tools/evidence/selftest.sh [--keep]     # --keep: leave the scratch directory
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_PY="$TOOLS/../dev/board.py"
keep=0
[[ ${1:-} == --keep ]] && keep=1

scratch="$(mktemp -d)"
if ((keep)); then
  trap 'echo "scratch kept: $scratch"' EXIT
else
  trap 'rm -rf "$scratch"' EXIT
fi
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 TZ=UTC
unset TABPACK_BOARD TABPACK_SESSION_DIR

fails=0
check() { # check <description> <command...>: the command must succeed
  local desc=$1
  shift
  if "$@" >/dev/null 2>&1; then
    echo "ok   - $desc"
  else
    echo "FAIL - $desc"
    fails=$((fails + 1))
  fi
}
not() { ! "$@"; }
# Files touched by a diff file, sorted and space-separated.
touched() { sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "$1" | sort | tr '\n' ' '; }
g() { git "$@" >/dev/null 2>&1; }
commit_file() { # commit_file <file> <content> <message>
  echo "$2" >>"$1"
  g add "$1"
  g commit -m "$3"
}
merge_no_ff() { g merge --no-ff -m "Merge branch '$1': $2" "$1"; }

# --- scratch repository ----------------------------------------------------------------
repo="$scratch/repo"
mkdir -p "$repo/tools/dev"
cd "$repo"
g init -b main
g config user.name 'Self Test'
g config user.email selftest@example.com
g config commit.gpgsign false
cp "$BOARD_PY" tools/dev/board.py
cat >tools/dev/py <<'EOF'
#!/usr/bin/env bash
# fake slot runner: pretends to run pytest
echo "..F."
echo "FAILED tests/test_x.py::test_bad - assert 0"
echo "1 failed, 3 passed, 2 deselected in 0.05s"
exit 1
EOF
chmod +x tools/dev/py
echo '.coord/' >.gitignore
commit_file README 'skeleton' 'chore: skeleton'
g tag -a checkpoint/00-skeleton -m 'checkpoint 00'

# a, b: plain branches; c: merged a and b before its own merge (2 merge bases);
# d: no commits; e: never merged; f: merged twice; g: fast-forwarded to a, then own
# work; h: merged a, b and f (3 merge bases); i: merged with a custom message;
# j: merged, then one more commit; k: merged p and a stub commit of main that
# conflicts with p (add/add), so its merge bases conflict.
g switch -c feat/a main && commit_file a.txt a1 'a: one' && commit_file a.txt a2 'a: two'
g switch -c feat/b main && commit_file b.txt b 'b: one'
g switch -c feat/c main && commit_file c.txt c 'c: own'
g merge --no-edit feat/a && g merge --no-edit feat/b && commit_file c.txt c2 'c: after'
g switch -c feat/d main
g switch -c feat/e main && commit_file e.txt e 'e: unmerged'
g switch -c feat/f main && commit_file f.txt f1 'f: one'
g switch -c feat/g main && g merge --no-edit feat/a && commit_file g.txt g 'g: own'
g switch -c feat/p main && commit_file shared.txt real 'p: add shared.txt'
g switch main
merge_no_ff feat/a 'thing a' && merge_no_ff feat/b 'thing b' && merge_no_ff feat/c 'c'
merge_no_ff feat/f 'first'
commit_file main.txt m 'main: direct commit'
g switch feat/f && commit_file f.txt f2 'f: two' && g switch main
merge_no_ff feat/f 'second' && merge_no_ff feat/g 'g'
g switch -c feat/h feat/e~1 && commit_file h.txt h 'h: own'
g merge --no-edit feat/a && g merge --no-edit feat/b && g merge --no-edit feat/f
g switch -c feat/i main && commit_file i.txt i 'i: own'
g switch -c feat/j main && commit_file j.txt j1 'j: one'
g switch main
merge_no_ff feat/h 'h'
g merge --no-ff -m 'custom merge message' feat/i
merge_no_ff feat/j 'j'
g switch feat/j && commit_file j.txt j2 'j: two (not merged)' && g switch main
g tag -a checkpoint/00b-fix -m 'checkpoint 00b'
commit_file shared.txt stub 'fix: add shared.txt stub'
g switch -c feat/k feat/p
g merge --no-edit main || { g checkout --ours shared.txt && g add shared.txt &&
  g commit --no-edit; }
commit_file k.txt k 'k: own'
g switch main
merge_no_ff feat/p 'p' || { g checkout --theirs shared.txt && g add shared.txt &&
  g commit --no-edit; }
merge_no_ff feat/k 'k'
refs_before="$(git for-each-ref --format='%(refname) %(objectname)')"

# --- export_branch_diffs.sh ----------------------------------------------------------
echo "# export_branch_diffs.sh"
d="$scratch/diffs"
"$TOOLS/export_branch_diffs.sh" --out "$d" >"$scratch/export1.log" 2>&1
for b in a b c f g h i j k p; do
  check "feat/$b exported" test -f "$d/feat__$b.diff"
done
check "feat/d (no own commits) skipped" grep -q 'skip    feat/d' "$scratch/export1.log"
check "feat/e (never merged) not exported by default" not test -e "$d/feat__e.diff"
check "a: own changes" test "$(touched "$d/feat__a.diff")" = 'a.txt '
check "c: 2 merge bases -> virtual base, own file only" \
  test "$(touched "$d/feat__c.diff")" = 'c.txt '
check "c: header names the virtual base" grep -q 'virtual merge base' "$d/feat__c.diff"
check "f: merged twice -> both commits" grep -q '^+f2' "$d/feat__f.diff"
check "f: diffed from before the first merge" grep -q '^+f1' "$d/feat__f.diff"
check "g: fast-forwarded peer left out" test "$(touched "$d/feat__g.diff")" = 'g.txt '
check "h: 3 merge bases -> own file only" test "$(touched "$d/feat__h.diff")" = 'h.txt '
check "i: custom merge message still found" grep -q 'custom merge message' \
  "$d/feat__i.diff"
check "j: diff ends at the merged commit" not grep -q '^+j2' "$d/feat__j.diff"
check "j: unmerged commit listed" grep -q 'j: two (not merged)' "$d/feat__j.diff"
check "k: conflicting merge bases -> own file only" \
  test "$(touched "$d/feat__k.diff")" = 'k.txt '
check "k: header names the fallback" grep -q 'merge bases conflict' "$d/feat__k.diff"
check "a: diff body equals git diff base..tip" \
  cmp <(sed -n '/^diff --git/,$p' "$d/feat__a.diff") \
  <(git diff --no-color "$(git merge-base 'main^{/feat.a.: thing a}^1' feat/a)" feat/a)
"$TOOLS/export_branch_diffs.sh" --out "$d" >"$scratch/export2.log" 2>&1
check "rerun: nothing rewritten" grep -q ' 0 written, 10 unchanged' "$scratch/export2.log"
"$TOOLS/export_branch_diffs.sh" --out "$d" feat/e >/dev/null 2>&1
check "explicit never-merged branch exported" grep -q 'not merged into HEAD' \
  "$d/feat__e.diff"
check "unknown branch fails" not "$TOOLS/export_branch_diffs.sh" --out "$d" feat/nope
check "--head selects another head" "$TOOLS/export_branch_diffs.sh" --head 'main~1' \
  --out "$scratch/diffs-head" feat/j

# --- record_checkpoint.sh ------------------------------------------------------------
echo "# record_checkpoint.sh"
tools/dev/board.py post --from a1 --kind status 'started; mail me@private.org' >/dev/null
printf '## Summary\n\nWave one. Contact: bob@corp.example\n' >"$scratch/notes.md"
"$TOOLS/record_checkpoint.sh" 01 wave1 --notes "$scratch/notes.md" \
  >"$scratch/rc1.log" 2>&1
md=evidence/checkpoints/01-wave1.md
check "record written" test -f "$md"
check "previous checkpoint is 00b" grep -q 'Previous checkpoint | `checkpoint/00b-fix`' "$md"
check "merged branches table" grep -q '| `feat/k` | 2 |' "$md"
check "test summary and exit code" grep -q '1 failed, 3 passed, 2 deselected' "$md"
check "failed tests listed" grep -q '^FAILED tests/test_x.py::test_bad' "$md"
check "exit code recorded" grep -q 'exit code: 1' "$md"
check "board status included" grep -q '#1 .* a1 (status)' "$md"
check "notes included" grep -q '^Wave one' "$md"
check "emails redacted" not grep -Eq 'bob@corp|me@private' "$md"
check "cumulative diff written" test -f evidence/diffs/checkpoint-01-wave1.diff
check "cumulative diff since 00b" grep -q '^# from:    checkpoint/00b-fix' \
  evidence/diffs/checkpoint-01-wave1.diff
"$TOOLS/record_checkpoint.sh" 00b fix --no-tests >"$scratch/rc2.log" 2>&1
check "--no-tests" grep -q 'Not run (--no-tests)' evidence/checkpoints/00b-fix.md
check "00b follows 00" grep -q '`checkpoint/00-skeleton`' evidence/checkpoints/00b-fix.md
check "existing tag reported" grep -q 'already exists' evidence/checkpoints/00b-fix.md
"$TOOLS/record_checkpoint.sh" 00b fix --no-tests >"$scratch/rc3.log" 2>&1
check "rerun leaves the diff unchanged" grep -q 'fix.diff unchanged' "$scratch/rc3.log"
check "bad arguments rejected" not "$TOOLS/record_checkpoint.sh" 1x2 bad/name
check "no commits or tags created" \
  test "$(git for-each-ref --format='%(refname) %(objectname)')" = "$refs_before"

# --- archive_board.py ----------------------------------------------------------------
echo "# archive_board.py"
tools/dev/board.py post --from a2 --to a1 --kind question \
  'Is `w|x` (K,<in>,out)? joe@x.io' >/dev/null
tools/dev/board.py post --from a1 --to a2 --kind answer 'yes' >/dev/null
tools/dev/board.py post --from integrator --to a1,a2 --kind contract 'both' >/dev/null
"$TOOLS/archive_board.py" >"$scratch/ab1.log"
c=evidence/coordination
check "board copied" test "$(wc -l <"$c/board.jsonl")" -eq 4
check "copy redacted" not grep -Eq 'joe@x.io|me@private.org' "$c/board.jsonl"
check "copy otherwise identical" cmp <(sed 's/[a-z.]*@[a-z.]*\.[a-z]*/[redacted-email]/g' \
  .coord/board.jsonl) "$c/board.jsonl"
check "per-agent counts" grep -q '^| a1 | 2 | 1 | 2 | status 1, answer 1 |$' "$c/board.md"
check "who talked to whom" grep -q '^| integrator | a2 | 1 | contract 1 |$' "$c/board.md"
check "pipes escaped" grep -Fq 'Is `w\|x` (K,&lt;in&gt;,out)?' "$c/board.md"
"$TOOLS/archive_board.py" >"$scratch/ab2.log"
check "rerun: unchanged" test "$(grep -c unchanged "$scratch/ab2.log")" -eq 2
check "missing board fails" not "$TOOLS/archive_board.py" --board "$scratch/none.jsonl"

# --- collect_reports.py --------------------------------------------------------------
echo "# collect_reports.py"
mkdir -p evidence/agents
cat >"$c/roster.md" <<'EOF'
| Id | Area | Owned paths | Deps |
| :-- | :-- | :-- | :-- |
| a1-first | NN | `src/a.py` | none |
| a2-second | Data \| IO | `src/b.py` | a1 |
| a3-silent | Docs | `docs/x.md` | none |
EOF
cat >evidence/agents/a1-first.md <<'EOF'
# a1-first report

## Summary

Implemented `f(x|y)` and
tests <fast>.

Second paragraph.

## Files
EOF
printf '# a2-second\n\nNo summary heading here.\n' >evidence/agents/a2-second.md
printf '# Integrator\n\n## 1. Summary\n- merged everything\n' >evidence/agents/integrator.md
"$TOOLS/collect_reports.py" >"$scratch/cr1.log"
i=evidence/agents/INDEX.md
check "index written" test -f "$i"
check "row with area and first paragraph" grep -Fq \
  '| a1-first | NN | Implemented `f(x\|y)` and tests &lt;fast&gt;. | [a1-first.md](a1-first.md) |' "$i"
check "missing Summary flagged" grep -q '_(no Summary section)_ No summary heading' "$i"
check "escaped pipe in roster" grep -Fq '| a2-second | Data \| IO |' "$i"
check "numbered Summary heading, list item" grep -q '| integrator | - | merged everything |' "$i"
check "missing reports listed" grep -q '`a3-silent`' "$i"
"$TOOLS/collect_reports.py" >"$scratch/cr2.log"
check "rerun: unchanged" grep -q unchanged "$scratch/cr2.log"

# --- snapshot_session.sh -------------------------------------------------------------
echo "# snapshot_session.sh"
proj="$scratch/home/.claude/projects/${repo//[^A-Za-z0-9]/-}"
mkdir -p "$proj/sub"
echo '{"m":"old"}' >"$proj/old.jsonl"
touch -d '2001-01-01' "$proj/old.jsonl"
echo '{"m":"new from x@y.org and git@github.com"}' >"$proj/new.jsonl"
echo '{"m":"nested, ignored"}' >"$proj/sub/newest.jsonl"
HOME="$scratch/home" "$TOOLS/snapshot_session.sh" after >"$scratch/ss1.log"
check "newest top-level transcript copied" grep -q 'new from' evidence/after/session.jsonl
check "transcript redacted" grep -q '\[redacted-email\] and git@github.com' \
  evidence/after/session.jsonl
check "sha256 row printed" grep -Eq '^\| `after/session.jsonl` \| `[0-9a-f]{64}` \|' \
  "$scratch/ss1.log"
HOME="$scratch/home" "$TOOLS/snapshot_session.sh" before >/dev/null
check "before refuses to overwrite" not env HOME="$scratch/home" \
  "$TOOLS/snapshot_session.sh" before
check "before --force overwrites" env HOME="$scratch/home" \
  "$TOOLS/snapshot_session.sh" before --force
check "--file picks a transcript" "$TOOLS/snapshot_session.sh" after \
  --file "$proj/old.jsonl"
check "--file content" grep -q old evidence/after/session.jsonl
check "missing project dir fails" not env HOME="$scratch/nowhere" \
  "$TOOLS/snapshot_session.sh" after

echo
if ((fails)); then
  echo "selftest: $fails check(s) FAILED"
  exit 1
fi
echo "selftest: all checks passed"
