# a52-evidence-tools: evidence scripts

## Summary

I wrote five scripts that the integrator runs at every checkpoint to fill `evidence/`,
plus a self-test and a usage README. They use only bash, git and the Python standard
library, and none of them commits, tags or changes a ref.

* `export_branch_diffs.sh` writes one diff per merged agent branch.
* `record_checkpoint.sh` writes the checkpoint record and the cumulative checkpoint diff.
* `archive_board.py` copies the coordination board, redacts it and renders it as
  markdown.
* `collect_reports.py` builds `evidence/agents/INDEX.md`.
* `snapshot_session.sh` copies the newest session transcript and redacts it.

Each branch diff contains only that agent's own changes, including for agents that
merged main or peer branches into their own branch. I checked this on the real
history: for all 31 merged branches, the changed files match what the branch's merge
commit brought into main. The one expected exception is a12, whose report reached
main through a21's merge.

## Files

* `tools/evidence/export_branch_diffs.sh`: `[--out DIR] [--head REV] [BRANCH...]` writes `evidence/diffs/<branch with / replaced by __>.diff`.
* `tools/evidence/record_checkpoint.sh`: `<NN> <name> [--notes FILE] [--no-tests]` writes `evidence/checkpoints/<NN>-<name>.md` and `evidence/diffs/checkpoint-<NN>-<name>.diff`.
* `tools/evidence/archive_board.py`: `[--board FILE] [--out DIR]` writes `evidence/coordination/board.jsonl` (the redacted copy) and `board.md`.
* `tools/evidence/collect_reports.py`: `[--agents-dir DIR] [--roster FILE] [--out FILE]` writes `evidence/agents/INDEX.md`.
* `tools/evidence/snapshot_session.sh`: `before|after [--from DIR | --file FILE] [--force]` writes `evidence/<which>/session.jsonl`.
* `tools/evidence/selftest.sh`: runs 69 checks in a throwaway `mktemp -d` repository.
* `tools/evidence/README.md`: usage, the checkpoint routine, and the rules for choosing diff bases.
* `evidence/agents/a52-evidence-tools.md`: this report.

## Design decisions

* **Diff base.** The spec is `git merge-base M^1 <branch>`, where M is the
  first-parent merge that brought the branch into HEAD. The first two cases follow
  that rule directly; the rest refine it.
  1. **Single merge base.** The script uses that merge base as specified.
  2. **Several merge bases.** This is the common case once an agent has merged main or
     peers that reached main first. The script diffs against the virtual base that
     `git merge` uses: it folds all the merge bases together with
     `git merge-tree --write-tree`, using fixed-identity `commit-tree` commits when
     there are more than two bases. Only the agent's own changes remain. Before this,
     a11 (merged a09 and a10) and a12 showed their peers' files in their diffs.
  3. **Merge bases that conflict with each other.** On the real history this happened
     for a07: it merged a04 to a06, whose force-added files hit add/add conflicts with
     `checkpoint/00b-data-fix`. Here the diff is `git diff M^1 <branch>`, limited to
     the paths the branch changed. This follows the coordinator's hint to diff against
     the first parent of the merge commit. a07's diff went from 12 files, mostly its
     peers', to its own 3 files.
  4. **Branch merged more than once.** The script finds the first merge through merge
     subjects that name the branch, as written by `tools/integrator/merge_branch.sh`,
     and diffs from before that merge. It checks that `M^2` is an ancestor of the
     branch, so peers that the agent fast-forwarded to are not taken for the branch.
  5. **Commits added after the last merge.** An example is a12's report, which was
     merged later. The diff ends at the newest merged commit. The later commits are
     listed in the header as "not merged yet", and the default branch set includes
     such branches. Otherwise a12 would have had no diff at checkpoint 01.
* **Idempotence.** Diff headers contain only SHAs and subjects, with no dates or
  decorations. All git diff options are pinned. Files are written through a temporary
  file and replaced only when their content changes.
* **Checkpoint order.** Numbers are compared as `(int, suffix)`, so 00 < 00b < 01. The
  previous checkpoint must be an ancestor of HEAD, and a tag with the same name is
  ignored, so a rerun after tagging still works. A failing test suite is recorded with
  its exit code and `FAILED`/`ERROR` lines and does not abort the script. The record
  and the notes go through the redactor. The diffs stay verbatim, so they still work
  with `git apply`.
* **Markdown escaping.** Board texts and report summaries are escaped for table cells:
  `|` becomes `\|`, and `<`, `>` and `&` become HTML entities outside code spans.
  Without this, text such as `feat/<id>` would disappear on GitHub.
* **Session transcripts.** The script only reads `*.jsonl` files directly in the
  project directory, so sub-agent transcripts in subdirectories are ignored. It hashes
  the copy rather than the live file, because the live file keeps growing. `before`
  needs `--force` to overwrite an existing snapshot.

## Tests

* `TMPDIR=<scratch> tools/evidence/selftest.sh` gives `selftest: all checks passed`
  (69 checks in about 9 s). It covers:
  * branches with 2 and 3 merge bases, conflicting merge bases, double merges,
    fast-forwarded peers, custom merge messages, partially merged branches, a skipped
    branch and an unknown branch;
  * idempotent reruns and the `--head` option;
  * a record of failing tests, `--no-tests`, the checkpoint order 00 < 00b < 01,
    redaction, and a check that no refs changed;
  * board counts, pairs, escaping and redaction;
  * the index with no `Summary` section, a numbered `Summary` section, and missing
    reports;
  * choosing the newest transcript, the `before` guard, and `--file`.
* On the real repository (read-only):
  * `export_branch_diffs.sh --head main --out <scratch>`, with objects redirected
    through `GIT_OBJECT_DIRECTORY` so that nothing was written to the real object
    store, produced 31 diffs. A second run left all of them unchanged.
  * `record_checkpoint.sh 02 dryrun --no-tests` in a `git clone --shared` copy found
    `checkpoint/01-foundations` as the previous checkpoint and listed 21 merges.
  * `archive_board.py --out <scratch>` rendered 130 messages from 49 senders.
  * `collect_reports.py --out <scratch>` ran; my branch contains no reports yet.
* `shellcheck tools/evidence/*.sh` is clean. `tools/dev/py -m ruff check tools/` and
  `ruff format --check tools/` pass. The anchored `evidence/*` exclude from 00b does
  not hide `tools/evidence/`.

## Coordination

* Posted `status` #42 when I started. I merged `checkpoint/00b-data-fix` as the
  integrator asked in #60.
* I followed the coordinator's note on merge subjects and trailers, and on agents that
  merged main or peers into their branches (see Design decisions).
* I merged no peer branches. I received no questions or findings.

## Open issues

* The commit message of `b8dfac4` says 66 self-test checks. There are 69.
* Building a virtual base writes a few unreferenced tree and commit objects into the
  repository. `git gc` removes them. No refs change.
