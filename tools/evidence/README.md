# Evidence tools

Scripts that write the provenance record in `evidence/` (see `evidence/README.md`).
They use only bash, git and the Python standard library. Run them from the checkout
whose evidence you are recording, normally the main checkout. None of them commits,
tags or changes refs. Rerunning a script with unchanged inputs leaves its outputs
byte-identical, except the checkpoint record, which has a date.

| Script | Writes |
| :-- | :-- |
| `export_branch_diffs.sh [--out DIR] [--head REV] [BRANCH...]` | `evidence/diffs/<branch, / -> __>.diff` for each merged `feat/*` branch |
| `record_checkpoint.sh <NN> <name> [--notes FILE] [--no-tests]` | `evidence/checkpoints/<NN>-<name>.md` and `evidence/diffs/checkpoint-<NN>-<name>.diff` |
| `archive_board.py [--board FILE] [--out DIR]` | `evidence/coordination/board.jsonl` (redacted copy) and `board.md` |
| `collect_reports.py [--agents-dir DIR] [--roster FILE] [--out FILE]` | `evidence/agents/INDEX.md` |
| `snapshot_session.sh before\|after [--from DIR \| --file FILE] [--force]` | `evidence/<before\|after>/session.jsonl` (redacted) |
| `redact_emails.py FILE...` | redacts personal email addresses in place (used by the others) |
| `selftest.sh [--keep]` | nothing: checks all of the above in a throwaway repository |

## At a checkpoint

```bash
tools/evidence/export_branch_diffs.sh
tools/evidence/archive_board.py
tools/evidence/collect_reports.py
tools/evidence/record_checkpoint.sh 02 wave2 --notes /tmp/notes.md
git add evidence && git commit -m 'chore(evidence): record checkpoint 02 (wave2)'
git tag -a checkpoint/02-wave2 -m 'checkpoint 02: wave2'
```

## Details

* **Branch diffs.** The default set is every local `feat/*` branch whose tip is merged
  into HEAD, plus every branch that a first-parent merge subject names (for example
  `Merge branch 'feat/a12-nn-model': ...`) and that has commits which are not merged
  yet. The diff is `git diff <base> <branch>` with `<base> = git merge-base M^1
  <branch>`, where M is the first merge of the branch into main. The header lists the
  commits, the diffstat and the exact command.
  * Several merge bases (the agent merged main or peer branches that reached main first):
    the base is the virtual merge base that `git merge` computes. Only the agent's
    own changes appear in the diff.
  * Merge bases that conflict with each other (add/add of files that agents force-added
    before `checkpoint/00b-data-fix`): the diff is `git diff M^1 <branch>`, limited to
    the paths the branch changed.
  * Commits after the last merge are listed as not merged yet. They are left out of the
    diff.
  * A branch that was never merged is diffed only when you name it. Branches with no
    commits of their own are skipped.
  * Building a virtual base writes a few git objects, but it never changes a ref.
* **Checkpoint record.** The previous checkpoint is the `checkpoint/<MM>-*` tag with the
  largest MM below NN that is an ancestor of HEAD. The order is 00 < 00b < 01 < 02. The
  test step runs `tools/dev/py -m pytest -q -m "not slow and not gpu"` and records its
  summary line, exit code and `FAILED`/`ERROR` lines. A failing suite is recorded and
  does not stop the script. The notes file is markdown and goes below the header table.
* **Board archive.** The source is `$TABPACK_BOARD` or `<main checkout>/.coord/board.jsonl`,
  the same file that `tools/dev/board.py` uses. `board.md` is rendered from the redacted
  copy and contains per-agent counts, who talked to whom, and every message.
* **Report index.** The areas come from the roster table. Each summary is the first
  paragraph of the report's `Summary` section. The index also lists roster agents that
  have no report.
* **Session snapshot.** The script takes the newest `*.jsonl` directly under
  `~/.claude/projects/<main checkout path, non-alphanumerics -> '-'>/`. It prints the
  sha256 row (raw and redacted) for the table in `evidence/README.md`. `before` refuses to
  overwrite an existing snapshot without `--force`.
