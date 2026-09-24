# Evidence

Provenance for how this repository was built by an integrator session plus parallel
sub-agents (see `docs/AGENTS_PROTOCOL.md`).

| Path | Content |
| :--- | :--- |
| `before/session.jsonl` | Claude Code session transcript at the very start (before any change) |
| `after/session.jsonl` | Session transcript at the end |
| `checkpoints/` | One file per checkpoint tag: git state, test results, notes |
| `diffs/` | One `.diff` per merged agent branch (`git diff <merge-base>..<branch>`) and per checkpoint |
| `agents/` | One report per agent (`<agent-id>.md`), written by the agent itself |
| `coordination/` | Archived copy of the coordination board (`board.jsonl`) and the agent roster |
| `commits.log` | `git log --graph` of the final history |

## Redaction

The repository is public, so personal email addresses in the transcripts are replaced
with `[redacted-email]` by `tools/evidence/redact_emails.py` (non-personal addresses such
as `git@github.com` are kept). Nothing else is modified.

| File | sha256 before redaction | sha256 as committed |
| :--- | :--- | :--- |
| `before/session.jsonl` | `fdd625ab7464ec4f3b70693e8483cabe94685679cac70bfb76903b21bb42c824` | `c6c92553ded7f3ad9cd1bfd0f62c1c122a80b203a167ab1c158ea0c2bea30be8` |
