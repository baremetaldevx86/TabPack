# Parallel agent protocol

This repository was built by one integrator (the main Claude Code session) and a fleet
of parallel sub-agents. Every agent follows this protocol. It exists so that 50+
workers can change one codebase at the same time without stepping on each other.

## 1. Where you work

* You own **one git worktree** at `/home/vedant/TabPack/.worktrees/<agent-id>` on the
  branch `feat/<agent-id>` (created for you from the `checkpoint/00-skeleton` commit).
* Run every command from your worktree: `cd /home/vedant/TabPack/.worktrees/<agent-id> && ...`
  (shell state does not persist between tool calls, so repeat the `cd` every time).
* Never touch the main checkout (`/home/vedant/TabPack`) or another agent's worktree.
  Never run `git checkout`/`switch` to another branch, `git push`, `git rebase`,
  `git commit --amend`, `git reset --hard` or `git stash` on shared history.

## 2. What you may edit

* Only the files listed as **owned** in your assignment. Creating new files inside
  your owned directories/prefixes is fine.
* Everything else is read-only for you, including `pyproject.toml`, `uv.lock`,
  `tests/conftest.py`, every `__init__.py` that already exists, and `src/**` files
  owned by others.
* Public signatures in the skeleton (`src/tabpack_repro/**`) are a **frozen contract**.
  Keep names, parameters, return types and documented semantics. You may add private
  helpers (`_name`). If a contract is wrong or ambiguous, post a `contract` message on
  the board addressed to `integrator` and continue with the most reasonable reading.
* Need a new dependency? Ask `integrator` on the board. Never run `uv add`/`uv sync`/
  `pip install`.

## 3. How to run Python

Always go through the slot runner, never `.venv/bin/python` directly:

```bash
cd /home/vedant/TabPack/.worktrees/<agent-id> && tools/dev/py -m pytest tests/<area> -q
cd /home/vedant/TabPack/.worktrees/<agent-id> && tools/dev/py -m ruff check src tests
cd /home/vedant/TabPack/.worktrees/<agent-id> && tools/dev/py -m ruff format <your files>
```

The runner puts *your* worktree's `src/` on `PYTHONPATH`, limits concurrency (the
machine has ~9 GB RAM shared by all agents), hides CUDA, and sets
`TABPACK_DATA_DIR` (real Churn data) and `TABPACK_REFERENCE_DIR` (official code clone).
Only tasks that explicitly need the GPU use `TABPACK_GPU=1`. Keep test runs short
(CPU, small shapes, a few seconds each). Run only your own tests while iterating.
Mark genuinely slow tests with `@pytest.mark.slow`.

## 4. Talking to other agents

The coordination board is an append-only log shared by all worktrees:

```bash
tools/dev/board.py post --from <agent-id> --kind status "started: LinearPack"
tools/dev/board.py post --from <agent-id> --to a11 --kind question "..."
tools/dev/board.py read --for <agent-id>          # everything addressed to you or all
tools/dev/board.py read --since <seq>             # only new messages
tools/dev/board.py done                           # who has finished (+ branches)
```

* Post `status` when you start, and again at significant milestones.
* Read the board for messages addressed to you **before you start, after every
  commit, and before you finish**. Answer questions addressed to you (`answer`).
* When your work depends on another agent's module, you can integration-test against
  it as soon as they post `done`: `git merge --no-edit feat/<their-id>` into your
  branch. Merge only branches whose owner posted `done`. Never cherry-pick or rebase.
* If you find a bug in someone else's module: post a `finding` addressed to them with
  a minimal reproduction. Do not fix their files yourself.
* You may also use the `SendMessage` tool to reach a peer directly if you can see it
  in `ListAgents`. The board remains the record, so also post the gist there.

## 5. Commits

* Small, focused commits in [Conventional Commits](https://www.conventionalcommits.org)
  style with a scope, e.g. `feat(nn): add LinearPack with per-member init`,
  `test(nn): check LinearPack against K independent nn.Linear layers`,
  `docs(agents): add a09 report`.
* Typically 2-4 commits: implementation, tests, and your agent report. Every commit
  must pass `ruff check` for the files it touches. Do not commit failing tests.
* Commit message body: what and why, in one short paragraph. End every message with
  two trailer lines:

  ```
  Agent: <agent-id>
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  ```
* Use `git add <paths>` (never `git add -A`/`.`), and double-check `git status` so you
  only commit owned files.

## 6. Definition of done

1. Implementation complete; contract honored; no `NotImplementedError` left in owned files.
2. Tests for your code under your owned test paths pass (`tools/dev/py -m pytest ...`).
3. `tools/dev/py -m ruff check <owned paths>` and `ruff format --check <owned paths>` pass.
4. Agent report committed at `evidence/agents/<agent-id>.md` with these sections:
   *Summary*, *Files*, *Design decisions*, *Tests* (the command and its result
   summary), *Coordination* (board messages sent/received and merges of peer
   branches), *Open issues*.
5. Post `done` on the board with `--branch feat/<agent-id>` and a one-line summary.
6. Your final answer to the integrator: branch name, commit list (`git log --oneline
   checkpoint/00-skeleton..HEAD`), test results, and any open issues or contract concerns.

## 7. The official code

The official implementation is cloned at `$TABPACK_REFERENCE_DIR`
(`/home/vedant/TabPack/.reference/tabpack`, commit `05a89e2`). Use it **only for
comparison**: read it to understand the intended semantics and import it in
`tests/parity/**` to compare numbers. Never copy its code into `src/`. Write your own
implementation.
