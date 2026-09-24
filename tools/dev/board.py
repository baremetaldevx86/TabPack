#!/usr/bin/env python3
"""A tiny append-only message board for coordinating parallel agents (stdlib only).

The board is a JSON-lines file shared by every worktree: <main-repo>/.coord/board.jsonl
(git-ignored; archived into evidence/coordination/ at checkpoints).

  tools/dev/board.py post --from a09 --kind status "LinearPack implemented, tests green"
  tools/dev/board.py post --from a11 --to a09 --kind question "Is weight (K,in,out)?"
  tools/dev/board.py post --from a09 --kind done --branch feat/a09-nn-linear "ready"
  tools/dev/board.py read --for a11            # messages to a11 or to all
  tools/dev/board.py read --since 120          # messages with seq > 120
  tools/dev/board.py status                    # latest status line per agent
  tools/dev/board.py done                      # which agents have posted 'done'

Kinds: status | question | answer | done | blocker | contract | review | finding
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

KINDS = (
    'status',
    'question',
    'answer',
    'done',
    'blocker',
    'contract',
    'review',
    'finding',
)


def _board_path() -> Path:
    env = os.environ.get('TABPACK_BOARD')
    if env:
        return Path(env)
    common = subprocess.run(
        ['git', 'rev-parse', '--git-common-dir'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    main = Path(common).resolve().parent
    return main / '.coord' / 'board.jsonl'


def _read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_post(args: argparse.Namespace) -> None:
    path = _board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        seq = sum(1 for line in f if line.strip()) + 1
        msg = {
            'seq': seq,
            'time': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
            'from': args.sender,
            'to': args.to,
            'kind': args.kind,
            'branch': args.branch,
            'text': ' '.join(args.text),
        }
        f.write(json.dumps(msg) + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f'posted #{seq}')


def _fmt(m: dict) -> str:
    to = f' -> {m["to"]}' if m.get('to') and m['to'] != 'all' else ''
    br = f' [{m["branch"]}]' if m.get('branch') else ''
    return (
        f'#{m["seq"]} {m["time"][11:19]} {m["from"]}{to} ({m["kind"]}){br}: {m["text"]}'
    )


def cmd_read(args: argparse.Namespace) -> None:
    for m in _read_all(_board_path()):
        if m['seq'] <= args.since:
            continue
        if (
            args.for_
            and m.get('to') not in (None, 'all', args.for_)
            and m['from'] != args.for_
        ):
            continue
        if args.kind and m['kind'] != args.kind:
            continue
        if args.sender and m['from'] != args.sender:
            continue
        print(_fmt(m))


def cmd_status(_: argparse.Namespace) -> None:
    latest: dict[str, dict] = {}
    for m in _read_all(_board_path()):
        if m['kind'] in ('status', 'done', 'blocker'):
            latest[m['from']] = m
    for agent in sorted(latest):
        print(_fmt(latest[agent]))


def cmd_done(_: argparse.Namespace) -> None:
    for m in _read_all(_board_path()):
        if m['kind'] == 'done':
            print(_fmt(m))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    post = sub.add_parser('post')
    post.add_argument('--from', dest='sender', required=True)
    post.add_argument('--to', default='all')
    post.add_argument('--kind', choices=KINDS, default='status')
    post.add_argument('--branch')
    post.add_argument('text', nargs='+')
    post.set_defaults(func=cmd_post)
    read = sub.add_parser('read')
    read.add_argument('--for', dest='for_')
    read.add_argument('--from', dest='sender')
    read.add_argument('--kind', choices=KINDS)
    read.add_argument('--since', type=int, default=0)
    read.set_defaults(func=cmd_read)
    sub.add_parser('status').set_defaults(func=cmd_status)
    sub.add_parser('done').set_defaults(func=cmd_done)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
