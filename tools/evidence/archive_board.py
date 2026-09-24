#!/usr/bin/env python3
"""Archive the coordination board into evidence/coordination/ (stdlib only).

Copies the board (<main checkout>/.coord/board.jsonl, or $TABPACK_BOARD, or --board)
to <out>/board.jsonl, redacts personal email addresses in the copy with
tools/evidence/redact_emails.py, and renders <out>/board.md from the redacted copy:
messages per agent, who talked to whom, and every message in order.
<out> defaults to evidence/coordination/ of the current checkout. Rerunning with an
unchanged board leaves both files unchanged.

  tools/evidence/archive_board.py
  tools/evidence/archive_board.py --board /path/to/board.jsonl --out /tmp/coord
"""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from json import JSONDecodeError, loads
from pathlib import Path

HERE = Path(__file__).resolve().parent
REDACTOR = HERE / 'redact_emails.py'
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
BROADCAST = 'all'
_CODE_SPAN = re.compile(r'(`+)(.+?)\1', re.DOTALL)


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def default_board() -> Path:
    """The board path used by tools/dev/board.py for the current checkout."""
    env = os.environ.get('TABPACK_BOARD')
    if env:
        return Path(env)
    common = Path(_git('rev-parse', '--path-format=absolute', '--git-common-dir'))
    return common.resolve().parent / '.coord' / 'board.jsonl'


def cell(text: object, *, newlines: str = '<br>') -> str:
    """Escape text for a markdown table cell; code spans are kept as they are."""
    s = '' if text is None else str(text)
    parts, pos = [], 0
    for m in _CODE_SPAN.finditer(s):
        parts.append(html.escape(s[pos : m.start()], quote=False))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(html.escape(s[pos:], quote=False))
    s = ''.join(parts).replace('|', r'\|')
    return s.replace('\r\n', '\n').replace('\n', newlines)


def recipients(msg: dict) -> list[str]:
    """Direct recipients of a message ([] for broadcasts); 'a,b' means both."""
    to = msg.get('to') or BROADCAST
    names = [t for t in re.split(r'[,\s]+', str(to)) if t]
    return [] if names in ([], [BROADCAST]) else names


def load(path: Path) -> tuple[list[dict], int]:
    msgs, bad = [], 0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            msg = loads(line)
        except JSONDecodeError:
            bad += 1
            continue
        if isinstance(msg, dict):
            msgs.append(msg)
        else:
            bad += 1
    return msgs, bad


def _kinds(counter: Counter) -> str:
    order = [k for k in KINDS if counter[k]] + sorted(set(counter) - set(KINDS))
    return ', '.join(f'{k} {counter[k]}' for k in order)


def _time(msg: dict) -> str:
    return str(msg.get('time', ''))[:19].replace('T', ' ')


def render(msgs: list[dict], bad: int) -> str:
    sent: Counter = Counter()
    direct_sent: Counter = Counter()
    received: Counter = Counter()
    kinds_by_agent: dict[str, Counter] = defaultdict(Counter)
    pairs: Counter = Counter()
    kinds_by_pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for m in msgs:
        sender, kind = str(m.get('from', '?')), str(m.get('kind', '?'))
        sent[sender] += 1
        kinds_by_agent[sender][kind] += 1
        to = recipients(m)
        if to:
            direct_sent[sender] += 1
        for r in to:
            received[r] += 1
            pairs[sender, r] += 1
            kinds_by_pair[sender, r][kind] += 1

    lines = ['# Coordination board', '']
    intro = (
        'Archived copy of the append-only coordination board (`board.jsonl`, written '
        'with `tools/dev/board.py`), rendered by `tools/evidence/archive_board.py`. '
        'Personal email addresses are redacted.'
    )
    lines += [intro, '']
    if msgs:
        first, last = str(msgs[0].get('time', '')), str(msgs[-1].get('time', ''))
        lines.append(
            f'{len(msgs)} messages from {len(sent)} senders, {first} to {last} '
            f'(times below are in that timezone).'
        )
    else:
        lines.append('The board is empty.')
    if bad:
        lines.append(f'{bad} malformed line(s) in board.jsonl were skipped here.')

    lines += [
        '',
        '## Messages per agent',
        '',
        (
            'Sent = all messages; Direct = sent to named agents (the rest went to '
            '`all`); Received = messages addressed to the agent by name.'
        ),
        '',
        '| Agent | Sent | Direct | Received | Kinds sent |',
        '| :-- | --: | --: | --: | :-- |',
    ]
    for agent in sorted(set(sent) | set(received)):
        lines.append(
            f'| {cell(agent)} | {sent[agent]} | {direct_sent[agent]} '
            f'| {received[agent]} | {_kinds(kinds_by_agent[agent])} |'
        )

    lines += ['', '## Who talked to whom', '']
    if pairs:
        lines += [
            'Messages addressed to named agents, most frequent pairs first.',
            '',
            '| From | To | Messages | Kinds |',
            '| :-- | :-- | --: | :-- |',
        ]
        for (a, b), n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(
                f'| {cell(a)} | {cell(b)} | {n} | {_kinds(kinds_by_pair[a, b])} |'
            )
    else:
        lines.append('No direct messages.')

    lines += [
        '',
        '## Messages',
        '',
        '| Seq | Time | From | To | Kind | Branch | Text |',
        '| --: | :-- | :-- | :-- | :-- | :-- | :-- |',
    ]
    for m in msgs:
        to = ', '.join(recipients(m)) or BROADCAST
        lines.append(
            f'| {cell(m.get("seq"))} | {_time(m)} | {cell(m.get("from"))} '
            f'| {cell(to)} | {cell(m.get("kind"))} | {cell(m.get("branch") or "")} '
            f'| {cell(m.get("text"))} |'
        )
    return '\n'.join(lines) + '\n'


def _write_if_changed(path: Path, data: bytes) -> str:
    if path.exists() and path.read_bytes() == data:
        return 'unchanged'
    path.write_bytes(data)
    return 'written'


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--board', type=Path, help='board.jsonl to archive')
    p.add_argument('--out', type=Path, help='output directory')
    args = p.parse_args(argv)
    board = args.board or default_board()
    out = args.out
    if out is None:
        out = Path(_git('rev-parse', '--show-toplevel')) / 'evidence' / 'coordination'
    if not board.is_file():
        print(f'archive_board: no board at {board}', file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    copy = out / 'board.jsonl'
    tmp = out / '.board.jsonl.tmp'
    try:
        tmp.write_bytes(board.read_bytes())
        report = subprocess.run(
            [sys.executable, str(REDACTOR), str(tmp)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        status = _write_if_changed(copy, tmp.read_bytes())
    finally:
        tmp.unlink(missing_ok=True)
    counts = re.search(r'(\d+) redacted of (\d+) matches', report)
    redacted = f'{counts[1]} email(s) redacted' if counts else report.strip()
    print(f'archive_board: {copy} {status} ({redacted})')

    msgs, bad = load(copy)
    md = out / 'board.md'
    status = _write_if_changed(md, render(msgs, bad).encode('utf-8'))
    print(f'archive_board: {md} {status} ({len(msgs)} messages, {bad} malformed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
