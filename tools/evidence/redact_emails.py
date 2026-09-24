#!/usr/bin/env python3
"""Redact personal email addresses in a text file (in place), keeping non-personal ones.

Used for session transcripts committed to this (public) repository.
Usage: tools/evidence/redact_emails.py FILE [FILE ...]
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
KEEP = (b'git@github.com', b'noreply@anthropic.com')
KEEP_DOMAINS = (b'@users.noreply.github.com',)
REPLACEMENT = b'[redacted-email]'


def _replace(match: re.Match[bytes]) -> bytes:
    s = match.group(0)
    if s in KEEP or s.endswith(KEEP_DOMAINS):
        return s
    return REPLACEMENT


def main(paths: list[str]) -> None:
    for path in map(Path, paths):
        raw = path.read_bytes()
        red, n = EMAIL.subn(_replace, raw)
        n_redacted = red.count(REPLACEMENT) - raw.count(REPLACEMENT)
        path.write_bytes(red)
        print(
            f'{path}: {n_redacted} redacted of {n} matches; '
            f'sha256 raw={hashlib.sha256(raw).hexdigest()} '
            f'redacted={hashlib.sha256(red).hexdigest()}'
        )


if __name__ == '__main__':
    main(sys.argv[1:])
