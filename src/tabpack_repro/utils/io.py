"""JSON I/O and provenance (a29)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays, Paths, enums, dataclasses."""
    raise NotImplementedError


def dump_json(path: str | Path, obj: Any) -> None:
    """Atomic write (tmp + rename), indent=2, via to_jsonable; creates parent dirs."""
    raise NotImplementedError


def load_json(path: str | Path) -> Any:
    raise NotImplementedError


def git_commit() -> str | None:
    """HEAD commit hash of the repository containing this package, or None."""
    raise NotImplementedError
