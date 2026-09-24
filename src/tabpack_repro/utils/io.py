"""JSON I/O and provenance (a29)."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import torch

_PACKAGE_DIR = Path(__file__).resolve().parent
_COMMIT_RE = re.compile(r'[0-9a-f]{40}([0-9a-f]{24})?')


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays, Paths, enums, dataclasses.

    The result only contains dict/list/str/int/float/bool/None and is valid strict
    JSON: NaN and +-inf become None (e.g. the NaN roc-auc of a single-class part).
    * numpy scalars -> python scalars; numpy arrays -> nested lists;
    * torch tensors (detached, moved to CPU) -> number if 0-d else nested lists;
    * Path -> str; Enum -> its value; dataclass instance -> dict of its fields
      (as dataclasses.asdict, without deep-copying); tuple -> list;
    * set/frozenset -> sorted list (by JSON text when the elements are not
      mutually comparable);
    * mapping keys: enums/numpy scalars are converted; str/int/float/bool/None keys
      are kept (json turns them into strings); other keys raise TypeError.
    Anything else raises TypeError.
    """
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        if isinstance(obj, np.complexfloating):
            raise TypeError(f'Object of type {type(obj).__name__} is not JSON-able')
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            raise TypeError(f'Complex arrays are not JSON-able (dtype {obj.dtype})')
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        if obj.is_complex():
            raise TypeError(f'Complex tensors are not JSON-able (dtype {obj.dtype})')
        obj = obj.detach().cpu()
        return to_jsonable(obj.item() if obj.ndim == 0 else obj.tolist())
    if isinstance(obj, PurePath):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        return {_to_json_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(x) for x in obj]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON-able')


def _to_json_key(key: Any) -> str | int | float | bool | None:
    if isinstance(key, (enum.Enum, np.generic, PurePath)):
        key = to_jsonable(key)
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    raise TypeError(f'Mapping keys of type {type(key).__name__} are not JSON-able')


def dump_json(path: str | Path, obj: Any) -> None:
    """Atomic write (tmp + rename), indent=2, via to_jsonable; creates parent dirs.

    The object is serialized before anything touches the disk, so a failure leaves
    an existing file unchanged and no temporary file behind. Strict JSON (no NaN).
    """
    path = Path(path)
    text = json.dumps(to_jsonable(obj), indent=2, allow_nan=False, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        # os.open honours the umask (unlike mkstemp's 0600), like a plain open().
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_json(path: str | Path) -> Any:
    """Parse a JSON file (utf-8)."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _git(*args: str) -> str | None:
    """stdout of `git <args>` run in this package's directory, or None on failure."""
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=_PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            # Read-only: never take the index lock (other processes may be committing).
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def git_commit() -> str | None:
    """HEAD commit hash of the repository containing this package, or None."""
    out = _git('rev-parse', 'HEAD')
    if out is None:
        return None
    commit = out.strip()
    return commit if _COMMIT_RE.fullmatch(commit) else None


def git_is_dirty() -> bool | None:
    """True if tracked files of that repository have uncommitted changes (untracked
    files are ignored), False if clean, None when git is unavailable."""
    out = _git('status', '--porcelain', '--untracked-files=no')
    return None if out is None else bool(out.strip())
