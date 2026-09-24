"""Helpers for parity tests against the official TabPack code.

The official repository is cloned OUTSIDE version control (default:
<repo>/.reference/tabpack, git-ignored; override with $TABPACK_REFERENCE_DIR) at
commit 05a89e21b955f12de84889d662e15ca534019aaa. It is imported only inside tests,
only for numerical comparison. Nothing from it is copied into src/.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REFERENCE_COMMIT = '05a89e21b955f12de84889d662e15ca534019aaa'


def reference_dir() -> Path | None:
    default = Path(__file__).resolve().parents[2] / '.reference' / 'tabpack'
    path = Path(os.environ.get('TABPACK_REFERENCE_DIR', default))
    return path if (path / 'src' / 'project' / 'tabpack.py').exists() else None


@pytest.fixture(scope='session')
def official():
    """Return a function importing an official module by name, e.g. 'project.nn'."""
    path = reference_dir()
    if path is None:
        pytest.skip('Official TabPack clone not found (set TABPACK_REFERENCE_DIR)')
    src = str(path / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)

    def import_(name: str) -> ModuleType:
        return importlib.import_module(name)

    return import_


def pytest_collection_modifyitems(config, items):
    for item in items:
        if 'tests/parity/' in str(item.fspath).replace(os.sep, '/'):
            item.add_marker(pytest.mark.parity)
