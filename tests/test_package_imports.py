"""Smoke test: every module of the package imports (contracts are importable)."""

import importlib
import pkgutil

import tabpack_repro


def test_all_modules_import() -> None:
    names = [
        m.name
        for m in pkgutil.walk_packages(tabpack_repro.__path__, 'tabpack_repro.')
        if not m.name.endswith('__main__')
    ]
    assert len(names) > 20
    for name in names:
        importlib.import_module(name)
