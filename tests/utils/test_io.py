from __future__ import annotations

import enum
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import torch

from tabpack_repro.utils import io as io_mod
from tabpack_repro.utils.io import (
    dump_json,
    git_commit,
    git_is_dirty,
    load_json,
    to_jsonable,
)


class Color(enum.Enum):
    RED = 'red'
    BLUE = 2


class Part(enum.StrEnum):
    VAL = 'val'


@dataclass
class Inner:
    path: Path
    values: np.ndarray


@dataclass
class Outer:
    name: str
    inner: Inner
    tags: set[str] = field(default_factory=set)


def _strict(obj: object) -> str:
    return json.dumps(obj, allow_nan=False)


# to_jsonable: scalars


@pytest.mark.parametrize(
    ('value', 'expected', 'kind'),
    [
        (None, None, type(None)),
        (True, True, bool),
        (3, 3, int),
        (1.5, 1.5, float),
        ('s', 's', str),
        (np.bool_(True), True, bool),
        (np.int8(-3), -3, int),
        (np.int64(2**40), 2**40, int),
        (np.uint32(7), 7, int),
        (np.float16(0.5), 0.5, float),
        (np.float32(0.25), 0.25, float),
        (np.float64(0.125), 0.125, float),
        (np.str_('x'), 'x', str),
    ],
)
def test_scalars(value: object, expected: object, kind: type) -> None:
    out = to_jsonable(value)
    assert out == expected
    assert type(out) is kind


@pytest.mark.parametrize(
    'value',
    [
        math.nan,
        math.inf,
        -math.inf,
        np.float32('nan'),
        np.float64('inf'),
        torch.tensor(float('nan')),
        torch.tensor(float('-inf'), dtype=torch.bfloat16),
    ],
)
def test_non_finite_become_none(value: object) -> None:
    assert to_jsonable(value) is None


def test_nan_metrics_dict_is_strict_json() -> None:
    # compute_metrics returns NaN roc-auc for single-class parts.
    metrics = {'accuracy': np.float64(0.8), 'roc-auc': np.float64('nan')}
    out = to_jsonable(metrics)
    assert out == {'accuracy': 0.8, 'roc-auc': None}
    assert _strict(out) == '{"accuracy": 0.8, "roc-auc": null}'


# to_jsonable: arrays and tensors


def test_numpy_arrays() -> None:
    assert to_jsonable(np.arange(6, dtype=np.int32).reshape(2, 3)) == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert to_jsonable(np.array([0.5, np.nan, np.inf])) == [0.5, None, None]
    assert to_jsonable(np.array([True, False])) == [True, False]
    assert to_jsonable(np.array(3.0)) == 3.0
    assert to_jsonable(np.zeros((2, 0))) == [[], []]
    obj = np.array([Path('a'), None], dtype=object)
    assert to_jsonable(obj) == ['a', None]


def test_torch_tensors() -> None:
    out = to_jsonable(torch.tensor(3))
    assert out == 3 and type(out) is int
    out = to_jsonable(torch.tensor(0.5))
    assert out == 0.5 and type(out) is float
    out = to_jsonable(torch.tensor(True))
    assert out is True
    assert to_jsonable(torch.tensor([1.0, float('nan')])) == [1.0, None]
    assert to_jsonable(torch.arange(4).reshape(2, 2)) == [[0, 1], [2, 3]]
    assert to_jsonable(torch.tensor([0.5, 1.5], dtype=torch.bfloat16)) == [0.5, 1.5]
    assert to_jsonable(torch.ones(2, requires_grad=True)) == [1.0, 1.0]
    assert to_jsonable(torch.ones(1)) == [1.0]  # 1-d stays a list


@pytest.mark.gpu
def test_cuda_tensor(cuda_device: torch.device) -> None:
    assert to_jsonable(torch.tensor([1, 2], device=cuda_device)) == [1, 2]
    assert to_jsonable(torch.tensor(0.5, device=cuda_device)) == 0.5


def test_complex_raises() -> None:
    with pytest.raises(TypeError):
        to_jsonable(np.complex64(1j))
    with pytest.raises(TypeError):
        to_jsonable(np.array([1j]))
    with pytest.raises(TypeError):
        to_jsonable(torch.tensor([1j]))


# to_jsonable: containers and other types


def test_paths_and_enums() -> None:
    assert to_jsonable(Path('/tmp/a b/c.json')) == '/tmp/a b/c.json'
    assert to_jsonable(PurePosixPath('x/y')) == 'x/y'
    assert to_jsonable(Color.RED) == 'red'
    assert to_jsonable(Color.BLUE) == 2
    out = to_jsonable(Part.VAL)
    assert out == 'val' and type(out) is str


def test_dataclasses() -> None:
    obj = Outer(
        name='run',
        inner=Inner(path=Path('p'), values=np.array([1.0, np.nan])),
        tags={'b', 'a'},
    )
    assert to_jsonable(obj) == {
        'name': 'run',
        'inner': {'path': 'p', 'values': [1.0, None]},
        'tags': ['a', 'b'],
    }
    with pytest.raises(TypeError):
        to_jsonable(Outer)  # the class itself is not an instance


def test_sets_are_sorted() -> None:
    assert to_jsonable({3, 1, 2}) == [1, 2, 3]
    assert to_jsonable(frozenset({'b', 'a'})) == ['a', 'b']
    mixed = to_jsonable({1, 'a', None})
    assert sorted(mixed, key=repr) == sorted([1, 'a', None], key=repr)
    assert to_jsonable({1, 'a', None}) == mixed  # deterministic


def test_nested_containers_and_keys() -> None:
    obj = {
        'a': (1, np.int64(2)),
        np.int64(5): [torch.tensor(1.0)],
        Color.RED: {'deep': [Path('q')]},
        Part.VAL: 1,
        3: 'int key',
    }
    out = to_jsonable(obj)
    assert out == {
        'a': [1, 2],
        5: [1.0],
        'red': {'deep': ['q']},
        'val': 1,
        3: 'int key',
    }
    assert all(type(k) in (str, int) for k in out)
    _strict(out)


def test_unsupported_raises() -> None:
    with pytest.raises(TypeError, match='object'):
        to_jsonable(object())
    with pytest.raises(TypeError, match='keys'):
        to_jsonable({(1, 2): 'tuple key'})
    with pytest.raises(TypeError):
        to_jsonable({'x': [lambda: 1]})


def test_input_is_not_mutated() -> None:
    data = {'a': [np.float64('nan')], 's': {2, 1}}
    to_jsonable(data)
    assert math.isnan(data['a'][0])
    assert data['s'] == {1, 2}


# dump_json / load_json


def test_dump_load_roundtrip(tmp_path: Path) -> None:
    obj = {'x': np.arange(3), 'y': {'z': Path('p')}, 'name': 'Füße'}
    path = tmp_path / 'report.json'
    dump_json(path, obj)
    assert load_json(path) == {'x': [0, 1, 2], 'y': {'z': 'p'}, 'name': 'Füße'}
    assert load_json(str(path)) == load_json(path)


def test_dump_format(tmp_path: Path) -> None:
    path = tmp_path / 'a.json'
    dump_json(path, {'a': [1], 'b': float('nan')})
    text = path.read_text(encoding='utf-8')
    assert text == '{\n  "a": [\n    1\n  ],\n  "b": null\n}\n'
    assert 'NaN' not in text


def test_dump_creates_parents_and_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / 'deep' / 'er' / 'out.json'
    dump_json(str(path), [1, 2])
    dump_json(path, [3])  # overwrite in place
    assert load_json(path) == [3]
    assert [p.name for p in path.parent.iterdir()] == ['out.json']


def test_dump_failure_keeps_old_file(tmp_path: Path) -> None:
    path = tmp_path / 'out.json'
    dump_json(path, {'ok': 1})
    with pytest.raises(TypeError):
        dump_json(path, {'bad': object()})
    assert load_json(path) == {'ok': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['out.json']


def test_dump_write_failure_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(src: object, dst: object) -> None:
        raise OSError('disk full')

    path = tmp_path / 'out.json'
    monkeypatch.setattr(io_mod.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='disk full'):
        dump_json(path, {'a': 1})
    assert list(tmp_path.iterdir()) == []


def test_dump_file_permissions_follow_umask(tmp_path: Path) -> None:
    path = tmp_path / 'perm.json'
    dump_json(path, {})
    reference = tmp_path / 'plain.json'
    reference.write_text('{}')
    assert path.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777


# git provenance


def test_git_commit_matches_head() -> None:
    commit = git_commit()
    assert commit is not None
    assert len(commit) == 40
    int(commit, 16)
    expected = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=Path(io_mod.__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit == expected


def test_git_is_dirty_returns_bool() -> None:
    assert isinstance(git_is_dirty(), bool)


def test_git_helpers_return_none_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError('git')

    monkeypatch.setattr(io_mod.subprocess, 'run', missing)
    assert git_commit() is None
    assert git_is_dirty() is None


def test_git_helpers_return_none_outside_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(io_mod, '_PACKAGE_DIR', tmp_path)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    assert git_commit() is None
    assert git_is_dirty() is None
