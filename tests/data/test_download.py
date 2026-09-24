"""Tests for tabpack_repro.data.download (a03).

No test touches the network: fake ``tar.gz`` bundles are built in ``tmp_path`` and
served through ``file://`` URLs by monkeypatching DATA_BUNDLE_URL/DATA_BUNDLE_SHA256.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import stat
import tarfile
from pathlib import Path

import pytest

from tabpack_repro.data import download
from tabpack_repro.data.download import (
    REQUIRED_FILES,
    download_dataset,
    get_data_dir,
    verify_dataset_dir,
)

BUNDLE_NAME = 'tabpack-data.tar.gz'
APPLE_DOUBLE = b'\x00\x05\x16\x07AppleDouble metadata'


def _dataset_files(name: str) -> dict[str, bytes]:
    files = {f: f'{name}:{f}'.encode() for f in REQUIRED_FILES}
    files['x_num.npy'] = f'{name}:x_num'.encode() * 100
    files['x_cat.npy'] = f'{name}:x_cat'.encode()
    return files


CHURN_FILES = _dataset_files('churn')
ADULT_FILES = _dataset_files('adult')


def _file(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    return info, data


def _dir(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _special(
    name: str, type_: bytes, linkname: str = ''
) -> tuple[tarfile.TarInfo, None]:
    """A symlink, hard link, FIFO or device member."""
    info = tarfile.TarInfo(name)
    info.type = type_
    info.linkname = linkname
    return info, None


def _members(
    datasets: dict[str, dict[str, bytes]],
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    """Members laid out like the official macOS-made bundle (with ``._*`` files)."""
    members = [_file('._data', APPLE_DOUBLE), _dir('data/')]
    for name, files in datasets.items():
        members += [_file(f'data/._{name}', APPLE_DOUBLE), _dir(f'data/{name}/')]
        for rel, data in files.items():
            parent, _, base = f'data/{name}/{rel}'.rpartition('/')
            members.append(_file(f'{parent}/._{base}', APPLE_DOUBLE))
            members.append(_file(f'{parent}/{base}', data))
    return members


def _write_bundle(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serve(monkeypatch: pytest.MonkeyPatch, bundle: Path) -> None:
    """Make download.py fetch `bundle` (via a file:// URL) and trust its sha256."""
    monkeypatch.setattr(download, 'DATA_BUNDLE_URL', bundle.as_uri())
    monkeypatch.setattr(download, 'DATA_BUNDLE_SHA256', _sha256(bundle))


def _break_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Any download attempt from now on fails loudly."""
    monkeypatch.setattr(
        download, 'DATA_BUNDLE_URL', (tmp_path / 'no-such-bundle.tar.gz').as_uri()
    )


def _read_tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob('*'))
        if p.is_file()
    }


def _all_names(root: Path) -> list[str]:
    return [p.name for p in root.rglob('*')]


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake bundle with churn + adult, AppleDouble files and a `../evil` member;
    returns the data dir to extract into."""
    members = _members({'churn': CHURN_FILES, 'adult': ADULT_FILES})
    members.append(_file('../evil', b'evil'))
    members.append(_file('data/../evil2', b'evil'))
    members.append(_file(str(tmp_path / 'abs-evil'), b'evil'))
    bundle = _write_bundle(tmp_path / 'server' / BUNDLE_NAME, members)
    _serve(monkeypatch, bundle)
    return tmp_path / 'root' / 'data'


# ----------------------------------------------------------------------------------
# get_data_dir / verify_dataset_dir
# ----------------------------------------------------------------------------------


def test_get_data_dir_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('TABPACK_DATA_DIR', str(tmp_path / 'somewhere'))
    assert get_data_dir() == tmp_path / 'somewhere'


def test_get_data_dir_defaults_to_repo_data(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('TABPACK_DATA_DIR', raising=False)
    data_dir = get_data_dir()
    repo = Path(download.__file__).resolve().parents[3]
    assert data_dir == repo / 'data'
    assert (data_dir.parent / 'pyproject.toml').is_file()


def _make_dataset_dir(path: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(data)
    return path


def test_verify_dataset_dir_accepts_complete_dir(tmp_path: Path):
    path = _make_dataset_dir(tmp_path / 'churn', CHURN_FILES)
    assert verify_dataset_dir(str(path)) == path
    assert isinstance(verify_dataset_dir(path), Path)


def test_verify_dataset_dir_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match='not found'):
        verify_dataset_dir(tmp_path / 'nope')


@pytest.mark.parametrize('missing', REQUIRED_FILES)
def test_verify_dataset_dir_missing_file(tmp_path: Path, missing: str):
    path = _make_dataset_dir(tmp_path / 'churn', CHURN_FILES)
    (path / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        verify_dataset_dir(path)


# ----------------------------------------------------------------------------------
# download_dataset: the happy path
# ----------------------------------------------------------------------------------


def test_download_extracts_only_the_requested_dataset(served: Path, tmp_path: Path):
    target = download_dataset('churn', data_dir=served)

    assert target == served / 'churn'
    assert verify_dataset_dir(target) == target
    assert _read_tree(target) == CHURN_FILES  # no ._* files, nothing else
    assert sorted(p.name for p in served.iterdir()) == ['.cache', 'churn']
    # The bundle is cached (and verified) under <data_dir>/.cache.
    cached = served / '.cache' / BUNDLE_NAME
    assert cached.is_file()
    assert _sha256(cached) == download.DATA_BUNDLE_SHA256
    assert os.listdir(served / '.cache') == [BUNDLE_NAME]  # no *.part leftovers
    # Members outside data/churn/ (incl. ../evil and absolute paths) are ignored.
    names = _all_names(tmp_path)
    assert not {'evil', 'evil2', 'abs-evil'} & set(names)
    assert not any(n.startswith('._') for n in names)


def test_download_other_dataset(served: Path):
    target = download_dataset('adult', data_dir=served)
    assert _read_tree(target) == ADULT_FILES
    assert not (served / 'churn').exists()


def test_download_dir_permissions_are_not_private(served: Path):
    target = download_dataset('churn', data_dir=served)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_download_uses_custom_cache_dir(served: Path, tmp_path: Path):
    cache_dir = tmp_path / 'my-cache'
    download_dataset('churn', data_dir=served, cache_dir=str(cache_dir))
    assert (cache_dir / BUNDLE_NAME).is_file()
    assert not (served / '.cache').exists()


def test_download_is_quiet_by_default(served: Path, capfd: pytest.CaptureFixture):
    download_dataset('churn', data_dir=served)
    download_dataset('churn', data_dir=served)
    out, err = capfd.readouterr()
    assert out == ''
    assert err == ''


def test_download_logs_progress(served: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger=download.__name__):
        download_dataset('churn', data_dir=served)
    assert 'Downloading' in caplog.text
    assert 'Extracting' in caplog.text


# ----------------------------------------------------------------------------------
# download_dataset: idempotency, caching and force
# ----------------------------------------------------------------------------------


def test_download_is_idempotent(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = download_dataset('churn', data_dir=served)
    marker = target / 'marker.txt'
    marker.write_text('kept')
    (served / '.cache' / BUNDLE_NAME).unlink()
    _break_url(monkeypatch, tmp_path)

    # Complete dataset: neither the network nor the cache is touched.
    assert download_dataset('churn', data_dir=served) == target
    assert marker.read_text() == 'kept'
    assert not (served / '.cache' / BUNDLE_NAME).exists()


def test_download_reuses_verified_cache(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    download_dataset('churn', data_dir=served)
    _break_url(monkeypatch, tmp_path)
    # A second dataset from the same (cached) bundle needs no download.
    assert _read_tree(download_dataset('adult', data_dir=served)) == ADULT_FILES


def test_download_repairs_incomplete_dataset(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = download_dataset('churn', data_dir=served)
    (target / 'y.npy').unlink()
    (target / 'stale.txt').write_text('stale')
    _break_url(monkeypatch, tmp_path)

    assert download_dataset('churn', data_dir=served) == target
    assert _read_tree(target) == CHURN_FILES


def test_download_force_reextracts(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = download_dataset('churn', data_dir=served)
    (target / 'info.json').write_bytes(b'tampered')
    (target / 'extra.txt').write_text('extra')
    _break_url(monkeypatch, tmp_path)  # force re-extracts from the verified cache

    assert download_dataset('churn', data_dir=served, force=True) == target
    assert _read_tree(target) == CHURN_FILES


def test_download_force_replaces_symlinked_target(served: Path, tmp_path: Path):
    elsewhere = _make_dataset_dir(tmp_path / 'elsewhere' / 'churn', CHURN_FILES)
    served.mkdir(parents=True)
    (served / 'churn').symlink_to(elsewhere, target_is_directory=True)

    target = download_dataset('churn', data_dir=served, force=True)
    assert not target.is_symlink()
    assert _read_tree(target) == CHURN_FILES
    assert _read_tree(elsewhere) == CHURN_FILES  # the link target is untouched


# ----------------------------------------------------------------------------------
# download_dataset: checksums and atomicity
# ----------------------------------------------------------------------------------


def test_download_rejects_wrong_sha256(served: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(download, 'DATA_BUNDLE_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='sha256'):
        download_dataset('churn', data_dir=served)
    # The corrupt download is deleted and nothing is extracted.
    assert os.listdir(served / '.cache') == []
    assert sorted(os.listdir(served)) == ['.cache']


def test_corrupt_cache_is_deleted_and_redownloaded(served: Path):
    cached = served / '.cache' / BUNDLE_NAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'truncated garbage')

    target = download_dataset('churn', data_dir=served)
    assert _read_tree(target) == CHURN_FILES
    assert _sha256(cached) == download.DATA_BUNDLE_SHA256


def test_corrupt_cache_and_corrupt_download_raise(
    served: Path, monkeypatch: pytest.MonkeyPatch
):
    cached = served / '.cache' / BUNDLE_NAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'truncated garbage')
    monkeypatch.setattr(download, 'DATA_BUNDLE_SHA256', '0' * 64)

    with pytest.raises(RuntimeError, match='corrupt'):
        download_dataset('churn', data_dir=served)
    assert not cached.exists()
    assert os.listdir(served / '.cache') == []


def test_interrupted_download_leaves_no_partial_file(
    served: Path, monkeypatch: pytest.MonkeyPatch
):
    real_urlopen = download.urllib.request.urlopen

    class Flaky:
        def __init__(self, response):
            self._response = response
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._response.close()

        def read(self, n: int = -1) -> bytes:
            self._calls += 1
            if self._calls > 1:
                raise ConnectionResetError('connection lost')
            return self._response.read(16)

    monkeypatch.setattr(
        download.urllib.request,
        'urlopen',
        lambda *args, **kwargs: Flaky(real_urlopen(*args, **kwargs)),
    )
    with pytest.raises(ConnectionResetError):
        download_dataset('churn', data_dir=served)
    assert os.listdir(served / '.cache') == []
    assert not (served / 'churn').exists()


# ----------------------------------------------------------------------------------
# download_dataset: bad names and malicious archives
# ----------------------------------------------------------------------------------


def test_unknown_dataset_is_a_clear_error(served: Path):
    with pytest.raises(ValueError, match=r"'higgs' is not in the data bundle") as e:
        download_dataset('higgs', data_dir=served)
    assert 'adult, churn' in str(e.value)
    assert sorted(os.listdir(served)) == ['.cache']  # no partial directory left


@pytest.mark.parametrize('name', ['', '.', '..', '../churn', 'a/b', '.cache', '._x'])
def test_invalid_dataset_name(served: Path, name: str):
    with pytest.raises(ValueError, match='Invalid dataset name'):
        download_dataset(name, data_dir=served)
    assert not served.exists()


def test_incomplete_dataset_in_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = {k: v for k, v in CHURN_FILES.items() if k != 'y.npy'}
    _serve(
        monkeypatch, _write_bundle(tmp_path / BUNDLE_NAME, _members({'churn': files}))
    )
    data_dir = tmp_path / 'data'
    with pytest.raises(RuntimeError, match=r'incomplete; missing: y\.npy'):
        download_dataset('churn', data_dir=data_dir)
    assert sorted(os.listdir(data_dir)) == ['.cache']


@pytest.mark.parametrize(
    'bad_member',
    [
        pytest.param(_file('data/churn/../../evil', b'evil'), id='dotdot'),
        pytest.param(_file('data/churn/splits/../../../evil', b'evil'), id='deep'),
        pytest.param(_special('data/churn/evil', tarfile.SYMTYPE, '/etc'), id='sym'),
        pytest.param(
            _special('data/churn/evil', tarfile.SYMTYPE, '../..'), id='relsym'
        ),
        pytest.param(_special('data/churn/evil', tarfile.LNKTYPE, '/etc/x'), id='hard'),
        pytest.param(_special('data/churn', tarfile.SYMTYPE, '/tmp'), id='symroot'),
        pytest.param(_special('data/churn/evil', tarfile.FIFOTYPE), id='fifo'),
        pytest.param(_special('data/churn/evil', tarfile.CHRTYPE), id='chardev'),
        pytest.param(_file('data/churn', b'evil'), id='file-as-root'),
    ],
)
def test_malicious_member_in_dataset_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_member
):
    members = _members({'churn': CHURN_FILES})
    members.insert(3, bad_member)
    _serve(monkeypatch, _write_bundle(tmp_path / 'server' / BUNDLE_NAME, members))
    data_dir = tmp_path / 'root' / 'data'

    with pytest.raises(RuntimeError, match=r'Unsafe|Unexpected'):
        download_dataset('churn', data_dir=data_dir)
    assert 'evil' not in _all_names(tmp_path)
    assert sorted(os.listdir(data_dir)) == ['.cache']  # nothing half-extracted


def test_member_names_round_trip_through_tarfile(tmp_path: Path):
    """Guard for the fixtures: tarfile keeps `..` and absolute member names."""
    bundle = _write_bundle(
        tmp_path / 'b.tar.gz',
        [_file('../evil', b'x'), _file('/abs/evil', b'x'), _file('a/../b', b'x')],
    )
    with tarfile.open(bundle) as tar:
        assert tar.getnames() == ['../evil', '/abs/evil', 'a/../b']


# ----------------------------------------------------------------------------------
# The real bundle (already cached by the integrator; never downloaded here)
# ----------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.data
def test_extract_churn_from_real_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, churn_dir: Path
):
    real_bundle = churn_dir.parent / '.cache' / BUNDLE_NAME
    if not real_bundle.is_file():
        pytest.skip(f'The real bundle is not cached at {real_bundle}')
    # A symlinked cache: even if the checksum failed, only the link would be deleted.
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    (cache_dir / BUNDLE_NAME).symlink_to(real_bundle)
    _break_url(monkeypatch, tmp_path)

    target = download_dataset('churn', data_dir=tmp_path / 'data', cache_dir=cache_dir)

    assert real_bundle.is_file()
    extracted = _read_tree(target)
    assert sorted(extracted) == sorted(
        [*REQUIRED_FILES, 'x_bin.npy', 'x_cat.npy', 'x_num.npy']
    )
    assert extracted == _read_tree(churn_dir)
