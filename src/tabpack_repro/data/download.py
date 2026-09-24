"""Download the official TabPack data bundle and extract a single dataset (a03).

The bundle (~190 MB) is ``tabpack-data.tar.gz`` from the Hugging Face dataset
``Yura52/tabpack-data``. Only ``data/<name>/`` is extracted. The archive was made on
macOS, so it contains AppleDouble ``._*`` files that must be skipped.

Only the standard library is used. Nothing is printed: progress is reported through
the ``tabpack_repro.data.download`` logger (INFO level), which is silent unless the
application configures logging.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DATA_BUNDLE_URL = (
    'https://huggingface.co/datasets/Yura52/tabpack-data/resolve/main/'
    'tabpack-data.tar.gz'
)
# sha256 of the bundle downloaded on 2026-09-24.
DATA_BUNDLE_SHA256 = '89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558'

# Files every dataset directory must contain (x_bin/x_cat/x_num are optional in
# general, but Churn has all of them).
REQUIRED_FILES = (
    'info.json',
    'y.npy',
    'splits/default/train.npy',
    'splits/default/val.npy',
    'splits/default/test.npy',
)

_BUNDLE_FILENAME = 'tabpack-data.tar.gz'
_BUNDLE_ROOT = 'data'
_CHUNK_SIZE = 1 << 20
_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """Return the data root: ``$TABPACK_DATA_DIR`` if set, else ``<repo>/data``.

    ``<repo>`` is the project root (the directory containing ``pyproject.toml``),
    found by walking up from this file. If no such directory exists (e.g. the
    package is installed as a wheel), ``<cwd>/data`` is returned.
    """
    env = os.environ.get('TABPACK_DATA_DIR')
    if env:
        return Path(env).expanduser()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file():
            return parent / 'data'
    return Path.cwd() / 'data'


def verify_dataset_dir(path: str | Path) -> Path:
    """Check that `path` holds a complete dataset; return it.

    A dataset is complete when every file of REQUIRED_FILES exists. Raise
    FileNotFoundError (naming the missing files) otherwise.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {path}')
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Dataset directory {path} is incomplete; missing: {", ".join(missing)}'
        )
    return path


def download_dataset(
    name: str = 'churn',
    *,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure ``<data_dir>/<name>`` exists and is complete; return its path.

    * Idempotent: if the dataset dir is already complete and not `force`, do nothing.
    * The bundle is cached at ``<cache_dir>/tabpack-data.tar.gz`` (default cache dir:
      ``<data_dir>/.cache``) and its sha256 must equal DATA_BUNDLE_SHA256, otherwise
      raise RuntimeError (and delete the corrupt file).
    * Download to a temporary file and rename atomically.
    * Extract only members under ``data/<name>/`` whose basename does not start
      with ``._``, with path-traversal protection.

    Details: `force` re-extracts the dataset but reuses a cached bundle whose sha256
    matches (the checksum pins its content). A cached bundle with a wrong checksum is
    deleted and downloaded again once; a freshly downloaded bundle with a wrong
    checksum is deleted and RuntimeError is raised. The dataset is extracted into a
    temporary sibling directory, checked for REQUIRED_FILES and then renamed into
    place, so ``<data_dir>/<name>`` is never left half-written. ValueError is raised
    for an invalid `name` or one that the bundle does not contain; RuntimeError for
    an incomplete dataset in the bundle and for unsafe archive members (symlinks,
    hard links, devices or ``..`` components) inside ``data/<name>/``.
    """
    _check_name(name)
    data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / '.cache'
    target = data_dir / name

    if not force and _is_complete(target):
        logger.debug('Dataset %r is already available at %s', name, target)
        return target

    bundle = _ensure_bundle(cache_dir / _BUNDLE_FILENAME)
    _extract_dataset(bundle, name, target, force=force)
    logger.info('Dataset %r is ready at %s', name, target)
    return target


# ----------------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------------


def _check_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name.startswith('.')
        or '/' in name
        or '\\' in name
    ):
        raise ValueError(f'Invalid dataset name: {name!r}')


def _is_complete(path: Path) -> bool:
    try:
        verify_dataset_dir(path)
    except FileNotFoundError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_bundle(path: Path) -> Path:
    """Return the path of a cached bundle whose sha256 matches, downloading it if
    needed."""
    expected = DATA_BUNDLE_SHA256
    if path.is_file():
        actual = _sha256_file(path)
        if actual == expected:
            logger.debug('Using the cached bundle %s', path)
            return path
        logger.warning(
            'Deleting the corrupt cached bundle %s (sha256 %s, expected %s)',
            path,
            actual,
            expected,
        )
        path.unlink()
    _download_bundle(DATA_BUNDLE_URL, path, expected)
    return path


def _download_bundle(url: str, path: Path, expected_sha256: str) -> None:
    """Download `url` to `path` atomically (temporary file + rename), checking the
    sha256 on the fly. On any failure nothing is left at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading %s to %s', url, path)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.part'
    )
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        request = urllib.request.Request(url, headers={'User-Agent': 'tabpack-repro'})
        with (
            os.fdopen(fd, 'wb') as out,
            urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                f'The data bundle downloaded from {url} is corrupt: sha256 is '
                f'{actual}, expected {expected_sha256}. The file was deleted.'
            )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _member_parts(member_name: str) -> tuple[bool, list[str]]:
    """Split an archive member name into components.

    Returns ``(is_absolute, parts)`` where empty and ``.`` components are dropped
    (``..`` is kept so that callers can reject it).
    """
    is_absolute = member_name.startswith('/')
    parts = [p for p in member_name.split('/') if p not in ('', '.')]
    return is_absolute, parts


def _extract_dataset(bundle: Path, name: str, target: Path, *, force: bool) -> None:
    """Extract ``data/<name>/`` from `bundle` into `target` (atomically)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(dir=target.parent, prefix=f'.{name}.', suffix='.partial')
    )
    try:
        tmp_dir.chmod(0o755)
        logger.info('Extracting %r from %s', name, bundle)
        available = _extract_members(bundle, name, tmp_dir)
        if name not in available:
            raise ValueError(
                f'Dataset {name!r} is not in the data bundle {bundle}. '
                f'Available datasets: {", ".join(sorted(available)) or "none"}.'
            )
        missing = [f for f in REQUIRED_FILES if not (tmp_dir / f).is_file()]
        if missing:
            raise RuntimeError(
                f'Dataset {name!r} in the data bundle {bundle} is incomplete; '
                f'missing: {", ".join(missing)}'
            )

        # Another process may have finished the same extraction in the meantime.
        if not force and _is_complete(target):
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        os.replace(tmp_dir, target)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_members(bundle: Path, name: str, dest: Path) -> set[str]:
    """Write the regular files and directories of ``data/<name>/`` into `dest`.

    Returns the names of all datasets found in the bundle. Members outside
    ``data/<name>/`` (including absolute and ``../`` paths) are never written.
    """
    available: set[str] = set()
    with tarfile.open(bundle, mode='r|gz') as tar:
        for member in tar:
            is_absolute, parts = _member_parts(member.name)
            if is_absolute or len(parts) < 2 or parts[0] != _BUNDLE_ROOT:
                continue
            if not parts[1].startswith('.') and (len(parts) > 2 or member.isdir()):
                available.add(parts[1])
            if parts[1] != name:
                continue
            rel = parts[2:]
            if any(p.startswith('._') for p in rel):
                continue  # AppleDouble metadata written by macOS tar
            if '..' in rel:
                raise RuntimeError(
                    f'Unsafe path in the data bundle (path traversal): {member.name!r}'
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    'Unsafe member in the data bundle (only regular files and '
                    f'directories are allowed): {member.name!r}'
                )
            if member.isdir():
                dest.joinpath(*rel).mkdir(parents=True, exist_ok=True)
                continue
            source = tar.extractfile(member)
            if not rel or source is None:
                raise RuntimeError(
                    f'Unexpected member in the data bundle: {member.name!r}'
                )
            path = dest.joinpath(*rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source, path.open('wb') as out:
                shutil.copyfileobj(source, out, _CHUNK_SIZE)
    return available
