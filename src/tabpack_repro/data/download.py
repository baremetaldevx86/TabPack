"""Download the official TabPack data bundle and extract a single dataset (a03).

The bundle (~190 MB) is ``tabpack-data.tar.gz`` from the Hugging Face dataset
``Yura52/tabpack-data``. Only ``data/<name>/`` is extracted. The archive was made on
macOS, so it contains AppleDouble ``._*`` files that must be skipped.
"""

from __future__ import annotations

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


def get_data_dir() -> Path:
    """Return the data root: ``$TABPACK_DATA_DIR`` if set, else ``<repo>/data``.

    ``<repo>`` is the project root (the directory containing ``pyproject.toml``),
    found by walking up from this file.
    """
    raise NotImplementedError


def verify_dataset_dir(path: str | Path) -> Path:
    """Check that `path` holds a complete dataset and return it.

    Raise FileNotFoundError otherwise.
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError
