# a03-data-download: agent report

## Summary

Implemented the `tabpack_repro.data.download` contract (`get_data_dir`,
`verify_dataset_dir`, `download_dataset`) using the standard library only
(`urllib.request`, `tarfile`, `hashlib`, `tempfile`, `shutil`, `logging`). The
function is idempotent, checks the bundle's sha256, downloads atomically, extracts
only `data/<name>/` (skipping AppleDouble `._*` files) and rejects unsafe archive
members. 44 offline tests pass, including one slow test that extracts Churn from the
real cached bundle and matches `data/churn` byte for byte.

## Files

* `src/tabpack_repro/data/download.py`: implementation. The public signatures and
  constants are unchanged from the skeleton. One docstring was re-wrapped because it
  was 89 characters long (E501).
* `tests/data/test_download.py`: 44 tests.
* `evidence/agents/a03-data-download.md`: this report.

Both source files were added with `git add -f` because of the `.gitignore` issue
described under *Open issues*.

## Design decisions

* **`get_data_dir`** returns `$TABPACK_DATA_DIR` (after `expanduser`) when it is set
  and not empty. Otherwise it walks up from `download.py` to the first directory
  that contains `pyproject.toml` and returns `<that dir>/data`. If no such directory
  exists (for example, when the package is installed as a wheel), it returns
  `<cwd>/data`.
* **`verify_dataset_dir`** raises `FileNotFoundError` when the directory is missing,
  and also when any of `REQUIRED_FILES` is missing; that message lists every missing
  file.
* **Idempotency:** if `<data_dir>/<name>` is already complete and `force` is false,
  the function returns immediately. It does not use the network, and it does not
  read, hash or even require the cached bundle.
* **Checksums:** the hash is computed while the download streams, so the ~190 MB
  file is never read twice. A cached bundle with the wrong hash is deleted (with a
  warning) and downloaded again once. A fresh download with the wrong hash is
  deleted and raises `RuntimeError` with both hashes in the message. `force=True`
  re-extracts the dataset but reuses a cached bundle whose hash matches, because the
  hash pins the content.
* **Atomic download:** the bundle is written to `mkstemp(dir=cache_dir, suffix=
  '.part')` and moved into place with `os.replace` only after the hash matches. A
  `finally` block removes the temporary file after any failure, including a network
  error midway.
* **Atomic extraction:** files are extracted into a `mkdtemp` sibling
  (`.<name>.*.partial`, chmod 755) and checked for `REQUIRED_FILES`. Only then does
  that directory replace the target: an existing incomplete or forced target is
  removed first, and a symlinked target is unlinked, not followed. If another
  process finished the same extraction in the meantime (and `force` is false), the
  temporary copy is discarded.
* **Archive safety:** the archive is read in stream mode (`r|gz`), which avoids gzip
  seeks and is about 20% faster on the real bundle. Files are written manually with
  `extractfile`, never with `extract`/`extractall`.
  * Members are processed only when they are relative and their normalized path
    starts with `data/<name>/`. All others are ignored and never written: `../evil`,
    absolute paths, other datasets, and `._data`.
  * Inside the prefix, any path component that starts with `._` is skipped.
  * A `..` component raises `RuntimeError`. So does any member that is not a
    regular file or directory: symlinks, hard links, FIFOs and devices.
  * Nothing is ever written as a symlink, so later members cannot escape through
    one.
* **Errors:**
  * An invalid `name` (empty, contains `/` or `\`, or starts with `.`) raises
    `ValueError` before any I/O.
  * A name that is not in the bundle raises `ValueError`, and the message lists the
    datasets that are available.
  * A dataset in the bundle that lacks required files raises `RuntimeError`.
* **Quiet:** nothing is printed. Progress goes to the `tabpack_repro.data.download`
  logger at INFO level, and a corrupt cache is logged at WARNING level.
* **Tests never use the network.** Fake bundles are served through `file://` URLs by
  monkeypatching the module constants, which `download.py` reads at call time. The
  real-bundle test points at the shared cache through a symlink and sets a broken
  URL. Even if the hash check failed, only the symlink would be deleted and nothing
  would be downloaded.

## Tests

```
tools/dev/py -m pytest tests/data/test_download.py -q
44 passed in 7.11s        (-m "not slow": 43 passed in ~0.6s)
tools/dev/py -m ruff check  src/tabpack_repro/data/download.py tests/data/test_download.py  -> All checks passed
tools/dev/py -m ruff format --check  (same files)                                           -> already formatted
```

The tests cover:

* `get_data_dir`, with and without the environment variable.
* `verify_dataset_dir`: a complete directory, a missing directory, and each required
  file missing.
* Extraction of exactly the requested dataset: no `._*` files, no other datasets,
  nothing from `../evil`, `data/../evil2` or absolute members.
* Directory permissions, a custom `cache_dir`, quiet output (`capfd`), and log
  messages (`caplog`).
* Idempotency: the network and the cache are left untouched.
* Reuse of the cache for a second dataset, repair of an incomplete dataset, and
  `force` (including a symlinked target).
* A wrong hash on download, and a corrupt cache that is then re-downloaded.
* A corrupt cache followed by a corrupt download.
* An interrupted download (no `.part` file is left).
* An unknown name, invalid names, and a bundle that lacks a required file.
* Nine malicious in-prefix members: `..`, deep `..`, absolute and relative
  symlinks, a hard link, a symlink as the dataset root, a FIFO, a character device,
  and a regular file at the dataset root.
* A fixture guard that checks tarfile keeps `..` and absolute member names.
* The slow data test on the real bundle (about 5-6 s, mostly gzip decompression in
  Python).

To run the tests before the integrator's fix, I copied the untracked skeleton data
package (`__init__`, `dataset`, `numerical`, `categorical`, `pipeline`) from the main
checkout into this worktree. These files are ignored by git and were not committed.

## Coordination

* Posted `status` #5 when I started.
* Posted `blocker` #6 to `integrator` about the `.gitignore` issue.
  a04 (#7), a05 (#9), a06 (#14) and a07 (#12) reported the same problem
  independently and used the same workaround.
* Received no questions addressed to a03, and merged no peer branches because
  nothing here depends on another module.

## Open issues

* **Blocker for the integrator:** the root `.gitignore` rule `data/` (line 12) also
  matches `src/tabpack_repro/data/` and `tests/data/`. As a result:
  * `checkpoint/00-skeleton` has no `tabpack_repro.data` package.
  * `tests/conftest.py` fails to import in a fresh worktree.
  * `ruff` (which respects gitignore and has `extend-exclude = ["data"]`) skips these
    paths unless files are named explicitly.

  Fix: anchor the rules as `/data/` and `/data` in `pyproject.toml`, then commit
  `src/tabpack_repro/data/__init__.py` together with the other skeleton files.

  If the integrator commits the skeleton `download.py` on `main`, merging this
  branch will conflict (add/add) on that file. Resolve it by taking this branch's
  version.
* When the cached bundle's hash does not match, it is re-downloaded once instead of
  raising immediately, a slightly more forgiving reading of the contract. A
  mismatch after a fresh download still deletes the file and raises `RuntimeError`.
* There is no cross-process lock. Two processes that download at the same moment
  each write their own temporary file, and the last `os.replace` wins; both files
  are verified. Two concurrent extractions are handled as described under *Design
  decisions*. A lock would need `fcntl`, which is not portable.
