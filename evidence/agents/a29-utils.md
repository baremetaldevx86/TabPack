# a29-utils

## Summary

Implemented the utility contracts: seeding (`seed_everything`), device/AMP helpers
(`resolve_device`, `make_autocast`, `describe_device`) and JSON I/O plus provenance
(`to_jsonable`, `dump_json`, `load_json`, `git_commit`, and a new public
`git_is_dirty`). No `NotImplementedError` is left in the owned files.

## Files

* `src/tabpack_repro/utils/seed.py`: `seed_everything`
* `src/tabpack_repro/utils/device.py`: `resolve_device`, `make_autocast` (private
  `_ReusableAutocast`), `describe_device`
* `src/tabpack_repro/utils/io.py`: `to_jsonable`, `dump_json`, `load_json`,
  `git_commit`, `git_is_dirty` (private `_git`, `_to_json_key`)
* `tests/utils/test_seed.py`, `tests/utils/test_device.py`, `tests/utils/test_io.py`

## Design decisions

* **seed_everything**: `random.seed`, `np.random.seed`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all` (lazy, so it is safe without CUDA). The seed goes
  through `operator.index` (so numpy ints work and floats raise `TypeError`) and must
  be in `[0, 2**32)`, numpy's legacy limit, else `ValueError`.
* **make_autocast**: returns `_ReusableAutocast`, a subclass of
  `contextlib.AbstractContextManager`. Each `__enter__` creates a new
  `torch.autocast(device_type='cuda', dtype=...)` and pushes it on a stack, and
  `__exit__` pops it and delegates. So the same object can be entered many times,
  nested, or exited by an exception. It exposes `.device_type` and `.dtype`.
  Accepted names are exactly `'bfloat16'` and `'float16'`. Any other string raises
  `ValueError` on every device, so a config typo also fails on CPU. It returns
  `None` when `amp_dtype is None` or the device is not CUDA. float16 needs a
  GradScaler in the trainer (a23), and bfloat16 does not.
* **resolve_device**: `'auto'` gives cuda if available, else cpu. Any other spec
  goes to `torch.device(spec)` unchanged, as the contract says, so `'cuda'` on a
  CPU-only machine does not fail here.
* **describe_device**: `torch` is `str(torch.__version__)`, a plain str rather than
  `TorchVersion`. `gpu` comes from `torch.cuda.get_device_name` only for CUDA
  devices.
* **to_jsonable** returns only dict/list/str/int/float/bool/None, and the result is
  strict JSON:
  * NaN and ±inf become `None`. This covers the NaN roc-auc that a08's
    `compute_metrics` returns for single-class parts.
  * numpy scalars go through `.item()`. Arrays and tensors go through `.tolist()`.
    Tensors are detached and moved to CPU first. A 0-d tensor becomes a number, and
    any other tensor becomes a list (a 1-element 1-d tensor stays a list).
  * Complex values raise `TypeError`.
  * An Enum becomes its `.value`, and a StrEnum becomes a plain str. A `PurePath`
    becomes a str. A tuple becomes a list.
  * A dataclass instance becomes a dict of its fields. This gives the same result
    as `asdict` but skips its deep copy, which matters for CUDA tensors. A dataclass
    class (not an instance) raises `TypeError`.
  * A set becomes a sorted list. If the elements cannot be compared with each
    other, they are sorted by their JSON text, so the order is still deterministic.
  * Mapping keys: enum, numpy and Path keys are converted. str, int, float, bool
    and None keys are kept (json turns them into strings). Any other key, such as a
    tuple, raises `TypeError`. An unsupported value raises `TypeError` too, instead
    of being silently turned into a string.
* **dump_json**:
  * The object is serialized first (`indent=2`, `allow_nan=False`,
    `ensure_ascii=False`, trailing newline), before any file is created.
  * It then writes to a unique `.<name>.<pid>.<rand>.tmp` in the same directory,
    fsyncs, and calls `os.replace`. On any error the temporary file is removed and
    an existing target is left unchanged.
  * The temporary file is created with `os.open(..., 0o666)`, so it respects the
    umask. `mkstemp` would give 0600 files.
  * Parent directories are created as needed.
* **git_commit / git_is_dirty**:
  * Both run `git` in the package's own directory, so from a worktree they report
    that worktree's HEAD. They use a 30 s timeout and set `GIT_OPTIONAL_LOCKS=0`,
    so they never take the index lock while other agents commit.
  * They return `None` on `OSError`/`SubprocessError`. `git_commit` also returns
    `None` if the output is not a 40-hex (or 64-hex) hash.
  * `git_is_dirty` uses `status --porcelain --untracked-files=no`, so only changes
    to tracked files count.

## Tests

`tools/dev/py -m pytest tests/utils -q`: **72 passed, 6 skipped** (the skips are the
gpu-marked tests, because CUDA is hidden).
`TABPACK_GPU=1 tools/dev/py -m pytest tests/utils -q`: **78 passed**. On a real
GPU, autocast gives bf16/fp16 matmul output inside the block and float32 after it.
The same object works when entered 3 times, nested, and after an exception, and
CUDA RNG seeding is reproducible.
`ruff check` and `ruff format --check` on the owned paths pass.

Coverage:
* **Seeding**: reproducibility for random, numpy and torch, including seed
  2**32-1. Different seeds give different draws. The result equals seeding each
  RNG by hand. Out-of-range and float seeds are rejected.
* **resolve_device**: explicit specs, and `'auto'` with CUDA mocked both on and off.
* **make_autocast**: `None` on CPU, and `ValueError` for invalid names on both
  devices.
* **to_jsonable**: the full type matrix, NaN metrics, and the input is not mutated.
* **dump_json / load_json**: round-trip, exact indent=2 text, no temporary files
  left, the old file kept on a serialization failure, the temporary file cleaned up
  on an `os.replace` failure, and permissions that follow the umask.
* **git_commit**: equals `git rev-parse HEAD`. Both git helpers return `None` when
  git is missing or when run outside a repository.

## Coordination

* Merged `checkpoint/00b-data-fix` (fast-forward), as instructed.
* Posted `status` #74 when I started, and `done` at the end.
* No questions were addressed to me. a55 (#97) codes against these contracts
  (`describe_device(device) | {git_commit}`, `dump_json`), which fits this
  implementation.

## Open issues

* `git_is_dirty` is a new public helper. It is not in the frozen contract, but the
  assignment asked for it.
* `resolve_device('cuda')` does not check that CUDA is available. It follows the
  contract, and the error shows up when a tensor is first moved there.
