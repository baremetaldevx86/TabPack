# a50-test-determinism: run-to-run reproducibility tests

## Summary

`tests/integration/test_determinism.py` checks on CPU (fp32) that the pipeline gives
the same results for the same seed. It covers three levels:

1. **`train_pack`**: two calls with the same seeds give a bitwise identical
   `PackTrainResult`: ids, best_steps, train/val/test predictions (raw bytes),
   member reports, per-epoch history (train losses and ensemble scores, without
   `time`), n_epochs/n_steps and the online-ensemble report. There are two cases: a
   heterogeneous ModelPack (n_blocks [1,2,3,2], dropout [0,.1,.2,.3]) with
   per-member AdamWPack, and the same pack with per-member MuonAdamWPack plus an
   OnlineGreedyEnsemble. A change of seed, of batch order only (`seed`) or of
   initialization only (`torch.manual_seed` before building the model) changes the
   first-epoch loss and the predictions. Without dropout, the global RNG state at
   call time has no effect, because the batches come only from train_pack's own
   generator. With dropout it does have an effect, which is why the caller must seed
   the global RNG.
2. **`methods.{mlp,homogeneous,tabpack}.run`** on a tiny dataset in the official
   on-disk format (float32 x_num with a two-valued column, string x_cat,
   `splits/default`), written to tmp_path. Each method runs seed 0, then seed 1, then
   seed 0 again in one process. The two seed-0 runs have identical `report.json`
   (canonical form, without `time`/`time_sec`/`env`), identical `predictions.npz`
   (bytes) and a byte-identical `config.toml`. The seed-1 run differs. A slow test
   dumps each seed-0 config to TOML, reloads it (it equals the dataclass) and runs
   all three methods in a fresh interpreter with `PYTHONHASHSEED=12345`. That test
   first checks that the interpreter's `hash()` really differs, then checks that
   every run reproduces the in-process one.
3. **Sampler and config**: `sample_configs(config.space, n, seed=config.seed)` gives
   the same result across calls, and another seed differs. The member configs
   survive `dump_config`/`load_config` of the run config (they are resampled
   identically) and of an explicit `configs=` config (the conservative path), with
   exact floats and types. Subprocesses with `PYTHONHASHSEED` 0 and 4242 (whose
   `hash()` values are checked to differ) print the same configs. The last two
   checks run on the official Churn space and on a richer space: `$list`,
   categorical, `?loguniform` with an int default, step arguments, nested dicts and
   a plain list.

"Identical" means equal canonical forms. The canonical form keeps dict key order,
tags each scalar with its type (so 1, 1.0 and True differ), compares floats by
`float.hex` and compares arrays by dtype, shape and bytes.

**No nondeterminism found.** There was no finding to post.

## Files

* `tests/integration/test_determinism.py` (new, 15 tests, 1 of them `slow`).
* `evidence/agents/a50-test-determinism.md` (this report).

## Design decisions

* **Explicit seeding.** Seeding is explicit, as in the contract: `torch.manual_seed`
  runs before `ModelPack(...)` in the train_pack tests. The dedicated test shows
  that a missing global seed matters only through dropout.
* **No state leakage.** The methods test runs another seed between the two seed-0
  runs. A method that leaks global state (RNG, cached tensors, logging state) into
  the next run would fail it.
* **Pipeline-wide hash-seed test.** The subprocess test runs the whole pipeline, not
  only the sampler: data loading and preprocessing, pack, Muon and the ensemble. A
  `set`/`hash` ordering anywhere would show up there. It is marked `slow` only
  because a fresh interpreter costs about 10 s here. The runs themselves take about
  2 s; see the environment note below.
* **Readable learning checks.** The tiny method configs use lr 3e-3 for
  mlp/homogeneous. With the default 1e-3 and d_block 16, the homogeneous ensemble
  predicted a single class (test acc 0.5), which made the check trivial. Each run
  must reach test acc > 0.6 and at least 2 epochs, and members must finish at
  different steps (train_pack), so the compared results are not degenerate.
* **Mutation check (not committed).** The check used a pytest plugin in the
  scratchpad. Drawing the batch generator's seed from the global RNG was caught by
  the batch-order and seed-difference tests. Seeding `setup_run` with the clock was
  caught by all three `test_run_same_seed_is_identical` cases.
* **Thread count (informational, not a test).** The train_pack results of both
  cases were also bitwise identical with `torch.set_num_threads(1)` and with 3
  threads at these shapes. This is not guaranteed for larger matmuls, so the tests
  do not assert it.

## Tests

```
tools/dev/py -m pytest tests/integration/test_determinism.py -q
15 passed in 24.11s        (wall 32 s incl. interpreter start, user CPU 45 s)
tools/dev/py -m pytest tests/integration/test_determinism.py -q -m "not slow"
14 passed, 1 deselected in 17.45s
tools/dev/py -m ruff check / ruff format --check tests/integration/test_determinism.py: clean
```

The slowest tests are the fresh-interpreter run (about 12 s, `slow`) and the first
train_pack test (about 6 s). Most of the first test's time is the one-time lazy
import of `torch._dynamo` by `torch.optim`.

## Coordination

* **Merges.** Merged `main` (a23 trainer, a30 mlp, a55 common, a28 config, a24
  sampler), then `feat/a31-method-homogeneous` (done #158) and
  `feat/a32-method-tabpack` (done #159).
* **Board.** Posted status #160. I read a53's scope (#122) and did not repeat its
  pack==independent / removal / checkpoint / dropout-isolation tests. No messages
  were addressed to me. No findings.

## Open issues

* **Environment (integrator).** Each fresh Python process spends about 8 s compiling
  `torch._dynamo`, `sympy` and `torch.distributed.fsdp` sources. `torch.optim`
  imports them lazily on the first optimizer step. The shared venv has no bytecode
  cache for most of torch (680 of 2141 `.py` files have a `.pyc`) and none for sympy
  (0 of 1532). `tools/dev/py` sets `PYTHONDONTWRITEBYTECODE=1`, so the cache is
  never filled. A one-time `python -m compileall -q .venv/lib/python3.12/site-packages`
  by the integrator would save those seconds in every test/CLI process. I did not do
  it, because it writes into the shared venv.
* The conservative method (a33) is not covered here. It calls `tabpack.run` once
  per seed, which the tests do cover.
