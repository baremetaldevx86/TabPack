# Checkpoint 01: foundations

Date: 2026-09-24T14:00:23+00:00

## Summary

First integration wave. Merged into main with --no-ff: a01 (CI, Makefile), a02 (dev tooling), a03-a06 (data download, loading, numerical and categorical preprocessing), a08 (metrics), a09-a12 (LinearPack, DropoutPack, MLPBackbonePack, ModelPack), a14 (Newton-Schulz), a18-a20 (batches, losses, PackState), a22 (early stopping). Also includes the skeleton fix tagged checkpoint/00b-data-fix: the unanchored 'data/' ignore rule had kept the data package out of the first skeleton tag; five agents reported it independently on the board.

Agents running at this checkpoint (concurrency capped at 20 by the environment): a07, a13, a15, a16, a17, a21, a23-a30, a36, a44, a45, a51, a52, a55.

## Parity highlights reported by agents

- a04 load_raw_dataset, a05 noisy-quantile, a06 bin/cat preprocessing: bit-identical to the official code on Churn (cardinalities [3, 2, 2, 2]).
- a08 binclass accuracy (round half to even): bit-identical to official calculate_metrics_pack.
- a09 LinearPack, a10 DropoutPack, a11 MLPBackbonePack: bit-identical outputs/grads vs official modules with copied weights (our weight layout is the transpose: (K, in, out)).
- a14 Newton-Schulz: bit-identical to the reference vendor/muon.py on CPU and CUDA.
- a18 batches, a22 stopping: identical to the official formulas.

## Tests on main

```
526 passed, 10 skipped in 26.89s
10 passed, 526 deselected in 4.47s
All checks passed!
```

## History since checkpoint/00-skeleton

```
c05f3ce Merge branch 'feat/a22-train-stopping': early stopping rule and finished-member pool
574666d chore(tools): add the integrator's --no-ff merge helper
6fc63a1 Merge branch 'feat/a20-train-state': per-member training state
26d21fb Merge branch 'feat/a19-train-losses': per-member loss functions
ebfe452 Merge branch 'feat/a18-train-batches': per-member training batches (official RNG order)
59fc4a9 Merge branch 'feat/a14-optim-ns': batched Newton-Schulz orthogonalization
8411474 Merge branch 'feat/a12-nn-model': ModelPack and one-hot encoding
a9c9a00 Merge branch 'feat/a11-nn-mlp': MLPBackbonePack with per-member depth
b41a8ec Merge branch 'feat/a10-nn-dropout': DropoutPack with per-member rates
c22d66f Merge branch 'feat/a09-nn-linear': LinearPack with per-member nn.Linear init
25d2e87 Merge branch 'feat/a08-metrics': metrics and batched pack scores
0a459c3 Merge branch 'feat/a06-data-categorical': binary/categorical preprocessing (bit-exact vs official)
871cf36 Merge branch 'feat/a05-data-numerical': noisy-quantile numerical transform (bit-exact vs official)
edbd974 Merge branch 'feat/a04-data-dataset': raw dataset loading with validated splits
a694022 Merge branch 'feat/a03-data-download': sha256-verified, traversal-safe dataset download
f523c84 Merge branch 'feat/a02-devtools': pre-commit hooks, editorconfig and tools/dev/check.sh
7b05333 Merge branch 'feat/a01-ci': GitHub Actions CI (lint, CPU tests, parity) and Makefile
e8e129c fix: track the data package that the data/ ignore rule swallowed
```
