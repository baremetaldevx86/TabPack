# Checkpoint 00: skeleton

Date: 2026-09-24T13:44:13+00:00

## Decisions (made with the user)

| Topic | Decision |
| :-- | :-- |
| Reduced TabPack | 32 sampled MLPs (official Churn: 64), greedy online ensemble of at most 16 members (official: 32), official search space |
| Baseline hyperparameters | Fixed untuned defaults: 3 blocks x 384, ReLU, dropout 0.1, lr 1e-3, wd 1e-4, batch 256, patience 16 |
| Homogeneous ensemble size | K = 16 (= TabPack's max ensemble size) |
| Optimizers | AdamW for the ordinary MLP and the homogeneous ensemble; Muon+AdamW for TabPack (official) |
| Protocol | 5 seeds per method; TabPack reported both single-run (seeds 0-4) and with the paper's conservative protocol (seed-0 main run selects configs, retrained with seeds 0-4) |
| Official code | Parity tests import it from a git-ignored clone (commit 05a89e2); results are also compared with its shipped Churn reports |
| Git flow | Agent feature branches in worktrees, merged into main with --no-ff by the integrator, checkpoint tags, all branches pushed |
| Transcripts | Committed with personal emails redacted (repo is public) |

## Environment

```
python 3.12.3
torch 2.11.0+cu128 cuda 12.8
gpu NVIDIA GeForce RTX 5050 Laptop GPU (12, 0)
numpy 2.5.3 sklearn 1.9.1 optuna 4.9.0
uv 0.12.8
RAM 9 GiB, CPUs 12
```

## Tests

```
.                                                                        [100%]
1 passed in 0.03s
```

## History

```
fac2a36 docs(evidence): add the wave-1 agent roster with file ownership
d0cd92a refactor: tighten contracts before parallel implementation
2b04bff chore(evidence): snapshot the session transcript before any change
1f0d3b5 docs: add parallel agent protocol, evidence layout and README stub
c8d0cab chore(tools): add slot-limited runner, coordination board and redactor
0a20a5c test: add shared fixtures, parity harness and import smoke test
0a3e71d feat: define frozen interface contracts for every module
93fc33f chore: scaffold project with uv, Python 3.12 and CUDA 12.8 PyTorch
```
