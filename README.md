# TabPack reproduction on Churn

A from-scratch reproduction of one experiment from the TabPack paper (ICML 2026). On
the Churn dataset it compares an ordinary MLP, a homogeneous MLP ensemble and a
reduced heterogeneous TabPack.

## Paper

*TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning*, by Yury
Gorishniy, Akim Kotelnikov, Ivan Rubachev and Artem Babenko (ICML 2026).

* Paper: [arXiv:2607.05380](https://arxiv.org/abs/2607.05380)
* Official code: [yandex-research/tabpack](https://github.com/yandex-research/tabpack)
  (compared against commit `05a89e2`)

```bibtex
@inproceedings{gorishniy2026tabpack,
    title={{TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning}},
    author={Yury Gorishniy and Akim Kotelnikov and Ivan Rubachev and Artem Babenko},
    booktitle={ICML},
    year={2026},
}
```

This repository is not affiliated with the authors. Cite the paper, not this
repository, for TabPack itself.

## TL;DR

**Question.** Out of the box on Churn, does a heterogeneous hyperparameter ensemble
(reduced TabPack) beat an untuned MLP and an untuned homogeneous MLP ensemble?

**Data.** Churn from the official TabPack data bundle: 10,000 bank customers, binary
label, and a fixed train/val/test split of 6,400 / 1,600 / 2,000. The metric is test
accuracy.

**Methods.**

| Method | What is trained | Final prediction | Optimizer |
| :-- | :-- | :-- | :-- |
| Ordinary MLP | 1 MLP: 3 blocks x 384, ReLU, dropout 0.1 (fixed, untuned defaults) | That model at its best val epoch | AdamW, lr 1e-3, wd 1e-4 |
| Homogeneous ensemble | K = 16 copies of the MLP recipe, trained as one pack. They differ only in initialization, dropout masks and batch order. | Uniform average of all 16 | AdamW (same recipe) |
| Reduced TabPack | 32 MLPs with sampled hyperparameters (official search space), trained as one pack | Online greedy ensemble of at most 16 entries | Muon + AdamW (official recipe) |

**Protocol.** Each method runs with 5 seeds (0-4). TabPack is reported in two ways.
The single-run numbers come from seeds 0-4. The paper's conservative protocol takes
the configs in the final ensemble of the seed-0 run and retrains only those, with
seeds 0-4. Val is the only split used for decisions, and test is used only for
reporting.

**Reductions versus the official Churn run.** The official run samples 64 models and
builds an ensemble of at most 32. Everything else follows the official config. The
official conservative result on Churn, from the reports shipped with the official
code, is 85.75 ± 0.14 % test accuracy.

Full specification: [docs/EXPERIMENT.md](docs/EXPERIMENT.md). It covers the
hypotheses, the decision rule declared before the runs, and the threats to validity.
One example is the Muon versus AdamW confound between TabPack and the baselines.

<!-- RESULTS:BEGIN -->
*Results pending: this block is filled in from `results/churn/summary.md` after the
runs.*
<!-- RESULTS:END -->

## What is TabPack

TabPack is an ensemble of MLPs whose members have different model and optimizer
hyperparameters, and it is trained in a single run. The run first draws K member
configurations (depth, dropout, learning rates, weight decay) at random from a fixed
search space. It then trains all K MLPs at once as one *pack*. Every layer holds K
weight matrices and runs as one batched matmul, and one optimizer step updates each
member with its own hyperparameters: Muon for the backbone weight matrices, AdamW for
the head and the biases. Each member stops early on its own validation score and then
leaves the pack, so the pack shrinks as training goes on. After every epoch, a greedy
ensemble (Caruana et al., 2004) is rebuilt from the validation predictions of the
finished members and of the current snapshots of the running ones, and it replaces
the old ensemble only if its validation score is strictly better. Training ends
when every member has finished or when this online ensemble has stopped improving.
The final prediction is the average over the selected ensemble entries.

## Quickstart

You need Linux, [uv](https://docs.astral.sh/uv/) and Python 3.12. uv installs Python
3.12 if it is missing. A CUDA GPU is optional: the locked torch build is for CUDA 12.8,
and every command also runs on the CPU, more slowly.

```bash
git clone https://github.com/baremetaldevx86/TabPack.git
cd TabPack

uv sync                   # or: make install  (uv sync --locked)
make download             # = uv run python -m tabpack_repro download   -> data/churn/
make experiment           # = uv run bash scripts/run_churn.sh
make test                 # fast CPU tests: pytest -m "not gpu and not slow"
```

* `make experiment` runs 20 training runs, one after another: 3 methods x 5 seeds,
  plus 5 conservative TabPack retrainings. It then writes the summary and the figures.
  Runs whose `report.json` already exists are skipped, so an interrupted experiment
  can be resumed. Extra flags go through `ARGS`, for example
  `make experiment ARGS="--dry-run"`.
* To summarize existing runs again, use `make report`, which runs
  `tabpack-repro summarize --runs-dir runs/churn --output results/churn`.
* To run a single method with a single seed:

  ```bash
  uv run python -m tabpack_repro run --config configs/churn/tabpack.toml \
    --seed 0 --output runs/churn/tabpack/seed-0
  ```

  `uv run tabpack-repro ...` does the same. The subcommands are `download`, `run`,
  `conservative`, `summarize` and `reference` (see `src/tabpack_repro/cli.py`).
* The parity tests compare this code with the official code. `make test-parity` first
  clones the official repository at the pinned commit into `.reference/tabpack`
  (git-ignored).
* For a CPU-only machine, run `make install-cpu`, then `export UV_NO_SYNC=1` so that
  `uv run` keeps the CPU build of torch.
* `make` with no target lists all targets.

Outputs:

* `runs/churn/<method>/seed-<s>/` holds `report.json`, `config.toml` and
  `predictions.npz` (the `.npz` file is git-ignored).
* `results/churn/` holds the summaries and figures.
* `results/reference/churn_official.json` holds the numbers shipped with the official
  code.

Section 7 of [docs/EXPERIMENT.md](docs/EXPERIMENT.md) gives compute estimates made
before any run: about 10 minutes for the whole suite on the laptop GPU listed below,
and about 1-2 hours on the CPU.

## Repository layout

| Path | Content |
| :-- | :-- |
| `src/tabpack_repro/` | The library: `data/`, `nn/` (pack layers), `optim/` (AdamW and Muon packs), `training/`, `ensembles/`, `methods/` (the three methods plus the conservative protocol), `reporting/`, `cli.py` |
| `configs/churn/` | One TOML file per method: `mlp`, `homogeneous`, `tabpack`, `tabpack-conservative` |
| `scripts/` | `run_churn.sh` / `run_churn.py` (the full experiment) and `benchmark_pack.py` |
| `tests/` | Unit tests that mirror `src/`. Also `tests/parity/` (checks against the official code) and `tests/integration/` |
| `results/` | `reference/` (official Churn numbers), `churn/` (our summaries and figures), `benchmark/` |
| `runs/` | Per-run output directories |
| `docs/` | The documents listed below |
| `evidence/` | Provenance of the multi-agent build (see below) |
| `tools/` | Tooling for the build: the agent runner and coordination board (`dev/`), evidence scripts (`evidence/`) and the integrator's merge helper (`integrator/`) |

Section 2 of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the module-level layout.

## Documentation

| Document | Content |
| :-- | :-- |
| [docs/EXPERIMENT.md](docs/EXPERIMENT.md) | The research question, data and preprocessing, method settings, protocol, decision rule, threats to validity and results |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The codebase: pack abstraction, training loop, online ensemble, parity map, differences from the official code |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Seeds, determinism, hardware, and how to rerun |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Day-to-day workflow: tests, lint, CI |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute changes |
| [docs/AGENTS_PROTOCOL.md](docs/AGENTS_PROTOCOL.md) | The rules every parallel agent followed |
| [docs/CONTRACT_AUDIT.md](docs/CONTRACT_AUDIT.md) | An audit of the module interface contracts |
| [evidence/README.md](evidence/README.md) | What the build evidence contains and how it was redacted |

## How this repository was built

This repository was written by Claude Code. A main *integrator* session worked with 55
parallel sub-agents.

* **Before any code.** The integrator and the user fixed the design: the reductions,
  the baseline defaults, K = 16, the seeds and the protocol. The record is
  [evidence/checkpoints/00-skeleton.md](evidence/checkpoints/00-skeleton.md). The
  integrator then wrote a skeleton with a frozen interface contract (signatures and
  docstrings) for every module, shared test fixtures and the tooling.
* **Parallel agents.** Each of the 55 agents owned a disjoint set of files (see
  [evidence/coordination/roster.md](evidence/coordination/roster.md)) and worked in its
  own git worktree on the branch `feat/<agent-id>`. The environment allowed at most 20
  agents to run at once, so they were started in rolling order as slots freed up.
* **Coordination.** Agents talked through an append-only board (`tools/dev/board.py`)
  where they posted status updates, questions, contract issues, bug findings and `done`
  messages. The board is archived in `evidence/coordination/`. An agent could merge a
  peer's branch to test against it only after the peer had posted `done`.
* **Integration.** The integrator reviewed each branch and merged it into `main` with
  `--no-ff`. At milestones it tagged checkpoints (`checkpoint/00-skeleton`,
  `checkpoint/01-foundations`, ...) and recorded each one in `evidence/checkpoints/`.
* **Evidence.** [evidence/](evidence/README.md) holds:
  * one report per agent;
  * the diff of every merged branch;
  * the commit graph;
  * the Claude Code session transcripts from before and after the build, with personal
    email addresses redacted by `tools/evidence/redact_emails.py`. The sha256 of each
    transcript before and after redaction is listed in `evidence/README.md`.

  Every commit carries an `Agent:` trailer that names its author: an agent id or
  `integrator`.
* **Interruption.** The build was interrupted once by an API usage limit. All agents
  were resumed afterwards.

## Relationship to the official code

This is an independent implementation. We read the official code to understand the
intended behavior. We also clone it, git-ignored, into `.reference/tabpack` at commit
`05a89e2`, and use that clone only for comparison:

* `tests/parity/` imports it to check our modules numerically against their official
  counterparts;
* `tabpack-repro reference` extracts the Churn numbers shipped in its reports.

No official code is copied into this repository. The baselines (ordinary MLP and
homogeneous ensemble) are our additions. Section 10 of
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) lists the official features we did not
implement and the implementation choices that differ.

## Hardware used

We used an NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, sm_120) with torch
2.11.0+cu128 (CUDA 12.8), Python 3.12.3, 12 CPUs and 9 GiB RAM (see
[evidence/checkpoints/00-skeleton.md](evidence/checkpoints/00-skeleton.md)). The
official Churn run used an NVIDIA A100-SXM4-80GB.

## License

This repository has no license file yet. The code is provided for research
reproduction. The datasets keep their original licenses. The official TabPack README
states that it imposes no license restrictions beyond the original licenses of the
datasets. See the paper for the dataset sources.
