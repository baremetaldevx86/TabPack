# Agent roster: wave 1 (55 parallel agents)

Every agent works in `.worktrees/<id>` on branch `feat/<id>`, branched from tag
`checkpoint/00-skeleton`. Paths are relative to the repository root, and `src/` means
`src/tabpack_repro/`. "Deps" lists the agents whose modules the task calls at runtime.
An agent can integration-test against a dependency by merging that agent's branch once
it has posted `done`.

| Id | Area | Owned paths | Deps |
| :-- | :-- | :-- | :-- |
| a01-ci | Infra | `.github/workflows/ci.yml`, `Makefile` | none |
| a02-devtools | Infra | `.pre-commit-config.yaml`, `.editorconfig`, `tools/dev/check.sh` | none |
| a03-data-download | Data | `src/data/download.py`, `tests/data/test_download.py` | none |
| a04-data-dataset | Data | `src/data/dataset.py`, `tests/data/test_dataset.py` | none |
| a05-data-numerical | Data | `src/data/numerical.py`, `tests/data/test_numerical.py` | none |
| a06-data-categorical | Data | `src/data/categorical.py`, `tests/data/test_categorical.py` | none |
| a07-data-pipeline | Data | `src/data/pipeline.py`, `tests/data/test_pipeline.py` | a03 a04 a05 a06 |
| a08-metrics | Metrics | `src/metrics.py`, `tests/test_metrics.py` | none |
| a09-nn-linear | NN | `src/nn/linear_pack.py`, `tests/nn/test_linear_pack.py` | none |
| a10-nn-dropout | NN | `src/nn/dropout_pack.py`, `tests/nn/test_dropout_pack.py` | none |
| a11-nn-mlp | NN | `src/nn/mlp_pack.py`, `tests/nn/test_mlp_pack.py` | a09 a10 |
| a12-nn-model | NN | `src/nn/model_pack.py`, `tests/nn/test_model_pack.py` | a09 a11 |
| a13-nn-packops | NN | `src/nn/pack_ops.py`, `tests/nn/test_pack_ops.py` | none (a09-a12 for integration tests) |
| a14-optim-ns | Optim | `src/optim/newton_schulz.py`, `tests/optim/test_newton_schulz.py` | none |
| a15-optim-adamw | Optim | `src/optim/adamw_pack.py`, `tests/optim/test_adamw_pack.py` | none |
| a16-optim-muon | Optim | `src/optim/muon_adamw_pack.py`, `tests/optim/test_muon_adamw_pack.py` | a14 a15 |
| a17-optim-groups | Optim | `src/optim/pack_utils.py`, `tests/optim/test_pack_utils.py` | a12 |
| a18-train-batches | Training | `src/training/batches.py`, `tests/training/test_batches.py` | none |
| a19-train-losses | Training | `src/training/losses.py`, `tests/training/test_losses.py` | none |
| a20-train-state | Training | `src/training/state.py`, `tests/training/test_state.py` | none |
| a21-train-evaluate | Training | `src/training/evaluate.py`, `tests/training/test_evaluate.py` | a08 a12 |
| a22-train-stopping | Training | `src/training/stopping.py`, `tests/training/test_stopping.py` | none |
| a23-train-trainer | Training | `src/training/trainer.py`, `tests/training/test_trainer.py` | a08 a12 a13 a17 a18 a19 a20 a21 a22 a26 |
| a24-sampler | Sampler | `src/sampler.py`, `tests/test_sampler.py` | none |
| a25-ens-greedy | Ensembles | `src/ensembles/greedy.py`, `tests/ensembles/test_greedy.py` | a08 |
| a26-ens-online | Ensembles | `src/ensembles/online.py`, `tests/ensembles/test_online.py` | a08 a25 a27 |
| a27-ens-aggregate | Ensembles | `src/ensembles/aggregate.py`, `tests/ensembles/test_aggregate.py` | none |
| a28-config | Config | `src/config.py` (loader functions only), `configs/churn/*.toml`, `tests/test_config.py` | none |
| a29-utils | Utils | `src/utils/seed.py`, `src/utils/device.py`, `src/utils/io.py`, `tests/utils/**` | none |
| a30-method-mlp | Methods | `src/methods/mlp.py`, `tests/methods/test_mlp.py` | a55 |
| a31-method-homogeneous | Methods | `src/methods/homogeneous.py`, `tests/methods/test_homogeneous.py` | a55 a12 a15 a17 a23 |
| a32-method-tabpack | Methods | `src/methods/tabpack.py`, `tests/methods/test_tabpack.py` | a55 a12 a16 a17 a23 a24 a26 |
| a33-method-conservative | Methods | `src/methods/conservative.py`, `tests/methods/test_conservative.py` | a32 |
| a34-cli | CLI | `src/cli.py`, `tests/test_cli.py` | a03 a28 a30-a33 a36 a38 |
| a35-experiment-runner | Experiment | `scripts/run_churn.py`, `scripts/run_churn.sh`, `tests/scripts/test_run_churn.py` | a28 a30-a33 a36 a37 |
| a36-report-summarize | Reporting | `src/reporting/summarize.py`, `tests/reporting/test_summarize.py` | a29 |
| a37-report-plots | Reporting | `src/reporting/plots.py`, `tests/reporting/test_plots.py` | none |
| a38-report-reference | Reporting | `src/reporting/reference.py`, `results/reference/churn_official.json`, `tests/reporting/test_reference.py` | none |
| a39-parity-nn | Parity | `tests/parity/test_parity_nn.py` | a09-a13 |
| a40-parity-optim | Parity | `tests/parity/test_parity_optim.py` | a14-a17 |
| a41-parity-ensemble | Parity | `tests/parity/test_parity_ensemble.py` | a08 a25 a26 |
| a42-parity-data | Parity | `tests/parity/test_parity_data.py` | a04-a07 |
| a43-parity-sampler | Parity | `tests/parity/test_parity_sampler.py` | a24 |
| a44-docs-architecture | Docs | `docs/ARCHITECTURE.md` | none |
| a45-docs-experiment | Docs | `docs/EXPERIMENT.md` | none |
| a46-docs-reproducibility | Docs | `docs/REPRODUCIBILITY.md` | none |
| a47-docs-development | Docs | `docs/DEVELOPMENT.md`, `CONTRIBUTING.md` | none |
| a48-docs-readme | Docs | `README.md` | none |
| a49-test-e2e | QA | `tests/integration/test_end_to_end.py` | all methods, a34 |
| a50-test-determinism | QA | `tests/integration/test_determinism.py` | a23 a30-a32 |
| a51-bench | Perf | `scripts/benchmark_pack.py`, `results/benchmark/**` | a12 a15 a16 a17 |
| a52-evidence-tools | Evidence | `tools/evidence/**` except `redact_emails.py` | none |
| a53-test-invariants | QA | `tests/integration/test_pack_invariants.py` | a09-a13 a15-a17 a20 |
| a54-contract-audit | QA | `docs/CONTRACT_AUDIT.md` | none (reviews all contracts) |
| a55-method-common | Methods | `src/methods/common.py`, `tests/methods/test_common.py` | a07 a08 a28 a29 |

Everyone also owns `evidence/agents/<id>.md` (their report).
