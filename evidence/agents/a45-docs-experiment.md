# a45-docs-experiment: agent report

## Summary

I wrote `docs/EXPERIMENT.md`, the specification of the Churn experiment. It has nine
sections:

1. The research question and hypotheses H1-H3.
2. The dataset facts and the exact preprocessing chain.
3. Full hyperparameter tables for the three methods.
4. A reduced-vs-official table, with the official shipped reference numbers.
5. The protocol: seeds 0-4, the TabPack single-run and conservative protocols,
   val/test discipline, and a decision rule declared before the runs.
6. Fairness notes and threats to validity.
7. Compute expectations.
8. The commands and the output layout.
9. A Results section with the `<!-- RESULTS -->` placeholder.

## Files

- `docs/EXPERIMENT.md` (new)
- `evidence/agents/a45-docs-experiment.md` (this report)

## Design decisions

- **Precedence.** `config.py` is stated as the top source of truth. I printed every
  default through `config_to_dict` and checked each value against the tables (MLP and
  homogeneous: 3 x 384, dropout 0.1, AdamW lr 1e-3 and wd 1e-4, K = 16; TabPack:
  32 models, max ensemble 16, ensemble patience 32, member patience 16, Muon momentum
  0.95, Nesterov, 5 Newton-Schulz steps, official search space).
- **Data facts are measured on `data/churn`, not assumed.** Measured values:
  - 10,000 rows, split 6,400 / 1,600 / 2,000, disjoint.
  - 7 numerical, 3 binary and 1 categorical feature (3 values), no NaN.
  - Positives: 1,304 / 326 / 407. Majority-class accuracy 79.625 % / 79.625 % /
    79.65 %.
  - No numerical column has exactly 2 unique values (the minimum is 4), so
    extract-bin is a no-op.
  - The noisy-quantile transform uses n_quantiles = 213.
  - The categorical cardinalities are `[3, 2, 2, 2]`, which agrees with a06, so the
    model input has 16 features.
  - Parameter counts: 302,593 for the 3-block MLP, 6,913 to 450,433 per TabPack
    member.
- **Official reference numbers come from the shipped reports** in the official clone
  (commit 05a89e2):
  - Main run online ensemble: 85.70 % test (val 87.625 %), 6 entries of which
    5 are unique.
  - Best single member: 85.60 %. Mean of the 57 finished members: 85.46 %.
  - Conservative: 85.75 ± 0.14 %.
  - Runtimes: 17.1 s on an A100, and 4.7-7.0 s per conservative seed.

  I also read `scripts/run_tabpack_experiment.py` and `lib/tools/evaluate.py` to pin
  down the conservative semantics: sorted unique ids, `n_models = len(selected)`,
  seeds `range(5)`, and the online ensemble re-run on the retrained members.
- **The main protocol is TabPack single-run.** Each baseline gets one run per seed, and
  so does TabPack. The conservative number is secondary and comparable with the paper.
  The doc points out that the conservative std excludes config-sampling variance.
- **Decision rule declared before the runs.** "A beats B" requires Δ ≥ 2·SE and
  Δ ≥ 0.10 pp (2 test errors); anything else is inconclusive. No significance claims
  are made with n = 5.
- **Muon vs AdamW confound.** The doc explains the choice (TabPack as shipped vs
  standard practice, and H1 isolates ensembling), says that H2 changes three factors at
  once, and names the two ablations that would separate them. The current configs
  cannot express those ablations.
- **Why the reduction probably matters little.** The official final ensemble (6
  entries) is far below both caps (32 and 16). The main effect of the reduction is the
  smaller pool of sampled configs.
- **Compute section.** All figures in it are labeled as estimates. The measured
  numbers come from `time_sec` in each `report.json`.

## Tests

Docs only, so there are no unit tests. To verify the content:

- I printed the config defaults through `tools/dev/py`.
- I computed the Churn statistics from `$TABPACK_DATA_DIR/churn`.
- I recomputed the official numbers from the shipped `report.json` files.
- Every number in the doc was cross-checked against these outputs.

## Coordination

- I posted `status` at the start (#36).
- I merged `checkpoint/00b-data-fix`, as the integrator asked in #60 (fast-forward).
- I asked a35 (#86) to confirm the order and paths of `run_churn.sh`, and a28 (#87) to
  confirm the config file names `configs/churn/{mlp,homogeneous,tabpack}.toml`.
- No messages were addressed to me before I finished.

## Open issues

- **Section 8 describes interfaces that other agents are still building.**
  `run_churn.sh`/`run_churn.py` (a35), the CLI flags (a34) and the config file names
  (a28) follow the contracts but were not yet implemented. The integrator should check
  section 8 against the merged scripts, including the figure file names in
  `results/churn/`.
- **The MLP input encoding is assumed.** The doc says the ordinary MLP uses the same
  16-feature input, `concat(x_num, one_hot(x_cat))`. The a30 contract only says "same
  data pipeline", so a30 must confirm this.
- **The compute budget is an estimate.** The integrator could replace it with measured
  times when filling in Results.
