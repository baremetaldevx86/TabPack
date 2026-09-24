# a44-docs-architecture

## Summary

Wrote `docs/ARCHITECTURE.md`, the codebase guide for new contributors. It has ten
sections: TabPack overview, repository layout and import layering, the pack
abstraction (`(K, B, d)` layout, ids versus pack positions, the pack invariant, why
removal is a dim-0 slice that keeps parameter identity), a Mermaid data-flow
flowchart from the raw Churn files to `report.json`, a Mermaid sequence diagram plus
step list for one epoch of `train_pack`, per-member hyperparameters (n_blocks,
dropout, `(K,)` optimizer tensors, Muon versus AdamW groups), the online greedy
ensemble with a worked example and a real official trace, how the three methods
reuse the machinery, where parity is tested (with a name-translation table), and the
differences from the official implementation. It describes contracts and semantics,
not line numbers.

## Files

* `docs/ARCHITECTURE.md` (new)
* `evidence/agents/a44-docs-architecture.md` (this report)

## Design decisions

* Sources: the skeleton contracts in `src/tabpack_repro/**` (authoritative), board
  messages from implementers (a09-a13, a15, a20, a21), and the official code at
  `05a89e2` (`project/tabpack.py`, `nn.py`, `optim.py`, `ensemble_utils_torch.py`,
  the Churn `main` and `eval-online-ensembles` configs and reports).
* Facts checked instead of assumed:
  * Churn shapes (10,000 rows, 6,400/1,600/2,000; 7 num, 3 bin, 1 cat, no binary
    numeric column) and, with the official `lib.data.build_dataset`, cardinalities
    `[3, 2, 2, 2]`, so the MLP input width is 16 and the first-block Muon scale is
    `sqrt(24)`.
  * The official `tabpack` Churn config has no numerical embeddings and a fixed
    `d_block = 384`; only `tabpack-cosine*` (TabPack†) uses embeddings. So these are
    scope restrictions, not deviations from the reproduced config.
  * The official optimizer reads `muon_update_scale` while `tabpack.py` sets
    `muon_scale`, so the official code uses the shape-based fallback; it is identical
    to ours when `d_block` is shared.
  * With a seeded optuna `RandomSampler`, the first 32 of 64 sampled configs equal the
    32-config sample (verified with `lib.tools.tune._sample_config`).
  * The official conservative-evaluation configs equal the configs of the sorted
    unique ids `[4, 22, 40, 50, 53]` of the main run's final ensemble (verified).
  * The official 64-member Churn run finished only 57 members before the ensemble
    patience ran out, and its final ensemble holds member 4's step-225 snapshot twice.
    Both facts illustrate the loop-termination and pool-multiset semantics.
* The worked example uses a toy 3-member pack with ensemble patience 1, so it shows
  first acceptance, rejection on a tie, acceptance that reuses an old snapshot held
  only by the current ensemble, and termination.
* The official DropoutPack output under bf16 is float32 and ours is bfloat16; per a10
  the values agree after bf16 rounding. The doc calls this equivalent, not a source
  of drift.

## Tests

Documentation only; no code or tests changed.

* Internal anchors: a script compared every `#anchor` link with the GitHub slugs of
  the headings. All 15 resolve.
* File links: `AGENTS_PROTOCOL.md` exists. `EXPERIMENT.md` (a45, done),
  `REPRODUCIBILITY.md` (a46) and `DEVELOPMENT.md` (a47) are written concurrently and
  resolve after merge. The one deep link, `EXPERIMENT.md#4-reduced-vs-official-tabpack`,
  matches the heading on `feat/a45-docs-experiment`.
* Mermaid: checked by eye, since no Mermaid CLI is installed. Every label with
  punctuation is quoted, edge labels are quoted, no node id is `end`, no message
  starts with `end`, and the `loop`/`opt` blocks are balanced.
* Prose lines are at most 88 characters. Only the two longest TOC link lines and
  some table rows are longer.

## Coordination

* Posted `status` (#35) at start; `done` at the end.
* Merged `checkpoint/00b-data-fix` (fast-forward), as the integrator asked in #60.
* Read without merging: `feat/a11-nn-mlp` (MLPBackbonePack schedule and the skipping
  of unused blocks), `feat/a13-nn-packops` (pack_ops docstrings), and
  `feat/a45-docs-experiment` (headings, for consistent terms and the deep link).
* Used from the board: a09 #33 (weight transpose, `pack_idx` to `member_idx`), a10
  #58/#59 (dropout dtype and RNG), a11/a13 #40/#53 (no removal while an autograd graph
  is alive), a12 #77 (one-hot for codes >= cardinality), a15 #92 (shared step storage),
  a20 #64 (PackState host/device split).
* No questions were addressed to a44.

## Open issues

* The descriptions of `tests/parity/*` (a39-a43) and
  `tests/integration/test_pack_invariants.py` (a53) come from the roster scopes,
  because those files had not landed yet. After merge, check that each file covers
  what the parity table says.
* Step 8 of the epoch description says that the trainer restores stopped members'
  best weights and predicts train/val/test, as the official code does. The contract
  implies this but does not spell it out, so check it against a23's `train_pack`.
