# a41-parity-ensemble: parity tests for metrics and ensembles

## Summary

`tests/parity/test_parity_ensemble.py` compares three of our modules with the official
code, imported through the `official` fixture. No mismatch was found.

* **Metrics (a08).** Binclass accuracy from `score_pack` and `make_score_fn` is
  bit-identical to the official `calculate_metrics_pack` and `make_emsemble_score_fn`.
  This was checked in float32 and float64 on 40 random packs. The packs include rows
  at exactly 0.5 and one ulp on either side of 0.5, rows of all 0 and all 1, and
  members quantized to multiples of 1/2 to 1/16. p = 0.5 counts as class 0 in both
  implementations. ROC-AUC agrees within 2e-6 on tie-free predictions. Log-loss agrees
  with the official cross-entropy within rtol 1e-5 for p in [0.01, 0.99].
* **Greedy selection (a25).** Our `greedy_ensemble` returns the same indices as the
  official `greedy_ensemble` with default options on **900/900** cases: 150 random
  pools per score, times 3 scores, times 2 size limits. Each pool has M in 2..40 and
  N in 450..550. The members share structure (a common signal and a common error
  term). About 30% of them are quantized, and some rows are exact duplicates. The
  scores are accuracy (each side uses its own score function), a continuous Brier
  score, and a Brier score quantized to 1/500 (the Brier scores use one function
  shared by both sides). Each pool is run with `max_ensemble_size=None` and with one
  random limit from {1, 2, 3, 5, 8, M, M+3}. The test also checks that the cases are
  non-trivial. Per score, 64-82% of the selections have more than one member and
  95-99% leave members out. 19-27% of the cases include at least one real tie-break,
  where several candidates reach the same improving best score. Tie-breaks are counted
  by wrapping the official score function.
* **Online ensemble (a26).** `OnlineGreedyEnsemble` and the official
  `project.tabpack.OnlineEnsemble` were driven with the same inputs over 60 simulated
  20-epoch runs, **762 updates** in total. The official ensemble was configured with
  `type='greedy'`, `update_type='latest'`, `include_current_ensemble_in_pool=True`,
  `update_part='val'`, `prediction_type=PROBS`, patience in {0, 1, 2, 4, 50} and
  `max_ensemble_size` in {None, 3, 8}. It scored with an official
  `lib.data.Task` + `make_emsemble_score_fn` for accuracy, or with a shared continuous
  Brier score. At every epoch the two classes agree on:
  * the `improved` flag;
  * `ids` and `steps` (int64);
  * the val score;
  * `is_running`;
  * the stored per-entry predictions, bit-identical to the official
    `_predictions_torch` and `_predictions`;
  * the averaged `predictions()`, bit-identical to the official
    `compute_ensemble_prediction`.

  After each run, `report(labels)` is also checked against what the official
  `update_online_ensembles` would record. `ids`, `steps` and `score_val` are equal.
  Accuracy and score are equal, ROC-AUC agrees within rel 1e-12, and log-loss agrees
  with cross-entropy within rel 1e-5 on train/val/test.

  The updates cover 471 accepted and 291 rejected updates, 331 updates with the same
  id twice in the ensemble (an old and a newer snapshot), 430 updates with finished
  members, and 8 updates where every member had finished (0 running rows). In 44 runs
  the ensemble stopped on patience.

Runtime: about 7-10 s for the whole module (11 tests) on a CPU shared with other
agents.

## Files

* `tests/parity/test_parity_ensemble.py`: the 11 parity tests.
* `evidence/agents/a41-parity-ensemble.md`: this report.

## Design decisions

* **Each side uses its own score function where parity holds bit for bit.** For
  accuracy, the official greedy and online code score with the official
  `make_emsemble_score_fn`, and our code scores with our `make_score_fn`. The test
  therefore covers the whole chain: metric, then selection, then acceptance. ROC-AUC
  and log-loss are only close, not bit-identical (a08 documents why). The greedy and
  online tests therefore use a Brier score shared by both sides as their continuous
  score. It is used as is, and also quantized to force tie-breaks with different
  individual scores.
* **The simulation follows the official epoch loop** (`project/tabpack.py` around the
  `update_online_ensembles` call). Running members are evaluated on val/test every
  epoch. Members that stop leave the running predictions in the same epoch and join
  the finished pool with their best step and their train/val/test predictions, so the
  official code drops the extra 'train' part. Finished predictions are `{}` until
  the first member stops, like our `FinishedPool`. Once all members have stopped,
  running predictions have 0 rows. Stopped ensembles are no longer updated, in both
  implementations, as in `update_online_ensembles`.
* **Separate buffers.** Each implementation gets its own numpy and tensor copies of
  every input, so aliasing cannot hide or cause a difference.
* **Coverage assertions.** Every test counts the situations it exercises and asserts
  minimums. A change to the generators that made the comparison trivial would
  therefore fail the test instead of passing silently. The seeds are fixed, and every
  assertion message names the score, the case or run, the seed and the settings.
* **Private attributes.** The test reads the stored snapshots from our
  `OnlineGreedyEnsemble._predictions` through one helper, `_stored_predictions`,
  because the public API only exposes averages. It also reads `_score`,
  `_predictions` and `_predictions_torch` from the official class.
* The sklearn report metrics are checked once per run, on the final ensemble.
  Checking them after every accepted update took about 20 s under load.

## Tests

`tools/dev/py -m pytest tests/parity/test_parity_ensemble.py -q`: **11 passed** in
7-10 s. `ruff check` and `ruff format --check` pass on the test file.

## Coordination

* Merged `main` at 3bf81ca, which includes a08, a25, a26 and a27 and the parity
  dependencies. The official `project.tabpack` imports without stubs, so the
  `sys.meta_path` workaround from a26's note (#127) was not needed.
* Board: status #136 (started) and the final `done`. I received a26's finding #127,
  a note on how to call the official update and what to compare, and applied it. I sent
  no findings, because nothing mismatched.

## Open issues

* None blocking. The test depends on one private attribute of a26's class
  (`OnlineGreedyEnsemble._predictions`). If a26 renames it, only the
  `_stored_predictions` helper needs to change.
