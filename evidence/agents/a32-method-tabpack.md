# a32-method-tabpack: reduced heterogeneous TabPack

## Summary

Implemented `tabpack_repro.methods.tabpack.run(config, output_dir)`, the reduced
heterogeneous TabPack method. It follows the setup in the official
`project/tabpack.py::main()`:

1. Member configs come from `config.configs` (deep copy; `len` must equal
   `n_models`) or from `sample_configs(config.space, n_models, seed=config.seed)`.
2. The configs are transposed into per-member lists, like the official
   `transpose_list_of_dicts` + `pop` + `assert not configs_T`. Every member must
   provide `model.{n_blocks, dropout}` and `optimizer.{lr, weight_decay, muon_lr}`.
   Any other sampled key or section raises `ValueError` ("not used: ...").
3. `ModelPack(d_block fixed, n_blocks list, dropout list, activation)` is built on the
   run device. `MuonAdamWPack(make_param_groups(model, muon=True), lr/weight_decay/
   muon_lr lists, fixed settings from config.optimizer, pack_size=K)`.
4. `OnlineGreedyEnsemble(score_fn=ctx.score_fns['val'], task, max_ensemble_size,
   patience)`, then `train_pack(..., seed=config.seed, online_ensemble=...)`.
5. The final prediction is the online ensemble's averaged val/test prediction.
   `metrics` is computed with `compute_metrics` on exactly the arrays written to
   `predictions.npz`.
6. The report follows the `methods/report.py` schema: finished `members` with their
   configs, `ensemble` {ids, steps, size, n_unique}, `best_member` by val, history,
   `n_epochs`/`n_steps`/`time_sec`, and env. `write_run` writes report.json,
   predictions.npz and config.toml.

## Files

* `src/tabpack_repro/methods/tabpack.py`: implementation.
* `tests/methods/test_tabpack.py`: 22 tests.
* `evidence/agents/a32-method-tabpack.md`: this report.

## Design decisions

* **The report records all K member configs in `member_configs` (index = member id).**
  The online ensemble pools running members at their latest epoch. Training stops
  when the ensemble's patience runs out, and members still running at that point are
  dropped (the official Churn run finished 57 of 64). So an ensemble id may have no
  entry in `members`. The official conservative script reads configs from
  `experiments.json`, which lists every member, for the same reason.
  `member_configs` plays that role, and a33 was told to use it (#126, #132).
* The other additions to the schema are `n_models`, `n_finished`, the full
  `online_ensemble` report (ids, steps, size, n_unique, score_val, metrics), and
  `best_member['config']`. They are additive and the schema keys keep their meaning.
* **Validation happens before any data is loaded.** This covers `n_models >= 1`, the
  explicit-config count, missing or unused per-member keys, and the online-ensemble
  variant. Sampling uses optuna's own seeded RNG, so doing it before `setup_run`
  (which seeds the global RNGs) does not change reproducibility. The model is still
  initialized after `seed_everything(seed)`, as in the official main().
* **Only the implemented online-ensemble variant is accepted.** Anything other than
  `type='greedy'`, `update_type='latest'` and
  `include_current_ensemble_in_pool=True` raises `ValueError` instead of being
  silently ignored.
* Per-member dropout and optimizer values are cast to `float`. `n_blocks` is passed
  unchanged, and `ModelPack` validates it.
* `report['metrics']` comes from the saved predictions, so it matches
  predictions.npz. It equals `online_ensemble.metrics` up to float rounding, since
  the online report averages the snapshots in numpy.
* The report written by `run()` always has `method='tabpack'`. a33 rewrites it to
  `tabpack-conservative-seed` for its per-seed runs, which was their default plan
  (#115/#132). The public signature is unchanged.

## Tests

`tools/dev/py -m pytest tests/methods/test_tabpack.py -q` -> **22 passed** in about 7 s on CPU (with
`tests/methods tests/training/test_trainer.py`: 70 passed, 1 GPU test skipped).

The tests use a small synthetic dataset written to disk in the official format
(info.json, x_num float32, x_cat strings, y int64, splits/default/*.npy). The run
uses 4 members, d_block 16, n_blocks in 1..2, patience 2, max_epochs 8, online
patience 2 and a maximum ensemble size of 3, on CPU with no AMP. Coverage:

* all three files exist, schema keys and types are present, and report.json equals
  the returned report;
* predictions.npz has keys val/test, float32 probabilities, and metrics that match
  `compute_metrics`;
* member configs equal `sample_configs(space, 4, seed=0)` and all differ; each
  member's config equals `member_configs[id]`;
* per-member hyperparameters reach ModelPack, MuonAdamWPack (one Muon group per
  block), OnlineGreedyEnsemble and train_pack (checked with spies around the real
  classes);
* ensemble fields are consistent: size <= 3, ids are a subset of member ids, and the
  fields match the online report;
* the ensemble's val score is >= the best member's val score, and best_member is the
  val argmax;
* a second run is identical: metrics, members, ensemble, history without time, and
  predictions bit for bit;
* explicit `configs` work and the input is not mutated; an `n_models` mismatch,
  extra/missing keys or sections, an unsupported online ensemble, and
  `n_models=0` raise before the data is loaded;
* `_split_member_configs` transposes correctly, requires the same keys in every
  member, and rejects non-dicts; sampling depends on the seed.

Before a23 posted `done`, the same 22 tests also passed in two scratch setups (never
committed): a pytest plugin with a minimal fake trainer, and a23's committed
`trainer.py` loaded from its branch without merging it.

Smoke run on real Churn (scratch script, not committed; CPU, no AMP, 8 members,
d_block 64, max_epochs 6, max ensemble size 4): 8 members finished, the ensemble was
ids [2, 2, 7] at steps [75, 100, 100], val 0.8644 / test 0.8575, best member val
0.8638 / test 0.8555, 37 s.

## Coordination

* Posted status #101 and #126 (report extensions, for a33). Answered a33's #115 in
  #132: `member_configs[id]` holds exactly the dicts that `TabPackConfig.configs`
  accepts, member ids are 0..K-1 in config order, and ensemble ids may repeat.
* Merged peer branches after their owners posted `done`: main, a55, a24, a25, a16,
  a17, a26, a28, a23 (after #153). No conflicts in owned files.

## Open issues

* None blocking. `member_configs`, `n_models`, `n_finished`, `online_ensemble` and
  `best_member.config` are additive report keys. a36 (summarize) and a54
  (contract audit) may want to document them in `methods/report.py`, which the
  integrator owns.
* `time_sec` is the training-loop time from `PackTrainResult` (data loading and
  report writing are excluded), as with the other pack methods.
