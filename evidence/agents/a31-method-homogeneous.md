# a31-method-homogeneous: homogeneous MLP ensemble run

## Summary

I implemented `run(config, output_dir)` in `src/tabpack_repro/methods/homogeneous.py`.
It builds a deep ensemble of `n_models` MLPs with identical hyperparameters and trains
them as one pack. The final prediction is the uniform average of every member's
best-checkpoint probabilities, with no selection. The function writes the common run
report (`methods/report.py`) through a55's `write_run`. 16 CPU tests pass. They run
end to end on a small dataset written to disk in the official format.

## Files

* `src/tabpack_repro/methods/homogeneous.py`: `run` and a private helper
  `_sorted_members`.
* `tests/methods/test_homogeneous.py`: tests.
* `evidence/agents/a31-method-homogeneous.md`: this report.

## Design decisions

* **Pipeline.** The steps are `setup_run(config.seed, config.data, config.training)`
  (a55; seeds the global RNG), then `ModelPack(pack_size=n_models, d_block, n_blocks,
  dropout, activation)`, then `AdamWPack(make_param_groups(model, muon=False), lr,
  weight_decay, beta1, beta2, eps, pack_size)`. After that come `train_pack(...,
  seed=config.seed, online_ensemble=None)` (a23), `average_predictions` (a27) on val
  and test, `compute_metrics` (a08), `base_report` + `write_run` (a55).
* **Scalar hyperparameters.** lr, weight_decay, betas and eps are passed as Python
  floats, so AdamWPack keeps them as floats: the same value for all members, with no
  (K,) tensors. The bias group's `weight_decay: 0.0` comes from `make_param_groups`.
  Every AdamWPack step uses its default `shared_step=True`, as specified in
  docs/EXPERIMENT.md §3.3.
* **Initialization is independent of the device.** The ModelPack is built on the CPU
  right after `setup_run` seeds the RNG, and then moved to `ctx.device`. a30 does the
  same for the MLP, so a given seed gives the same initial weights on CPU and GPU.
* **Batch generator seed.** `train_pack` gets `seed=config.seed`, like a30's
  `torch.Generator` for the plain MLP.
* **Every member is in the ensemble.** Without an online ensemble, `train_pack` runs
  until every member has stopped. `run` checks that the finished ids are exactly
  `0..K-1` and raises `RuntimeError` otherwise. Members, their predictions and the
  ensemble ids/steps are reordered by id: `train_pack` returns them in finishing
  order, and id order makes the report easier to read. The average is taken over the
  id-ordered predictions.
* **Report.** `members[i] = {id, best_step, config: None, metrics: {train, val, test}}`.
  The integrator's task asked for `config: None`, because every member shares the
  top-level `config.model`/`config.optimizer`. The ensemble is `{'ids': [0..K-1],
  'steps': best steps, 'size': K, 'n_unique': K}`. `best_member` comes from a55's
  `best_member`, by val score with the first member winning ties. `metrics` covers
  only val and test, as in the schema. `history`, `n_epochs`, `n_steps` and
  `time_sec` come from `PackTrainResult`. `time_sec` is the training-loop time, which
  matches a30's MLP report. In `history`, `ensemble_val`/`ensemble_test` are None
  because there is no online ensemble.
* **Input validation.** `n_models` must be a positive int (bools and floats are
  rejected). The check runs before any data is loaded or any file is written.

## Tests

The tests write a tiny on-disk dataset to `tmp_path` in the official format:
`info.json`, float32 `x_num.npy` with 4 columns, `x_cat.npy` with string categories,
int64 `y.npy`, and `splits/default/{train,val,test}.npy` with 320/160/160 rows. The
config uses `n_models=3`, `d_block=16`, `n_blocks=1`, `patience=2`, device cpu and
no AMP. The tests call the real a07/a12/a15/a17/a23/a28/a55 code, with no fakes. A
spy wraps `homogeneous.train_pack` to record the setup it receives and the
`PackTrainResult` it returns.

* Report top-level keys, schema version, method, dataset name, seed, config and
  env; the val/test metric keys; score == accuracy.
* Members sorted by id, with `config` None, train/val/test metrics and a best_step in
  `(0, n_steps]`. The ensemble dict is exact, and best_member is the first max-val
  member.
* The history has one row per epoch, the last row has `n_running == 0` and
  `n_finished == K`, and no ensemble scores. The last epoch equals the last best
  epoch + patience + 1.
* Files: report.json round-trips to the returned report. predictions.npz holds float32
  val/test arrays with probabilities in [0, 1], and `compute_metrics` on them gives
  the reported metrics. config.toml is present.
* The setup passed to train_pack: an AdamWPack, float lr/wd in every group, no Muon
  group, wd in {0.0, config wd}, every parameter in exactly one group, identical
  n_blocks/dropout/d_block for all members, `online_ensemble=None`, and the training
  settings and seed forwarded.
* The final prediction equals the mean of all K members' predictions (atol 1e-6).
* Members have pairwise different predictions.
* The ensemble accuracy is at least the mean member accuracy - 0.025 (4 rows) on val
  and test. The ensemble log-loss is at most the members' mean log-loss; this holds
  exactly by Jensen's inequality. The test accuracy is above 0.75.
* The same seed gives an identical report (wall-clock fields excluded) and bitwise
  identical predictions. A different seed gives different predictions.
* Multiclass with 3 classes and K=2: predictions have shape (N, 3), rows sum to 1,
  and they equal the member average. K=1: the ensemble metrics equal the member's.
* Bad `n_models` (0, -1, 2.0, True) raises ValueError, and nothing is written.

Command: `tools/dev/py -m pytest tests/methods/test_homogeneous.py -q`, which gives
`16 passed`. `ruff check` and `ruff format --check` pass on both owned code files.

**Smoke run on real Churn.** I ran `configs/churn/homogeneous.toml` (K = 16,
3 x 384, seed 0) with `device = 'cpu'` in float32, so the results are not the
experiment's GPU/bfloat16 numbers. The run took 29 epochs (725 steps), with 132 s of
training time on CPU.

| | Val acc | Test acc | Test ROC-AUC | Test log-loss |
| :-- | :-- | :-- | :-- | :-- |
| Uniform ensemble | 0.8644 | 0.8595 | 0.8550 | 0.3451 |
| Best member by val (id 7) | | 0.8560 | 0.8512 | 0.3480 |

The members' test accuracy has a mean of 0.8567, a minimum of 0.8525 and a maximum
of 0.8595. On this seed, the ensemble matches the best member on test and beats the
member mean by +0.28 pp.

## Coordination

* Posted `status` #102 (start) and #135 (implementation committed; tests waiting for
  a23/a28), then `done`.
* Merged `main` (three times), `feat/a15-optim-adamw`, `feat/a55-method-common`,
  `feat/a17-optim-groups`, `feat/a28-config` and `feat/a23-train-trainer`, each after
  its owner posted `done`.
* Before a23 and a28 were done, I ran the tests against a scratch overlay (read-only
  copies of their in-progress files, outside the repo) and committed nothing that
  depended on them.
* I received no questions or findings.

## Open issues

* None in this module. Report-shape choices other agents may rely on: members and
  ensemble ids are in id order, not finishing order, and `members[*].config` is None
  (a30's MLP fills it with `{model, optimizer}`). a36/a37 should read member
  hyperparameters from the top-level `config` for this method.
