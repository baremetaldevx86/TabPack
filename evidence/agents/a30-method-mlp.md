# a30-method-mlp: ordinary MLP baseline

## Summary

Implemented `tabpack_repro.methods.mlp.run(config, output_dir)`, the ordinary MLP
baseline of the Churn comparison. It is deliberately independent of the pack
modules: a plain `nn.Sequential` trained with `torch.optim.AdamW`, using only the
shared plumbing of `methods/common.py` (a55) for setup, the report skeleton, the best
member and the run artifacts. A full default-config run on the real Churn data takes
about 25 s on CPU (test accuracy 0.8495, 25 epochs, best at epoch 8) and 9 s on the
GPU with bf16 autocast (test accuracy 0.857, 23 epochs).

## Files

* `src/tabpack_repro/methods/mlp.py`: `run` plus private helpers (`_one_hot`,
  `_encode_inputs`, `_make_model`, `_make_optimizer`, `_loss`, `_predict`, ...).
* `tests/methods/test_mlp.py`: 16 tests.
* `evidence/agents/a30-method-mlp.md`: this report.

## Design decisions

* **Input**: `concat(x_num, one_hot(x_cat))`, computed once per part before training.
  The one-hot is a private reimplementation (not `nn.OneHotEncoding`): a code
  `>= cardinality` is clamped to an extra class whose column is dropped, so unknown
  codes encode to zeros, like the pack model.
* **Model**: `n_blocks x [Linear -> activation -> Dropout]` of width `d_block`, then a
  `Linear` head with 1 output (binclass/regression) or `n_classes` (multiclass).
  Default PyTorch initialization. The activation must be ReLU, GELU or SiLU (the set
  the pack models accept). The model is built on CPU after `setup_run` has seeded the
  global RNGs, then moved to the device, so its initialization does not depend on
  the device.
* **Optimizer**: `torch.optim.AdamW` with two parameter groups: the weight matrices
  (`ndim >= 2`) with `optimizer.weight_decay`, the biases with 0.0.
* **Batches**: one epoch visits every training row once, in the order of
  `torch.randperm(n_train, generator=g)` with `g = torch.Generator(device)` seeded by
  `config.seed`; the last batch may be smaller (epoch size = ceil(n / batch_size), as
  in the pack trainer).
* **Loss**: BCE with logits (binclass), cross-entropy (multiclass), MSE (regression;
  the data pipeline rejects regression anyway). The forward pass runs under the
  `RunContext` autocast; logits are cast to float32 before the loss and before
  sigmoid/softmax.
* **Evaluation and early stopping**: after every epoch, val and test predictions
  (probabilities) are scored with `ctx.score_fns`. Same semantics as the pack
  trainer: the first evaluation always improves, later ones only with a strictly
  higher val score; stop when the consecutive non-improving evaluations exceed
  `patience` (-1 disables) or when `max_epochs` epochs are done (-1 disables).
  `patience=-1` together with `max_epochs=-1` would never stop and raises
  `ValueError` before any work is done.
* **Best state**: on each improvement the model `state_dict` is deep-copied and the
  val/test predictions of that epoch are kept. The final `metrics`,
  `predictions.npz` and the member's val/test metrics are exactly those best-epoch
  predictions; the best state is restored once at the end to compute the train
  predictions for the member's train metrics (no per-epoch train evaluation).
* **Evaluation batches**: `eval_batch_size`, halved on CUDA OOM.
* **Report** (schema of `methods/report.py`, no extra top-level keys): `method='mlp'`,
  `members=[{'id': 0, 'best_step', 'config': {'model': ..., 'optimizer': ...},
  'metrics': {'train', 'val', 'test'}}]` (member config in TabPack's per-member
  format), `ensemble=None`, `best_member` from `common.best_member`,
  `n_epochs`, `n_steps`, `time_sec` (training loop incl. per-epoch evaluation), and
  `history` with one record per epoch: `epoch` (1-based), `step` (optimizer steps so
  far), `time` (seconds since the start of training), `train_loss` (mean loss over
  the epoch's training rows), `val_score`, `test_score`.

## Tests

```
tools/dev/py -m pytest tests/methods/test_mlp.py -q   ->  16 passed in 6.6s (CPU)
tools/dev/py -m pytest tests/methods -q               ->  50 passed (with a55's tests)
tools/dev/py -m ruff check / ruff format --check      ->  clean on both owned files
```

The tests write a 700-row learnable dataset in the official on-disk format (string
categories plus a two-valued numerical column that the pipeline turns into a
category) to `tmp_path`, set `config.data.path` to it, and run with `d_block=16`,
`n_blocks=1`, `patience=2`, `batch_size=64` on CPU. They check:

* the three files, `report.json` equal to the returned report, the exact set of
  report keys and the common fields;
* the single member, `metrics` = the member's val/test metrics, `best_member`,
  accuracy > 0.6 on val and test, and metrics recomputed from `predictions.npz`
  (float32, shape (N,), in [0, 1]);
* `config.toml` round trips through `load_config`;
* the history (keys, 1-based epochs, steps = epoch x ceil(400/64), monotonic time)
  and a replay of the stopping rule on the recorded val scores: the run stops exactly
  when the consecutive bad evaluations exceed the patience, the best epoch is the
  first epoch with the maximal val score, and the final metrics equal its record;
* `max_epochs=3`, `patience=0`, and `ValueError` for `patience=max_epochs=-1`;
* bitwise-identical predictions, history and member metrics for the same seed, and
  different predictions for another seed;
* multiclass (3 classes: probabilities of shape (N, 3) summing to one);
* the RunContext autocast is entered once per training step and per evaluation
  batch;
* one-hot encoding of unknown codes as zeros, model structure, AdamW param groups,
  rejected activation;
* a 2-epoch smoke run on the real Churn data (`@pytest.mark.data`, skipped when the
  data is missing).

Manual runs (not tests) of the default `configs/churn/mlp.toml` settings on the real
Churn data, seed 0: CPU float32, 25 epochs (best at epoch 8), test accuracy 0.8495,
about 25 s; GPU (RTX 5050, bf16 autocast), 23 epochs, test accuracy 0.857, about 9 s.

## Coordination

* Posted `status` #90 (start) and #130 (implementation done, waiting for a28).
* Merged `main` (twice), then the finished branches `feat/a07-data-pipeline`,
  `feat/a55-method-common` (includes a29; merged again after its update) and
  `feat/a28-config`. Until a28 landed, the tests were run locally with an
  uncommitted shim for `dump_config`/`load_config`; the committed tests use only the
  real modules.
* No questions or findings were addressed to a30.

## Open issues

* None known. The MLP uses PyTorch's default `nn.Linear` initialization, which is the
  same distribution as `LinearPack` (a09), so the baseline and the packs start from
  equally distributed weights (not the same draws).
