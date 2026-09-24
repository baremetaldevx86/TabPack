# Experiment specification: TabPack vs MLP baselines on Churn

This document specifies the one experiment this repository reproduces from
*TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning* (Gorishniy et
al., ICML 2026, [arXiv:2607.05380](https://arxiv.org/abs/2607.05380)). It covers the
question, the data, the three methods, the protocol and the commands. The design was
fixed at checkpoint 00 ([`evidence/checkpoints/00-skeleton.md`](../evidence/checkpoints/00-skeleton.md)),
together with the user and before any result existed.

Sources of truth, in order of precedence:

1. [`src/tabpack_repro/config.py`](../src/tabpack_repro/config.py): every default value.
   If this document and `config.py` disagree, `config.py` wins and this document has a bug.
2. The method contracts in [`src/tabpack_repro/methods/`](../src/tabpack_repro/methods/)
   (`mlp.py`, `homogeneous.py`, `tabpack.py`, `conservative.py`, `report.py`).
3. The official Churn config `experiments/tabpack/churn/main/config.json` and the
   official README (conservative protocol) at commit `05a89e2` of
   [yandex-research/tabpack](https://github.com/yandex-research/tabpack). We use the
   official code only for comparison and never copy it.

## 1. Research question

> Out of the box on Churn, does a heterogeneous hyperparameter ensemble (TabPack,
> reduced) beat an untuned single MLP and an untuned homogeneous MLP ensemble?

"Out of the box" means that no method has hyperparameters tuned on Churn. The only
data-driven decisions allowed are the ones each method makes by itself during one run,
using the validation split: early stopping for every method, and also snapshot and
ensemble-member selection for TabPack.

Hypotheses, compared on test accuracy:

| Id | Claim | What changes between the two arms |
| :-- | :-- | :-- |
| H1 | Homogeneous ensemble > ordinary MLP | Ensembling alone (same recipe, K = 16) |
| H2 | Reduced TabPack > homogeneous ensemble | Hyperparameter diversity, optimizer (Muon+AdamW) and greedy selection, all at once (see section 6) |
| H3 | Reduced TabPack > ordinary MLP | Everything above |

This experiment does not test:

* tuned baselines,
* datasets other than Churn,
* the full-size TabPack (64 models),
* the paper's efficiency claims beyond the wall-clock times we record.

## 2. Dataset

We use Churn from the official TabPack data bundle (`tabpack-data.tar.gz`, Hugging Face
dataset `Yura52/tabpack-data`, sha256 pinned in `tabpack_repro.data.download`). It has
bank-customer records with a binary "customer left" label.

| Property | Value |
| :-- | :-- |
| Rows | 10,000 |
| Split (`splits/default`, shipped with the data, fixed) | train 6,400 / val 1,600 / test 2,000 (disjoint) |
| Features | 7 numerical (float32), 3 binary (0/1), 1 categorical (3 values) |
| Task | Binary classification (`info.json`: `binclass`) |
| Metric | Accuracy (`info.json`: `score = accuracy`), higher is better |
| Positives | train 1,304 / val 326 / test 407 (20.4 % on every part) |
| Majority-class accuracy | train 79.625 %, val 79.625 %, test 79.65 % |
| Missing values | None |

### Preprocessing

The steps below run in the official order (`lib.data.build_dataset`, implemented in
`tabpack_repro.data.pipeline.build_dataset`). The config is `DataConfig`: `num_policy =
'noisy-quantile'`, `extract_bin_from_num = true`, `bin_policy = 'convert-to-cat'`,
`cat_policy = 'ordinal'`, `seed = 0`.

1. **Load** `x_num (10000, 7)`, `x_bin (10000, 3)`, `x_cat (10000, 1)` (strings) and
   `y (10000,)` (int64 in {0, 1}), then index them with the `default` split.
2. **Extract binary columns from numerical ones.** A numerical column with exactly two
   unique values, counted over all parts, would move to the binary features. On Churn
   this changes nothing: the smallest number of unique values in any numerical column
   is 4. All 7 numerical columns stay.
3. **Numerical features ("noisy quantile").** We fit
   `QuantileTransformer(n_quantiles = max(min(6400 // 30, 1000), 10) = 213,
   output_distribution = 'normal', subsample = 1e9, random_state = 0)` on
   `x_train + N(0, 1e-5)`, with the noise drawn from `np.random.RandomState(0)`. We
   then transform the clean train, val and test parts, set NaN to 0 and drop columns
   that are constant on train (none on Churn). The result is float32.
4. **Binary to categorical.** The 3 binary columns are cast to the categorical dtype
   (strings such as `'0.0'`) and appended after the categorical column, which gives 4
   categorical columns.
5. **Ordinal encoding.** An `OrdinalEncoder` is fitted on train with sorted categories.
   The train cardinalities are `[3, 2, 2, 2]`. A category that is unknown in val or test
   would get the code `train_max + 1`, but none occurs on Churn.
6. **Model input.** The model sees `concat(x_num, one_hot(x_cat))`, which is
   7 + (3 + 2 + 2 + 2) = **16 input features**. An unknown code would one-hot encode to
   all zeros.
7. **Labels and predictions.** The model outputs one logit per example. Training uses
   BCE-with-logits. Predictions are sigmoid probabilities. Accuracy rounds the
   probability half to even, so p = 0.5 counts as class 0, matching the official code.

The data seed is always 0 and does not depend on the run seed, as in the official code.
Every run of every method therefore sees exactly the same preprocessed tensors.

## 3. Methods

### 3.1 Settings shared by all three methods

| Setting | Value | Config field |
| :-- | :-- | :-- |
| Data and preprocessing | Section 2 | `data.*` |
| Batch size | 256, so 25 steps per epoch (6,400 / 256) | `training.batch_size` |
| Epoch limit | None (-1): early stopping only | `training.max_epochs` |
| Early stopping | Patience 16 on val accuracy with strict improvement. A model stops after 17 consecutive epochs without improvement and keeps its best-val checkpoint. | `training.patience` |
| Evaluation | val and test after every epoch, batch size 32,768 (each whole part in one batch) | `training.eval_batch_size` |
| Mixed precision | bfloat16 autocast on CUDA only; CPU runs are float32 | `training.amp_dtype` |
| Device | `auto` (CUDA if available) | `training.device` |
| Width, activation | `d_block = 384`, ReLU | `model.*` / top level |
| Seed | Controls initialization, dropout masks, batch order and, for TabPack, config sampling (`seed_everything(seed)` plus a device `torch.Generator`) | `seed` |

Every member of a pack sees each training row once per epoch, in its own random order
(`torch.rand((K, N)).argsort(1)`). The pack loss is the **sum** of the members'
mean losses, so the gradient scale does not depend on K.

### 3.2 Ordinary MLP (`method = 'mlp'`, `MLPMethodConfig`)

The MLP is a plain `nn.Sequential` trained with `torch.optim.AdamW`. It does not use the
pack code on purpose, so it also serves as a sanity reference.

| Hyperparameter | Value |
| :-- | :-- |
| Architecture | 3 x [Linear -> ReLU -> Dropout] with `d_block = 384`, then a linear head (16 -> 384 -> 384 -> 384 -> 1), 302,593 parameters |
| Dropout | 0.1 |
| Optimizer | AdamW, lr 1e-3, betas (0.9, 0.999), eps 1e-8 |
| Weight decay | 1e-4 on weight matrices, 0 on biases |
| Batch size, patience, AMP | 256, 16, bfloat16 (section 3.1) |
| Final prediction | The single model at its best-val epoch; the test accuracy is taken at that epoch |
| Tuning | None: fixed defaults, not tuned on Churn |

### 3.3 Homogeneous MLP ensemble (`method = 'homogeneous'`, `HomogeneousEnsembleConfig`)

This is a deep ensemble of K identical-hyperparameter MLPs, trained together as one
pack (`ModelPack` + `AdamWPack`). The members differ only in initialization, dropout
masks and batch order.

| Hyperparameter | Value |
| :-- | :-- |
| Members K | 16 (= TabPack's maximum ensemble size, see below) |
| Member architecture | Same as the ordinary MLP: 3 x 384, ReLU, dropout 0.1 |
| Optimizer | AdamWPack, lr 1e-3, betas (0.9, 0.999), eps 1e-8, one shared step counter |
| Weight decay | 1e-4 on weight matrices (backbone and head), 0 on biases (`make_param_groups(muon=False)`) |
| Early stopping | Per member, patience 16. The run ends when every member has stopped. |
| Final prediction | **Uniform** average of all 16 members' best-checkpoint probabilities. There is no selection and no weighting. |
| Also reported | Each member's own train, val and test metrics |
| Tuning | None |

K = 16 caps the inference-time ensemble size at the same number as TabPack's
`max_ensemble_size`. This makes the prediction budgets comparable.

### 3.4 Reduced heterogeneous TabPack (`method = 'tabpack'`, `TabPackConfig`)

TabPack samples one hyperparameter config per member. It trains all members as one pack
with per-member hyperparameters, and during training it keeps a greedy ensemble chosen
on val.

**Fixed settings**

| Setting | Value |
| :-- | :-- |
| Sampled members `n_models` | **32** |
| Width, activation | `d_block = 384` (not sampled), ReLU |
| Optimizer | **MuonAdamWPack** (Muon for backbone weights, AdamW for the rest) |
| Adam betas, eps | (0.9, 0.999), 1e-8 |
| Muon | momentum 0.95, Nesterov, 5 Newton-Schulz steps (quintic coefficients 3.4445, -4.7750, 2.0315) |
| Muon scale | `sqrt(max(1, out / in))` per block: sqrt(384 / 16) ≈ 4.90 for block 0, 1 for the other blocks |
| Shared step | `true` (one step counter for all members) |
| Batch size, member patience, AMP | 256, **16**, bfloat16 |
| Online ensemble | Greedy, `update_type = 'latest'`, current ensemble included in the pool, **patience 32**, **`max_ensemble_size = 16`** |

**Search space** (identical to the official Churn space)

| Hyperparameter | Distribution | Range | Where it acts |
| :-- | :-- | :-- | :-- |
| `model.n_blocks` | Uniform integer | {1, 2, 3, 4} | Backbone depth (6,913 to 450,433 parameters per member) |
| `model.dropout` | 0 with probability 1/2, else uniform | [0, 0.5] | After every block |
| `optimizer.lr` | Log-uniform | [1e-4, 5e-3] | AdamW part: head weight and all biases |
| `optimizer.weight_decay` | Log-uniform | [1e-3, 1.0] | Decoupled decay of the head weight (`p *= 1 - lr * wd`) and of the backbone weights (`p *= 1 - muon_lr * wd`); biases 0 |
| `optimizer.muon_lr` | Log-uniform | [1e-3, 1e-1] | Muon part: every backbone block weight, including the first layer |

The official format is `['_tune_', 'int', 1, 4]`,
`['_tune_', '?uniform', 0.0, 0.0, 0.5]` and `['_tune_', 'loguniform', lo, hi]`.
Configs are drawn with `optuna.samplers.RandomSampler(seed = run seed)`, with one
`study.ask()` per member in member order, so a run seed fixes all 32 configs.

**Parameter groups, all methods**

| Parameters | MLP / homogeneous | TabPack |
| :-- | :-- | :-- |
| Backbone block weights | AdamW, lr 1e-3, wd 1e-4 | Muon, per-member `muon_lr` and `weight_decay` |
| Head weight | AdamW, lr 1e-3, wd 1e-4 | AdamW, per-member `lr` and `weight_decay` |
| Biases | AdamW, lr 1e-3, wd 0 | AdamW, per-member `lr`, wd 0 |

**Training loop and online ensemble** (official semantics, `training.trainer.train_pack`
and `ensembles.online.OnlineGreedyEnsemble`):

1. After every epoch, evaluate every running member on val and test and update its best
   checkpoint. Members that hit their patience (17 epochs without strict improvement)
   stop, move to the finished pool at their best epoch, and are removed from the pack
   and from the optimizer.
2. Update the online ensemble. The pool contains the current ensemble's entries, the
   finished members at their best epoch, and the running members at their **latest**
   epoch (snapshots). Each entry is an (id, step, prediction) triple, so one id can
   appear more than once.
3. Run greedy forward selection (Caruana et al., 2004; start from the best single
   entry; add the entry that most improves the uniform average; stop when nothing
   strictly improves or 16 entries are selected) on the pool's **val** probabilities.
   Accept the candidate ensemble only if its val accuracy is strictly better than the
   current one, and then reset the ensemble patience. Otherwise the ensemble patience
   decreases by 1.
4. The run ends when no member is running, or when the ensemble has gone 33
   consecutive updates without improving (patience 32). Members that are still running
   at that point are dropped. This is also how the official run ends: 57 of its 64
   members finished.

**Final prediction:** the uniform average of the online ensemble's entries. **Also
reported:** every finished member's config and metrics, the ensemble's ids, steps, size
and number of unique members, and the best single member by val.

## 4. Reduced vs official TabPack

| Setting | Official (`experiments/tabpack/churn/main`) | This reproduction |
| :-- | :-- | :-- |
| `n_models` | 64 | **32** |
| `online_ensembles.greedy.options.max_ensemble_size` | 32 | **16** |
| `seed` (main run) | 0 | 0 (plus 1-4 for the single-run protocol) |
| Batch size / `n_epochs` / member patience | 256 / -1 / 16 | 256 / -1 / 16 |
| `amp_dtype` | bfloat16 | bfloat16 |
| Model | ReLU, `d_block` 384 | ReLU, `d_block` 384 |
| Optimizer | MuonAdamWPack, `shared_step = true` | MuonAdamWPack, `shared_step = true` |
| Muon defaults | momentum 0.95, Nesterov, 5 NS steps | momentum 0.95, Nesterov, 5 NS steps |
| Online ensemble | greedy, `latest`, current ensemble in pool, patience 32 | greedy, `latest`, current ensemble in pool, patience 32 |
| Sampler / search space | RandomSampler, space of section 3.4 | Same |
| Data config | noisy-quantile, extract bin, convert-to-cat, ordinal | Same |
| Conservative evaluation | 5 seeds (0-4), selected configs only | Same |
| Implementation | Official code | From-scratch reimplementation, checked by `tests/parity/` |
| Hardware | NVIDIA A100-SXM4-80GB | NVIDIA GeForce RTX 5050 Laptop GPU |

Only the two values in **bold** differ. The user's brief asked for a *reduced*
TabPack, and halving both keeps the pool-to-cap ratio of the official run (2:1).
Neither cap is likely to bind on Churn. The official final ensemble had only 6 entries
(5 unique members), well below both 32 and 16. The larger effect is a smaller pool
(32 instead of 64 sampled configs), which may lower the ensemble's quality a little.

**Official reference numbers** (from the reports shipped with the official code;
`tabpack-repro reference` extracts them to `results/reference/churn_official.json`):

| Official Churn result | Test accuracy |
| :-- | :-- |
| Main run (seed 0, 64 models): online greedy ensemble | 85.70 % (val 87.625 %) |
| Main run: best single member by val | 85.60 % (val 87.25 %) |
| Main run: mean of the 57 finished members | 85.46 % |
| Conservative evaluation (5 seeds, 5 selected configs) | **85.75 ± 0.14 %** |

The main run took 17.1 s on the A100. Each conservative seed took 4.7-7.0 s.

## 5. Protocol

### 5.1 Runs

| Method | Seeds | Runs | Output |
| :-- | :-- | :-- | :-- |
| Ordinary MLP | 0, 1, 2, 3, 4 | 5 | `runs/churn/mlp/seed-<s>/` |
| Homogeneous ensemble (K = 16) | 0, 1, 2, 3, 4 | 5 | `runs/churn/homogeneous/seed-<s>/` |
| Reduced TabPack, single run | 0, 1, 2, 3, 4 | 5 | `runs/churn/tabpack/seed-<s>/` |
| Reduced TabPack, conservative | Retraining seeds 0, 1, 2, 3, 4 (configs from TabPack seed 0) | 5 | `runs/churn/tabpack-conservative/seed-<s>/` |

That is 20 training runs in total. Nothing is tuned between runs, and all settings are
the defaults of section 3.

### 5.2 TabPack, single-run protocol (primary)

Each seed is one complete, independent TabPack run: fresh config sampling, fresh
initialization and fresh batch order. The reported number is the test accuracy of the
run's **final online ensemble**. This matches how a practitioner uses TabPack, and it
gives TabPack the same protocol as the baselines: one training run per seed, with every
choice made on val. Its seed-to-seed std therefore includes the variance from config
sampling.

### 5.3 TabPack, conservative protocol (secondary, paper-comparable)

This is the paper's protocol, as used in the official
`scripts/run_tabpack_experiment.py --eval-online-ensembles greedy
--eval-online-ensembles-n-seeds 5`:

1. **Selection.** Take the main run, which is TabPack single run **seed 0**. The
   selected configs are the sorted **unique** ids of its final online ensemble.
2. **Retraining.** For each seed s in 0-4, train a new TabPack pack from scratch that
   contains only the selected configs (`configs = selected`,
   `n_models = len(selected)`), with the same online-ensemble settings (greedy, at most
   16 entries, patience 32). The seed changes initialization, dropout and batch order,
   but not the configs.
3. **Report** the online-ensemble test accuracy of each retraining run, as mean ± std
   over the 5 seeds. The aggregate is in `runs/churn/tabpack-conservative/report.json`.

This protocol is "conservative" because the reported score does not come from the exact
trained weights that won the val-based selection. Only the configs carry over, and new
weights are trained for them. Its std covers only training noise for one fixed config
set. It excludes the config-sampling variance, so it is not directly comparable with
the baselines' std.

### 5.4 Validation and test discipline

* **Val (1,600 rows) is the only split used for decisions.** It drives early stopping
  (all methods), best-checkpoint choice (all methods), snapshot and greedy member
  selection and ensemble stopping (TabPack), and the choice of the configs to retrain
  (conservative).
* **Test (2,000 rows) is used only for reporting.** Methods compute test predictions
  alongside val so that they can report them at the val-chosen point. No decision reads
  a test score, and nothing is re-run or changed after looking at test results.
* The hyperparameter defaults, K, the reductions and the seeds were all fixed at
  checkpoint 00, before any run.

### 5.5 Reported quantities

For every run, `report.json` (schema in `methods/report.py`) contains val and test
accuracy, ROC-AUC and log-loss of the final prediction; per-member metrics; the
ensemble composition; the number of epochs and steps; the wall-clock time; and the
device. `tabpack-repro summarize` aggregates each method over seeds as mean, sample std
(ddof = 1), min, max and n. For ensembles it also gives the mean individual-member test
accuracy and, for TabPack, the mean ensemble size.

### 5.6 Decision rule (declared before the runs)

With 5 seeds per arm we do not claim statistical significance. For two arms A and B,
let Δ = mean_A - mean_B (test accuracy) and SE = sqrt(s_A² / 5 + s_B² / 5). We say
**"A beats B"** only if Δ ≥ 2 · SE **and** Δ ≥ 0.10 pp (2 test errors). Otherwise the
comparison is **inconclusive**. The primary comparisons (H1-H3) use TabPack's
single-run numbers. We also report the conservative number and compare it with the
official 85.75 ± 0.14 %.

## 6. Fairness notes and threats to validity

**Untuned baselines, by design.** The question is about out-of-the-box performance, so
the MLP and the homogeneous ensemble use common defaults (3 x 384, dropout 0.1,
lr 1e-3, wd 1e-4) that were never tuned on Churn. A tuned MLP would probably do better.
This experiment says nothing about TabPack against *tuned* baselines. The comparison
also has an asymmetry: TabPack's search space was designed by its authors across many
datasets, while the baseline defaults are ours. Neither side, however, saw Churn's
val or test data before the runs.

**Muon vs AdamW confound, and why we accepted it.** TabPack trains with Muon+AdamW
while the baselines use AdamW. We chose this for three reasons:

* Muon+AdamW is part of TabPack's published recipe (`optimizer.type = MuonAdamWPack`
  in the official Churn config), and its search space contains `muon_lr`. TabPack
  with AdamW only would be a variant the authors did not ship, with a search space that
  was not designed for it.
* AdamW is the standard optimizer for an ordinary MLP.
* The homogeneous ensemble must use exactly the MLP's recipe, so that H1 isolates the
  effect of ensembling.

This decision was made with the user at checkpoint 00. The cost is that H2 changes
three things at once: hyperparameter diversity, the optimizer, and val-based greedy
selection instead of uniform averaging. A TabPack win therefore supports "TabPack as
shipped beats untuned baselines", but it does **not** show that the gain comes from
heterogeneity. Two ablations would separate the effects: a homogeneous ensemble trained
with Muon+AdamW at a fixed `muon_lr`, and TabPack with AdamW only. The current configs
cannot express either one (`TabPackConfig.optimizer` is Muon+AdamW only and
`HomogeneousEnsembleConfig.optimizer` is AdamW only), so both are out of scope.

**Unequal use of the validation set.** TabPack uses val for early stopping, snapshot
choice, member selection and ensemble stopping. The baselines use it only for early
stopping and checkpoint choice. TabPack's **val** scores are therefore optimistically
biased, and a val-test gap is expected. Test scores are not affected, so all
comparisons use test.

**Compute is not matched.** Per run, the trained parameters are:

* MLP: 0.30 M.
* Homogeneous ensemble: 16 x 0.30 M ≈ 4.8 M.
* TabPack: 32 members of 0.23 M on average ≈ 7.3 M, stopped by the ensemble's
  patience.

At inference, the MLP uses 1 model, the homogeneous ensemble 16, and TabPack at most
16 entries (the official run used 6). We record wall-clock time per run so that
cost can be read next to accuracy.

**Small test set.** With 2,000 test rows, **one test error is 0.05 pp**, and the
binomial standard error of an accuracy near 86 % is about 0.78 pp. Differences between
methods on Churn are expected to be a few tenths of a point, that is, a handful of
test rows. All methods are scored on the same test rows, so the test-set sampling error
is shared and does not affect the ranking the way it affects absolute numbers. Still,
the result holds for this split only. Val granularity is 1 / 1,600 = 0.0625 pp, so ties
on val are common, and the strict-improvement rules decide them.

**Seed variance.** Five seeds give a rough std estimate: for n = 5, the 95 % interval
for σ spans about 0.6 s to 2.9 s. The official conservative std is 0.14 pp, which is
about 3 test errors. We report per-seed values next to the mean and std, and we apply
the conservative decision rule of section 5.6. Seeds are not paired across methods:
the same integer seed drives different random streams in each method.

**Fixed split.** Every run uses the one shipped `default` split. The variance we
report comes from training randomness, not from resampling the data.

**Implementation and hardware.** This is a from-scratch reimplementation. Its
components are checked against the official code in `tests/parity/`, and the official
shipped numbers (section 4) give an external check for TabPack. Differences in hardware
(A100 vs RTX 5050 Laptop) and nondeterministic CUDA kernels under bfloat16 autocast mean
that a rerun with the same seed can differ slightly. CPU runs use float32 and are not
bit-comparable with GPU runs.

**Class imbalance.** About 20 % of rows are positive, and accuracy with a 0.5 threshold
is the official metric. The majority class alone scores 79.65 % on test. ROC-AUC and
log-loss are reported as secondary metrics, but no decision uses them.

## 7. Compute budget (expectations)

These are estimates made before the runs. The measured times are recorded in each
`report.json` (`time_sec`) and in `results/churn/summary.md`.

| Item | Expectation |
| :-- | :-- |
| Machine | 12 CPUs, 9 GiB RAM (shared), NVIDIA GeForce RTX 5050 Laptop GPU, CUDA 12.8 |
| Steps per epoch | 25 (6,400 / 256) |
| Epochs per MLP / homogeneous run | Around 20-60: best epoch plus 17 patience epochs |
| Epochs per TabPack run | Around 40-80: last ensemble improvement plus 33 epochs (the official run took 61 epochs until its last member finished) |
| GPU memory | Under 1 GB per run: parameters, optimizer state, best checkpoints and full-part eval activations of at most 32 x 0.45 M-parameter members |
| Host RAM | Under 2 GB per process. Run the methods **sequentially**, because RAM is shared. |
| Time per run on GPU | Seconds (MLP) to about 1 minute (TabPack); for reference, the official 64-model run took 17 s on an A100 |
| Whole suite on GPU (20 runs + summary) | About 10 minutes |
| Whole suite on CPU (`--device cpu`, float32) | Roughly an order of magnitude slower: about 1-2 hours |
| Disk | About 190 MB for the downloaded bundle (only `data/churn` is extracted); a few MB of reports and predictions |

## 8. How to run

Set up the environment and data once:

```bash
uv sync
uv run python -m tabpack_repro download --name churn     # -> data/churn/
```

Run the whole experiment with one command:

```bash
bash scripts/run_churn.sh
```

`scripts/run_churn.sh` (a wrapper around `scripts/run_churn.py`) runs the steps below
in this order, one at a time. The manual equivalent is:

```bash
# 1. Baselines and TabPack single runs, seeds 0-4.
for s in 0 1 2 3 4; do
  for m in mlp homogeneous tabpack; do
    uv run python -m tabpack_repro run \
      --config configs/churn/$m.toml --seed $s --output runs/churn/$m/seed-$s
  done
done

# 2. Conservative protocol: configs selected by TabPack seed 0, retrained with seeds 0-4.
uv run python -m tabpack_repro conservative \
  --source-run runs/churn/tabpack/seed-0 --n-seeds 5 \
  --output runs/churn/tabpack-conservative

# 3. Official reference numbers (needs the official clone, $TABPACK_REFERENCE_DIR).
uv run python -m tabpack_repro reference --output results/reference/churn_official.json

# 4. Aggregate over seeds and methods.
uv run python -m tabpack_repro summarize --runs-dir runs/churn --output results/churn
```

Notes:

* `uv run tabpack-repro ...` is the same as `uv run python -m tabpack_repro ...`.
* `--device cpu` (on `run`) forces the CPU.
* `--seed` overrides the `seed` in the config file.
* Step 2 needs `runs/churn/tabpack/seed-0/report.json`. It can be resumed: seeds whose
  `report.json` already exists are skipped.

Outputs:

```
runs/churn/
  mlp/seed-<s>/                 report.json, config.toml, predictions.npz
  homogeneous/seed-<s>/         report.json, config.toml, predictions.npz
  tabpack/seed-<s>/             report.json, config.toml, predictions.npz
  tabpack-conservative/
    report.json                 aggregate: source_run, selected_ids, per-seed val/test scores, mean, std
    seed-<s>/                   report.json, config.toml, predictions.npz
results/churn/
  summary.json, summary.md, summary.csv, figures (PNG)
results/reference/
  churn_official.json           official shipped Churn numbers
```

`report.json` and `config.toml` are committed. `predictions.npz` is git-ignored
(`runs/**/*.npz`), because it holds only the final val and test probabilities and can
be regenerated.

## 9. Results

The integrator fills this section after the runs, from `results/churn/summary.md` and
`results/reference/churn_official.json`. It will hold:

* the main table (test accuracy mean ± std, val accuracy, n seeds, time) for the MLP,
  the homogeneous ensemble, TabPack single run and TabPack conservative;
* the per-seed values;
* the comparison with the official numbers;
* the figures;
* a verdict on H1-H3 under the decision rule of section 5.6.

<!-- RESULTS -->
