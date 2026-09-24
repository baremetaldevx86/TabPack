# a51-bench: pack vs sequential training throughput

## Summary

`scripts/benchmark_pack.py` is a GPU micro-benchmark for TabPack's efficiency claim.
It times one pack of K MLPs (`ModelPack` with `AdamWPack` or `MuonAdamWPack`) against
the same K MLPs trained one at a time as `K=1` packs, for K in {1, 2, 4, 8, 16, 32},
with and without bf16 autocast. It reports median ms/step, samples/s, speedup
(sequential / pack) and peak CUDA memory to `results/benchmark/pack_throughput.json`
and a Markdown table in `results/benchmark/pack_throughput.md`, together with the
environment (GPU name, torch/CUDA/cuDNN versions, git commit). The `--cpu` flag runs
tiny sizes for a smoke test and writes `results/benchmark/pack_throughput_cpu.{json,md}`.

**Headline (RTX 5050 Laptop GPU, torch 2.11 + CUDA 12.8, WSL2; run at commit
`3d693d9`, 89.6 s).** One step of a K = 32 pack costs about as much as one step of
a single MLP. The pack is therefore much faster than training the 32 MLPs one after
another:

| optimizer | AMP | pack ms/step | sequential ms/step | speedup (median) | speedup (min) | pack peak MiB |
| :-- | :-- | --: | --: | --: | --: | --: |
| AdamWPack | fp32 | 10.54 | 342.69 | 32.5x | 30.6x | 275 |
| AdamWPack | bf16 | 8.09 | 208.63 | 25.8x | 27.9x | 270 |
| MuonAdamWPack | fp32 | 22.34 | 275.08 | 12.3x | 12.0x | 239 |
| MuonAdamWPack | bf16 | 22.12 | 339.74 | 15.4x | 15.6x | 233 |

* **Pack step time is nearly flat in K.** A K = 1 step takes 7-10 ms and a K = 32
  step takes 8-22 ms. The speedup grows roughly linearly with K: 1.8-2.7x at K = 2,
  2.2-4.6x at K = 4, 4.5-7.3x at K = 8, 10-20x at K = 16 and 12-33x at K = 32.
* **Why.** At this size a step is bound by kernel launches (about 27 us per CUDA
  launch here, measured), not by compute. Packing replaces K sets of launches with
  one set of batched matmuls.
* **Muon.** Muon's Newton-Schulz iterations on 32 x (384 x 384) matrices add real GPU
  work, so the Muon pack grows from about 9 ms at K = 1 to about 22 ms at K = 32.
  Its speedup (12-15x) is lower than AdamW's (26-33x).
* **bf16 autocast.** It barely changes the time per step, because the step is not
  compute-bound. The autocast casts are extra launches.
* **Memory.** Peak memory grows linearly: 24.7 MiB for one member, 275 MiB for
  K = 32 (fp32, AdamW). That is about 8 MiB per member, of which about 4.6 MiB is
  parameters, gradients and AdamW state. The rest is activations for 256 rows x 384
  features. All sizes fit easily in the 8 GB GPU.
* **Noise.** The machine was shared with about 50 agents' test runs (load average
  4-5 on 12 logical CPUs), and the laptop GPU changes its clocks. Individual rows are
  therefore noisy. For example, AdamW fp32 gives 5.1x at K = 8 but 20.2x at K = 16,
  and the K = 1 rows, where both sides run identical work, deviate from 1.0x by up to
  about 30% (0.46x in one earlier run). Two earlier full runs (a few minutes before
  this one, at higher load, with 7 and 11 repeats) gave K = 32 speedups of
  37.2x/32.7x/19.9x/22.5x and 37.9x/19.6x/28.8x/23.5x for AdamW fp32/bf16 and Muon
  fp32/bf16, with the same trend.

The full tables are in `results/benchmark/pack_throughput.md`.

## Files

* `scripts/benchmark_pack.py`: the benchmark (CLI, timing, JSON and Markdown output).
* `results/benchmark/pack_throughput.{json,md}`: the GPU run.
* `results/benchmark/pack_throughput_cpu.{json,md}`: the CPU smoke run.
* `evidence/agents/a51-bench.md`: this report.

## Design decisions

* **Workload.** The data is random but shaped like Churn after the official
  preprocessing: 6,400 train rows, 7 numerical features and 4 categorical features
  with cardinalities [3, 2, 2, 2]. The official config converts the 3 binary features
  to categoricals (`bin_policy = 'convert-to-cat'`), so d_in = 7 + 9 = 16 after one-hot,
  not ~13. The model is d_block = 384, n_blocks = 3, dropout 0.1, ReLU, binary head
  (302,593 parameters per member). Every member gets its own batch of 256 rows per
  step, drawn from its own random permutation of the rows, as in training.
* **What one timed step contains.** It contains the same work as one step of the
  official training loop: gather the member batches, run the forward pass (under
  `torch.autocast(bf16)` when AMP is on, logits cast back to float32), compute the
  per-member BCE mean and sum it over members, then `zero_grad`, `backward` and
  `optimizer.step`. The batch indices are made before timing. The hyperparameters are
  passed as per-member `(K,)` tensors, as in TabPack. Param groups come from
  `make_param_groups` (Muon on every backbone weight for `MuonAdamWPack`). TF32 and
  cuDNN autotuning are off, as in the official code.
* **Timing.** Each configuration is warmed up and then timed in `--repeats` blocks of
  `--steps` steps. There is a `torch.cuda.synchronize()` before and after each block,
  the clock is `time.perf_counter`, and the table shows the median block. The JSON
  also keeps every block and the minimum. Peak memory is
  `torch.cuda.max_memory_allocated` after a `reset_peak_memory_stats`.
* **Sequential baseline.** Member m (seed `seed + m`) is built, warmed up, timed and
  freed before member m+1 is built, so only one model is alive at a time. Its cost
  does not depend on K, so each member is measured once and reused for every K that
  includes it. The sequential time for K is the sum over the first K members of the
  same repeat block. Member measurements are interleaved with the pack runs (the new
  members are measured just before the pack of that K), so that load drift on the
  shared machine affects both sides alike. `--seq-members M` measures only M members
  and scales their sum by K / M. `--sequential extrapolate` uses K times the K=1
  pack time. `--sequential skip` times the pack only.
* **Self-contained.** The script needs only a12 (ModelPack), a15/a16 (optimizers)
  and a17 (param groups). It computes the loss and autocast inline instead of calling
  a19/a29, so it has no other dependencies.

## Tests

The script has no owned test path. It was checked by running it:

```bash
# final GPU run (default flags; about 90-160 s depending on host load)
TABPACK_GPU=1 tools/dev/py scripts/benchmark_pack.py
#   -> results/benchmark/pack_throughput.{json,md}; 24 rows (2 optimizers x 2 AMP x 6 K)
# CPU smoke run (tiny sizes, K in {1, 2, 4}, both optimizers; about 10 s)
tools/dev/py scripts/benchmark_pack.py --cpu
#   -> results/benchmark/pack_throughput_cpu.{json,md}
tools/dev/py -m ruff check scripts/benchmark_pack.py          # All checks passed!
tools/dev/py -m ruff format --check scripts/benchmark_pack.py # already formatted
```

The other modes and error paths were also exercised on CPU:
`--sequential extrapolate` (adds K = 1 when it is missing), `--sequential skip`,
`--seq-members 2`, `--amp both` and `--ks 0` (argparse error). Before a17 landed, the
same checks ran against the real a12/a15/a16 modules with a scratch stand-in for
`make_param_groups`.

## Coordination

* Posted `status` #39 at the start and `done` at the end.
* Merged `feat/a12-nn-model` (with a09/a10/a11), `feat/a15-optim-adamw` and
  `feat/a16-optim-muon` after their `done` posts, then `main` (with a17) on the
  integrator's instruction.
* Before a17 landed, I checked the harness with throwaway stand-ins for
  `make_param_groups` (and before that for every dependency) in the scratchpad.
  None of them are committed.

## Open issues

* **Noisy numbers.** The numbers come from a shared, loaded laptop (WSL2, other
  agents' pytest runs, laptop GPU DVFS). The trend and order of magnitude are solid,
  but single rows can be off by 2x. For publication-quality numbers, rerun on an idle
  machine: `TABPACK_GPU=1 tools/dev/py scripts/benchmark_pack.py --repeats 21 --seq-members 0`.
* **Scaled sequential baseline for K > 8.** By default, the sequential time for
  K > 8 is K times the mean of the 8 members actually measured. The JSON keeps each
  member's median step time (`sequential.member_ms_per_step`) as evidence, and their
  spread is noise, not architecture. This is also why "sequential samples/s" is
  constant for K >= 8 in the tables. `--seq-members 0` measures all 32 members
  (adds about 2-3 min).
* **Input width.** d_in is 16, not "~13". The official Churn config converts the 3
  binary features to categoricals of cardinality 2.
* **Data.** The data is random, not the real Churn rows. Throughput does not depend
  on the values.
* **No automated test.** I own no test path, so there is none. A smoke test could
  call `main(['--cpu', '--ks', '1,2', '--steps', '1', '--repeats', '1', '--warmup',
  '0', '--optimizer', 'adamw', '--out', str(tmp_path / 'b.json')])` (a few seconds).
