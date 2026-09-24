"""Micro-benchmark: training throughput of one pack of K MLPs vs K MLPs one at a time.

TabPack trains K MLPs as one "pack": every layer is a batched matmul over the pack
dimension, so K members cost one set of kernel launches instead of K. This script
measures what that buys on a Churn-sized problem:

* **pack**: one ``ModelPack`` with ``pack_size=K`` and one packed optimizer
  (``AdamWPack`` or ``MuonAdamWPack``); every member gets its own batch of
  ``--batch-size`` rows per step (as in training).
* **sequential**: the same K MLPs trained one after another, each as a ``K=1`` pack
  with its own optimizer (model m is built, warmed up, timed and freed before model
  m+1 is built). Its "step" is the time for all K members to take one step each.
  Member m does not depend on K, so members are measured once (interleaved with the
  pack runs) and the first K of them make up the baseline for pack size K.
  ``--sequential extrapolate`` uses K x the K=1 pack time instead (cheaper).

A timed step covers exactly what the training loop does per batch: gather the
member batches, forward (under bf16 autocast if enabled, logits cast back to
float32), per-member BCE loss summed over members, ``zero_grad``, ``backward`` and
``optimizer.step``. Index tensors are pre-generated (per-member random permutations
of the rows) outside the timed region. Every configuration is warmed up first, the
device is synchronized around each timed block, and the median over ``--repeats``
blocks of ``--steps`` steps is reported.

Reported per (optimizer, amp, K): ms/step, samples/s (K * batch_size rows per step),
speedup = sequential_ms / pack_ms, and peak memory (``torch.cuda.max_memory_allocated``,
CUDA only).

Examples::

    # Full GPU benchmark (exclusive GPU lock):
    TABPACK_GPU=1 tools/dev/py scripts/benchmark_pack.py
    # CPU smoke test with tiny sizes:
    tools/dev/py scripts/benchmark_pack.py --cpu
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_GPU = REPO_ROOT / 'results' / 'benchmark' / 'pack_throughput.json'
DEFAULT_OUT_CPU = REPO_ROOT / 'results' / 'benchmark' / 'pack_throughput_cpu.json'

# Churn after the official preprocessing (extract_bin_from_num + bin_policy
# 'convert-to-cat'): 7 numerical features and 4 categorical features with
# cardinalities [3, 2, 2, 2] -> 7 + 9 = 16 input dims after one-hot. Train split:
# 6,400 rows.
CHURN_N_NUM = 7
CHURN_CAT_CARDS = (3, 2, 2, 2)
CHURN_TRAIN_ROWS = 6400

# (GPU default, CPU smoke default) for the size flags left unset on the command line.
SIZE_DEFAULTS: dict[str, tuple[Any, Any]] = {
    'ks': ([1, 2, 4, 8, 16, 32], [1, 2, 4]),
    'steps': (50, 3),
    'warmup': (10, 1),
    'repeats': (5, 2),
    'n_rows': (CHURN_TRAIN_ROWS, 256),
    'batch_size': (256, 32),
    'd_block': (384, 32),
    'n_blocks': (3, 2),
    'amp': ('both', 'none'),
}

# Fixed optimizer hyperparameters (inside the official Churn search space; they do
# not affect the cost of a step).
LR = 1e-3
WEIGHT_DECAY = 1e-2
MUON_LR = 2e-2


# ---------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--ks',
        type=_int_list,
        default=None,
        help='comma-separated pack sizes (default 1,2,4,8,16,32; --cpu: 1,2,4)',
    )
    p.add_argument('--steps', type=int, default=None, help='timed steps per repeat')
    p.add_argument('--warmup', type=int, default=None, help='untimed warmup steps')
    p.add_argument('--repeats', type=int, default=None, help='timed repeats (median)')
    p.add_argument(
        '--amp',
        choices=['none', 'bf16', 'both'],
        default=None,
        help='bf16 autocast for the forward pass (default both; --cpu: none)',
    )
    p.add_argument(
        '--optimizer',
        choices=['adamw', 'muon', 'both'],
        default='both',
        help='AdamWPack, MuonAdamWPack (Muon on hidden weights) or both',
    )
    p.add_argument(
        '--sequential',
        choices=['measure', 'extrapolate', 'skip'],
        default='measure',
        help='measure: really train K separate K=1 packs one at a time; '
        'extrapolate: K x the measured K=1 pack time; skip: pack only',
    )
    p.add_argument('--n-rows', type=int, default=None, help='rows in the train set')
    p.add_argument('--batch-size', type=int, default=None, help='rows/member/step')
    p.add_argument('--d-block', type=int, default=None, help='hidden width')
    p.add_argument('--n-blocks', type=int, default=None, help='hidden blocks')
    p.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true', help='run on CPU with tiny sizes')
    p.add_argument(
        '--out',
        type=Path,
        default=None,
        help='output JSON path; a Markdown table is written next to it (.md). '
        f'Default: {DEFAULT_OUT_GPU.relative_to(REPO_ROOT)} '
        f'({DEFAULT_OUT_CPU.relative_to(REPO_ROOT)} with --cpu)',
    )
    args = p.parse_args(argv)

    slot = 1 if args.cpu else 0
    for name, defaults in SIZE_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, defaults[slot])
    if args.out is None:
        args.out = DEFAULT_OUT_CPU if args.cpu else DEFAULT_OUT_GPU

    for name in ('steps', 'repeats', 'n_rows', 'batch_size', 'd_block', 'n_blocks'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be >= 1')
    if args.warmup < 0:
        p.error('--warmup must be >= 0')
    if not args.ks or min(args.ks) < 1:
        p.error('--ks must be a non-empty list of positive integers')
    if args.batch_size > args.n_rows:
        p.error('--batch-size must be <= --n-rows')
    if not 0.0 <= args.dropout < 1.0:
        p.error('--dropout must be in [0, 1)')
    return args


def _int_list(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(',') if x.strip()]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f'expected e.g. 1,2,4, got {text!r}') from err


@dataclass(frozen=True)
class Data:
    x_num: Tensor  # (N, n_num) float32
    x_cat: Tensor  # (N, n_cat) int64 ordinal codes
    y: Tensor  # (N,) float32 in {0, 1}

    @property
    def n_rows(self) -> int:
        return self.y.shape[0]


def make_data(n_rows: int, device: torch.device, seed: int) -> Data:
    """A random Churn-shaped binary classification dataset on ``device``."""
    g = torch.Generator().manual_seed(seed)
    x_num = torch.randn(n_rows, CHURN_N_NUM, generator=g)
    x_cat = torch.stack(
        [torch.randint(card, (n_rows,), generator=g) for card in CHURN_CAT_CARDS], 1
    )
    y = torch.randint(2, (n_rows,), generator=g).float()
    return Data(x_num.to(device), x_cat.to(device), y.to(device))


def make_batches(
    n_rows: int, pack_size: int, batch_size: int, device: torch.device, seed: int
) -> list[Tensor]:
    """One epoch of full-size per-member batches: a list of int64 (K, B) indices.

    Every member walks its own random permutation of the rows (like training);
    the last incomplete batch is dropped so that every timed step has equal work.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.rand(pack_size, n_rows, generator=g).argsort(dim=1)
    n_batches = n_rows // batch_size
    perm = perm[:, : n_batches * batch_size].to(device)
    return list(perm.split(batch_size, dim=1))


def make_model_and_optimizer(
    pack_size: int, args: argparse.Namespace, optimizer: str, device: torch.device
) -> tuple[ModelPack, torch.optim.Optimizer]:
    model = ModelPack(
        n_num_features=CHURN_N_NUM,
        cat_cardinalities=list(CHURN_CAT_CARDS),
        n_classes=2,
        pack_size=pack_size,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        activation='ReLU',
    ).to(device)

    def per_member(value: float) -> Tensor:
        # TabPack samples hyperparameters per member: pass (K,) tensors.
        return torch.full((pack_size,), value, dtype=torch.float32, device=device)

    opt: torch.optim.Optimizer
    if optimizer == 'adamw':
        opt = AdamWPack(
            make_param_groups(model, muon=False),
            lr=per_member(LR),
            weight_decay=per_member(WEIGHT_DECAY),
            pack_size=pack_size,
        )
    elif optimizer == 'muon':
        opt = MuonAdamWPack(
            make_param_groups(model, muon=True),
            lr=per_member(LR),
            weight_decay=per_member(WEIGHT_DECAY),
            muon_lr=per_member(MUON_LR),
            pack_size=pack_size,
        )
    else:
        raise ValueError(f'unknown optimizer {optimizer!r}')
    return model, opt


def make_autocast(
    amp: str, device: torch.device
) -> Callable[[], contextlib.AbstractContextManager[Any]]:
    """A factory of fresh autocast contexts (bf16) or no-op contexts."""
    if amp == 'none':
        return contextlib.nullcontext
    assert amp == 'bf16', amp
    return lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------
class Trainer:
    """Holds one pack + optimizer and runs training steps over pre-made batches."""

    def __init__(
        self,
        pack_size: int,
        args: argparse.Namespace,
        optimizer: str,
        amp: str,
        data: Data,
        device: torch.device,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        self.model, self.optimizer = make_model_and_optimizer(
            pack_size, args, optimizer, device
        )
        self.model.train()
        self.data = data
        self.autocast = make_autocast(amp, device)
        self.batches = make_batches(
            data.n_rows, pack_size, args.batch_size, device, seed
        )
        self._batch_iter = self._cycle()
        self.n_params_per_member = (
            sum(p.numel() for p in self.model.parameters()) // pack_size
        )

    def _cycle(self) -> Iterator[Tensor]:
        while True:
            yield from self.batches

    def step(self) -> Tensor:
        idx = next(self._batch_iter)  # (K, B)
        with self.autocast():
            logits = self.model(self.data.x_num[idx], self.data.x_cat[idx])
        logits = logits.float()  # (K, B)
        losses = F.binary_cross_entropy_with_logits(
            logits, self.data.y[idx], reduction='none'
        ).mean(dim=1)
        # As in TabPack: sum (not mean) so the gradient scale is independent of K.
        loss = losses.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def run(self, n_steps: int, device: torch.device) -> float:
        """Run n_steps steps; return the wall-clock seconds (device-synchronized)."""
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(n_steps):
            self.step()
        _synchronize(device)
        return time.perf_counter() - start


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _reset_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mib(device: torch.device) -> float | None:
    if device.type != 'cuda':
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def _summarize(
    seconds_per_repeat: Sequence[float], pack_size: int, args: argparse.Namespace
) -> dict[str, Any]:
    ms_per_step = [1000.0 * s / args.steps for s in seconds_per_repeat]
    median_ms = statistics.median(ms_per_step)
    rows_per_step = pack_size * args.batch_size
    return {
        'ms_per_step': median_ms,
        'ms_per_step_min': min(ms_per_step),
        'ms_per_step_repeats': ms_per_step,
        'samples_per_s': rows_per_step / (median_ms / 1000.0),
    }


def bench_pack(
    pack_size: int,
    args: argparse.Namespace,
    optimizer: str,
    amp: str,
    data: Data,
    device: torch.device,
) -> dict[str, Any]:
    _reset_memory(device)
    trainer = Trainer(pack_size, args, optimizer, amp, data, device, args.seed)
    trainer.run(args.warmup, device)
    seconds = [trainer.run(args.steps, device) for _ in range(args.repeats)]
    result = _summarize(seconds, pack_size, args)
    result['peak_mem_mib'] = _peak_memory_mib(device)
    result['n_params_per_member'] = trainer.n_params_per_member
    del trainer
    _reset_memory(device)
    return result


@dataclass(frozen=True)
class MemberTiming:
    seconds: list[float]  # one entry per repeat (``steps`` steps each)
    peak_mem_mib: float | None


def bench_member(
    member: int,
    args: argparse.Namespace,
    optimizer: str,
    amp: str,
    data: Data,
    device: torch.device,
) -> MemberTiming:
    """Train member ``member`` alone (a K=1 pack): build, warm up, time, free."""
    _reset_memory(device)
    trainer = Trainer(1, args, optimizer, amp, data, device, args.seed + member)
    trainer.run(args.warmup, device)
    seconds = [trainer.run(args.steps, device) for _ in range(args.repeats)]
    peak = _peak_memory_mib(device)
    del trainer
    _reset_memory(device)
    return MemberTiming(seconds, peak)


def sequential_result(
    members: Sequence[MemberTiming], pack_size: int, args: argparse.Namespace
) -> dict[str, Any]:
    """The first K members trained strictly one after another.

    Member m (seed ``seed + m``) is built, warmed up, timed for ``repeats`` blocks of
    ``steps`` steps and freed before member m+1 is built. Member m does not depend on
    K, so every member is measured once and shared by all K >= m + 1. Repeat r of the
    sequential run is the sum over members of their r-th block, i.e. the time for
    all K members to take ``steps`` steps each; the peak memory is the maximum over
    members (only one model is alive at a time).
    """
    used = members[:pack_size]
    assert len(used) == pack_size
    seconds = [sum(block) for block in zip(*(m.seconds for m in used), strict=True)]
    result = _summarize(seconds, pack_size, args)
    peaks = [m.peak_mem_mib for m in used if m.peak_mem_mib is not None]
    result['peak_mem_mib'] = max(peaks) if peaks else None
    result['mode'] = 'measured'
    return result


def extrapolate_sequential(
    pack_size: int, k1_pack: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Sequential cost estimated as K x the measured K=1 pack step."""
    ms = pack_size * k1_pack['ms_per_step']
    return {
        'ms_per_step': ms,
        'ms_per_step_min': pack_size * k1_pack['ms_per_step_min'],
        'ms_per_step_repeats': [pack_size * x for x in k1_pack['ms_per_step_repeats']],
        'samples_per_s': pack_size * args.batch_size / (ms / 1000.0),
        'peak_mem_mib': k1_pack['peak_mem_mib'],
        'mode': 'extrapolated',
    }


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------
def describe_env(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'gpu_total_mem_mib': (
            torch.cuda.get_device_properties(device).total_memory / 2**20
            if device.type == 'cuda'
            else None
        ),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if device.type == 'cuda' else None,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_threads': torch.get_num_threads(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'git_commit': _git_commit(),
        'timestamp_utc': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    return env


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _fmt(x: float | None, digits: int = 2) -> str:
    return '-' if x is None else f'{x:,.{digits}f}'


_OPT_NAMES = {'adamw': 'AdamWPack', 'muon': 'MuonAdamWPack'}
_AMP_NAMES = {'none': 'float32', 'bf16': 'bf16 autocast'}


def render_markdown(report: dict[str, Any]) -> str:
    env, cfg = report['env'], report['config']
    gpu = env['gpu']
    device = f'`{env["device"]}`'
    if gpu:
        device += f' ({gpu}, {_fmt(env["gpu_total_mem_mib"], 0)} MiB)'
    lines = [
        '# TabPack pack-training throughput',
        '',
        'One pack of K MLPs (`ModelPack` + packed optimizer; one step = every member',
        'takes one step on its own batch) vs the same K MLPs trained one at a time',
        '(K separate `K=1` packs). Generated by `scripts/benchmark_pack.py`.',
        '',
        '* **ms/step**: median wall-clock time for all K members to take one step',
        "  (sequential: sum of the K members' step times).",
        '* **samples/s**: K x batch_size rows per step / step time.',
        '* **speedup**: sequential ms/step / pack ms/step.',
        '* **peak MiB**: `torch.cuda.max_memory_allocated` over the configuration',
        '  (sequential: models live one at a time).',
        '',
        '## Environment',
        '',
        f'* device: {device}',
        (
            f'* torch {env["torch"]}, CUDA {env["cuda"]}, cuDNN {env["cudnn"]}, '
            f'Python {env["python"]}'
        ),
        f'* platform: {env["platform"]}; TF32 matmul: {env["tf32_matmul"]}',
        f'* git commit: `{env["git_commit"]}`; run at {env["timestamp_utc"]}',
        '',
        '## Configuration',
        '',
        (
            f'* data: {cfg["n_rows"]} random rows shaped like Churn: {cfg["n_num"]} '
            f'numerical + categorical cardinalities {cfg["cat_cardinalities"]} '
            f'-> d_in = {cfg["d_in"]}; binary target, BCE loss'
        ),
        (
            f'* model: d_block = {cfg["d_block"]}, n_blocks = {cfg["n_blocks"]}, '
            f'dropout = {cfg["dropout"]}, ReLU; '
            f'{cfg["n_params_per_member"]:,} parameters per member'
        ),
        (
            f'* batch: {cfg["batch_size"]} rows per member per step; '
            f'warmup {cfg["warmup"]} steps, then {cfg["repeats"]} repeats x '
            f'{cfg["steps"]} steps (median)'
        ),
        f'* sequential baseline: {cfg["sequential"]}',
        '',
    ]
    groups = [
        (
            opt,
            amp,
            [r for r in report['results'] if (r['optimizer'], r['amp']) == (opt, amp)],
        )
        for opt in cfg['optimizers']
        for amp in cfg['amps']
    ]
    groups = [g for g in groups if g[2]]
    headline = [
        f'* {_OPT_NAMES[opt]}, {_AMP_NAMES[amp]}: K = {rows[-1]["k"]} -> '
        f'{_fmt(rows[-1]["speedup"])}x faster than sequential '
        f'({_fmt(rows[-1]["pack"]["ms_per_step"])} vs '
        f'{_fmt(rows[-1]["sequential"]["ms_per_step"])} ms per step of all members)'
        for opt, amp, rows in groups
        if rows[-1]['sequential'] is not None
    ]
    if headline:
        lines += ['## Headline (largest K)', '', *headline, '']
    for opt, amp, rows in groups:
        lines += [
            f'## {_OPT_NAMES[opt]}, {_AMP_NAMES[amp]}',
            '',
            (
                '| K | pack ms/step | sequential ms/step | speedup '
                '| pack samples/s | sequential samples/s '
                '| pack peak MiB | sequential peak MiB |'
            ),
            '| --: | --: | --: | --: | --: | --: | --: | --: |',
        ]
        for r in rows:
            pack, seq = r['pack'], r['sequential'] or {}
            speedup = r['speedup']
            lines.append(
                f'| {r["k"]} | {_fmt(pack["ms_per_step"])} '
                f'| {_fmt(seq.get("ms_per_step"))} '
                f'| {_fmt(speedup)}{"x" if speedup is not None else ""} '
                f'| {_fmt(pack["samples_per_s"], 0)} '
                f'| {_fmt(seq.get("samples_per_s"), 0)} '
                f'| {_fmt(pack["peak_mem_mib"], 1)} '
                f'| {_fmt(seq.get("peak_mem_mib"), 1)} |'
            )
        lines.append('')
    lines += [f'Total benchmark wall time: {report["wall_time_s"]:.1f} s.', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        sys.exit(
            'CUDA is not available: run with TABPACK_GPU=1 (tools/dev/py), '
            'or pass --cpu for a CPU smoke run'
        )
    # Same numerics as the official code: no TF32, no cuDNN autotuning.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    optimizers = ['adamw', 'muon'] if args.optimizer == 'both' else [args.optimizer]
    amps = ['none', 'bf16'] if args.amp == 'both' else [args.amp]
    ks = sorted(set(args.ks))
    if args.sequential == 'extrapolate' and 1 not in ks:
        ks = [1, *ks]  # the extrapolation needs the K=1 pack time

    wall_start = time.perf_counter()
    data = make_data(args.n_rows, device, args.seed)
    results: list[dict[str, Any]] = []
    n_params_per_member = None
    for optimizer in optimizers:
        for amp in amps:
            k1_pack = None
            members: list[MemberTiming] = []
            for k in ks:
                if args.sequential == 'measure':
                    # Interleave: measure the new sequential members, then the pack.
                    members += [
                        bench_member(m, args, optimizer, amp, data, device)
                        for m in range(len(members), k)
                    ]
                pack = bench_pack(k, args, optimizer, amp, data, device)
                n_params_per_member = pack.pop('n_params_per_member')
                if k == 1:
                    k1_pack = pack
                if args.sequential == 'measure':
                    seq = sequential_result(members, k, args)
                elif args.sequential == 'extrapolate':
                    assert k1_pack is not None
                    seq = extrapolate_sequential(k, k1_pack, args)
                else:
                    seq = None
                speedup = (
                    None if seq is None else seq['ms_per_step'] / pack['ms_per_step']
                )
                results.append(
                    {
                        'optimizer': optimizer,
                        'amp': amp,
                        'k': k,
                        'pack': pack,
                        'sequential': seq,
                        'speedup': speedup,
                    }
                )
                print(
                    f'[{optimizer:5s} {amp:4s}] K={k:3d}  '
                    f'pack {pack["ms_per_step"]:8.2f} ms/step  '
                    + (
                        f'seq {seq["ms_per_step"]:8.2f} ms/step  '
                        f'speedup {speedup:6.2f}x  '
                        if seq is not None and speedup is not None
                        else ''
                    )
                    + f'peak {_fmt(pack["peak_mem_mib"], 1)} MiB',
                    flush=True,
                )

    report: dict[str, Any] = {
        'env': describe_env(device),
        'config': {
            'ks': ks,
            'steps': args.steps,
            'warmup': args.warmup,
            'repeats': args.repeats,
            'amps': amps,
            'optimizers': optimizers,
            'sequential': args.sequential,
            'n_rows': args.n_rows,
            'n_num': CHURN_N_NUM,
            'cat_cardinalities': list(CHURN_CAT_CARDS),
            'd_in': CHURN_N_NUM + sum(CHURN_CAT_CARDS),
            'd_block': args.d_block,
            'n_blocks': args.n_blocks,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'n_params_per_member': n_params_per_member,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'muon_lr': MUON_LR,
            'seed': args.seed,
            'cpu_smoke': args.cpu,
        },
        'results': results,
        'wall_time_s': time.perf_counter() - wall_start,
    }

    out_json: Path = args.out
    out_md = out_json.with_suffix('.md')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + '\n')
    out_md.write_text(render_markdown(report))
    print(f'wrote {out_json} and {out_md} ({report["wall_time_s"]:.1f} s)')
    return report


if __name__ == '__main__':
    main()
