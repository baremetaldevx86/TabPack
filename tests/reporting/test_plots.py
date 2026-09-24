"""Tests for tabpack_repro.reporting.plots (a37).

Figures are rendered from small fabricated summaries / run reports that follow the
a36 summary shape (board #110) and the run-report schema (methods/report.py).
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tabpack_repro.reporting.plots import (
    plot_member_scores,
    plot_method_comparison,
    plot_online_ensemble_history,
)

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


# ----------------------------------------------------------------------------------
# Fabricated inputs
# ----------------------------------------------------------------------------------


def _block(values: list[float | None]) -> dict[str, Any]:
    finite = [v for v in values if v is not None]
    n = len(finite)
    return {
        'n': n,
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite, ddof=1)) if n > 1 else 0.0,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'values': values,
    }


def _row(method: str, test: list[float], **extra: Any) -> dict[str, Any]:
    seeds = list(range(len(test)))
    return {
        'method': method,
        'display_name': extra.pop('display_name', method),
        'n': len(test),
        'seeds': seeds,
        'paths': [None] * len(test),
        'test': _block(test),
        'val': _block([v + 0.004 for v in test]),
        'time_sec': _block([12.0] * len(test)),
        'n_models': None,
        'ensemble_size': None,
        'ensemble_n_unique': None,
        'member_test': None,
        'best_member_test': None,
        **extra,
    }


def make_summary() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'dataset': 'churn',
        'metric': 'score',
        'std_ddof': 1,
        'methods': [
            _row('mlp', [0.8530, 0.8555, 0.8490, 0.8575, 0.8540], display_name='MLP'),
            _row(
                'homogeneous',
                [0.8575, 0.8590, 0.8560, 0.8605, 0.8570],
                display_name='MLP ensemble (homogeneous, K=16)',
                member_test=_block([0.8531, 0.8529, 0.8540, 0.8520, 0.8533]),
                ensemble_size=_block([16.0] * 5),
            ),
            _row(
                'tabpack',
                [0.8600, 0.8585, 0.8615],
                display_name='TabPack (reduced, single run)',
                member_test=_block([0.8490, 0.8502, 0.8497]),
            ),
            _row(
                'tabpack-conservative',
                [0.8575, 0.8580, 0.8560, 0.8590, 0.8570],
                display_name='TabPack (reduced, conservative)',
                aggregate_path='runs/tabpack-conservative/report.json',
                source_run='runs/tabpack/seed-0',
                selected_ids=[3, 7, 11],
            ),
        ],
    }


def make_tabpack_report(n_members: int = 12, n_epochs: int = 40) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    members = []
    for i in range(n_members):
        lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-3))))
        n_blocks = int(rng.integers(1, 5))
        test = float(rng.uniform(0.84, 0.86))
        members.append(
            {
                'id': i,
                'best_step': int(rng.integers(100, 2000)),
                'config': {
                    'model': {'n_blocks': n_blocks, 'dropout': 0.1},
                    'optimizer': {'lr': lr, 'weight_decay': 0.01, 'muon_lr': 0.02},
                },
                'metrics': {
                    part: {'accuracy': test, 'roc_auc': 0.85, 'score': test}
                    for part in ('train', 'val', 'test')
                },
            }
        )
    history = []
    ens = 0.845
    for epoch in range(1, n_epochs + 1):
        if epoch % 3 == 0:
            ens = min(ens + 0.002, 0.8625)
        n_running = max(n_members - epoch // 3, 0)
        history.append(
            {
                'epoch': epoch,
                'step': epoch * 25,
                'time': 0.5 * epoch,
                'n_running': n_running,
                'n_finished': n_members - n_running,
                'train_loss': 1.0 / epoch,
                'ensemble_val': ens + 0.003,
                'ensemble_test': ens,
            }
        )
    return {
        'schema_version': 1,
        'method': 'tabpack',
        'dataset': 'churn',
        'seed': 0,
        'config': {'method': 'tabpack', 'seed': 0, 'n_models': n_members},
        'metrics': {
            'val': {'accuracy': 0.8655, 'score': 0.8655},
            'test': {'accuracy': 0.8625, 'score': 0.8625},
        },
        'members': members,
        'ensemble': {
            'ids': [3, 3, 7, 11, 1],
            'steps': [5, 9, 7, 3, 2],
            'size': 5,
            'n_unique': 4,
        },
        'best_member': {'id': 3, 'metrics': members[3]['metrics']},
        'n_epochs': n_epochs,
        'n_steps': n_epochs * 25,
        'time_sec': 20.0,
        'env': {'device': 'cpu', 'gpu': None, 'torch': '2.8', 'git_commit': None},
        'history': history,
    }


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def _assert_png(path: Path, *, min_bytes: int = 5_000) -> None:
    assert path.is_file(), path
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC
    assert len(data) > min_bytes, f'{path} is suspiciously small ({len(data)} B)'


def _png_dpi(path: Path) -> tuple[float, float]:
    from PIL import Image

    with Image.open(path) as image:
        return image.info['dpi']


@pytest.fixture(autouse=True)
def _no_open_pyplot_figures():
    """The plots never go through pyplot; if pyplot is loaded, nothing is open."""
    yield
    pyplot = sys.modules.get('matplotlib.pyplot')
    if pyplot is not None:
        assert pyplot.get_fignums() == []


# ----------------------------------------------------------------------------------
# plot_method_comparison
# ----------------------------------------------------------------------------------


def test_method_comparison_writes_png(tmp_path: Path) -> None:
    out = tmp_path / 'figures' / 'nested' / 'comparison.png'  # parents are created
    plot_method_comparison(make_summary(), out)
    _assert_png(out, min_bytes=20_000)
    dpi = _png_dpi(out)
    assert dpi == pytest.approx((150, 150), abs=0.5)


def test_method_comparison_accepts_str_path(tmp_path: Path) -> None:
    out = tmp_path / 'comparison.png'
    plot_method_comparison(make_summary(), str(out))
    _assert_png(out)


def test_method_comparison_does_not_mutate_summary(tmp_path: Path) -> None:
    summary = make_summary()
    before = copy.deepcopy(summary)
    plot_method_comparison(summary, tmp_path / 'c.png')
    assert summary == before


def test_method_comparison_single_seed_and_unordered_rows(tmp_path: Path) -> None:
    summary = make_summary()
    # Rows out of canonical order, an unknown method, and a single-seed row.
    summary['methods'] = [
        summary['methods'][2],
        _row('some-new-method', [0.851], display_name='New'),
        summary['methods'][0],
    ]
    out = tmp_path / 'c.png'
    plot_method_comparison(summary, out)
    _assert_png(out)


def test_method_comparison_minimal_rows(tmp_path: Path) -> None:
    # Only the keys a plot really needs; no display_name, member blocks, dataset.
    summary = {
        'methods': [
            {
                'method': 'mlp',
                'n': 2,
                'seeds': [0, 1],
                'test': {'n': 2, 'mean': 0.85, 'std': 0.01, 'values': [0.84, 0.86]},
            },
            {'method': 'tabpack', 'test': {'values': [0.86, 0.87]}},
        ]
    }
    out = tmp_path / 'c.png'
    plot_method_comparison(summary, out)
    _assert_png(out)


def test_method_comparison_empty_summary(tmp_path: Path) -> None:
    out = tmp_path / 'c.png'
    plot_method_comparison({'methods': []}, out)
    _assert_png(out, min_bytes=2_000)


def test_method_comparison_other_formats_follow_suffix(tmp_path: Path) -> None:
    out = tmp_path / 'c.svg'
    plot_method_comparison(make_summary(), out)
    assert out.read_text().lstrip().startswith('<?xml')


# ----------------------------------------------------------------------------------
# plot_online_ensemble_history
# ----------------------------------------------------------------------------------


def test_history_writes_png(tmp_path: Path) -> None:
    out = tmp_path / 'figs' / 'history.png'
    plot_online_ensemble_history(make_tabpack_report(), out)
    _assert_png(out, min_bytes=20_000)
    assert _png_dpi(out) == pytest.approx((150, 150), abs=0.5)


def test_history_without_ensemble_scores(tmp_path: Path) -> None:
    report = make_tabpack_report()
    report['method'] = 'homogeneous'
    for h in report['history']:
        h['ensemble_val'] = None
        h['ensemble_test'] = None
    out = tmp_path / 'history.png'
    plot_online_ensemble_history(report, out)
    _assert_png(out)


def test_history_missing_keys_and_nan(tmp_path: Path) -> None:
    report = make_tabpack_report(n_epochs=6)
    for i, h in enumerate(report['history']):
        del h['n_finished']
        del h['epoch']  # falls back to 1-based position
        if i < 2:
            h['ensemble_val'] = None  # before the first ensemble update
            h['ensemble_test'] = float('nan')
    out = tmp_path / 'history.png'
    plot_online_ensemble_history(report, out)
    _assert_png(out)


def test_history_plain_mlp_style(tmp_path: Path) -> None:
    # Per-epoch dicts of the plain MLP (a30): no member counts, no ensemble.
    report = {
        'method': 'mlp',
        'seed': 1,
        'history': [
            {
                'epoch': e,
                'step': 25 * e,
                'time': 0.1 * e,
                'train_loss': 1.0 / e,
                'val_score': 0.84 + 0.001 * e,
                'test_score': 0.83 + 0.001 * e,
            }
            for e in range(1, 11)
        ],
    }
    out = tmp_path / 'history.png'
    plot_online_ensemble_history(report, out)
    _assert_png(out)


@pytest.mark.parametrize('history', [[], None])
def test_history_empty(tmp_path: Path, history: Any) -> None:
    report = make_tabpack_report()
    report['history'] = history
    out = tmp_path / 'history.png'
    plot_online_ensemble_history(report, out)
    _assert_png(out, min_bytes=2_000)


# ----------------------------------------------------------------------------------
# plot_member_scores
# ----------------------------------------------------------------------------------


def test_member_scores_writes_png(tmp_path: Path) -> None:
    out = tmp_path / 'figs' / 'members.png'
    report = make_tabpack_report(n_members=32)
    before = copy.deepcopy(report)
    plot_member_scores(report, out)
    _assert_png(out, min_bytes=20_000)
    assert _png_dpi(out) == pytest.approx((150, 150), abs=0.5)
    assert report == before


def test_member_scores_without_ensemble_or_configs(tmp_path: Path) -> None:
    # Homogeneous-style report: member configs are None, hyperparameters come from
    # the run config; no ensemble selection, no final metrics.
    report = make_tabpack_report()
    report['method'] = 'homogeneous'
    report['config'] = {'model': {'n_blocks': 3}, 'optimizer': {'lr': 1e-3}}
    report['ensemble'] = None
    del report['metrics']
    for m in report['members']:
        m['config'] = None
    out = tmp_path / 'members.png'
    plot_member_scores(report, out)
    _assert_png(out)


def test_member_scores_missing_hyperparameters(tmp_path: Path) -> None:
    report = make_tabpack_report()
    report['config'] = {}
    for m in report['members']:
        m['config'] = None
    report['members'][0]['metrics'] = {}  # member without metrics is skipped
    out = tmp_path / 'members.png'
    plot_member_scores(report, out)
    _assert_png(out)


def test_member_scores_no_members(tmp_path: Path) -> None:
    out = tmp_path / 'members.png'
    plot_member_scores({'method': 'tabpack', 'seed': 0, 'members': []}, out)
    _assert_png(out, min_bytes=2_000)


# ----------------------------------------------------------------------------------
# No global pyplot state
# ----------------------------------------------------------------------------------


def test_module_does_not_import_pyplot_or_switch_backend(tmp_path: Path) -> None:
    out = str(tmp_path / 'a.png')
    code = (
        'import sys, matplotlib\n'
        'backend = matplotlib.rcParams._get_backend_or_none()\n'
        'from tabpack_repro.reporting import plots\n'
        f'plots.plot_method_comparison({make_summary()!r}, {out!r})\n'
        'assert "matplotlib.pyplot" not in sys.modules, "pyplot was imported"\n'
        'assert matplotlib.rcParams._get_backend_or_none() == backend\n'
    )
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())
    _assert_png(tmp_path / 'a.png')


def test_member_scores_without_any_test_scores(tmp_path: Path) -> None:
    report = make_tabpack_report()
    for m in report['members']:
        m['metrics'] = {'val': {'score': 0.85}}
    out = tmp_path / 'members.png'
    plot_member_scores(report, out)
    _assert_png(out, min_bytes=2_000)


def test_member_scores_only_n_blocks_recorded(tmp_path: Path) -> None:
    # The y axis is shared: a note on the empty lr panel must keep the y ticks.
    report = make_tabpack_report()
    for m in report['members']:
        del m['config']['optimizer']
    out = tmp_path / 'members.png'
    plot_member_scores(report, out)
    _assert_png(out)
