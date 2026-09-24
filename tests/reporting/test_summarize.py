"""Tests for reporting/summarize.py (a36), on fabricated run reports."""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import pytest

from tabpack_repro.reporting.summarize import (
    METHOD_ORDER,
    collect_runs,
    summarize,
    to_markdown,
    write_summary,
)

# Test/val scores per (method, seed); chosen to be exact at 2 decimals in %.
MLP = {0: (0.8600, 0.8550), 1: (0.8650, 0.8600), 2: (0.8700, 0.8650)}
HOMOGENEOUS = {0: (0.8610, 0.8570), 1: (0.8630, 0.8590), 2: (0.8620, 0.8580)}
TABPACK = {0: (0.8580, 0.8620), 1: (0.8570, 0.8630), 2: (0.8560, 0.8640)}
# Aggregate scores of the conservative protocol (the source of truth).
CONSERVATIVE = {0: (0.8575, 0.8625), 1: (0.8585, 0.8635), 2: (0.8595, 0.8645)}


def _metrics(test: float, val: float) -> dict[str, Any]:
    return {
        'val': {'score': val, 'accuracy': val, 'roc-auc': 0.8},
        'test': {'score': test, 'accuracy': test, 'roc-auc': 0.8},
    }


def _member(member_id: int, test: float, val: float) -> dict[str, Any]:
    metrics = _metrics(test, val)
    metrics['train'] = {'score': 0.9}
    return {
        'id': member_id,
        'best_step': 10 * member_id,
        'config': None,
        'metrics': metrics,
    }


def make_report(
    method: str,
    seed: int,
    test: float,
    val: float,
    *,
    time_sec: float = 10.0,
    members: list[tuple[float, float]] | None = None,
    ensemble_ids: list[int] | None = None,
    n_models: int | None = None,
    dataset: str = 'churn',
) -> dict[str, Any]:
    """A report following methods/report.py (members given as (test, val) pairs)."""
    member_list = [_member(i, t, v) for i, (t, v) in enumerate(members or [])]
    config: dict[str, Any] = {'method': method, 'seed': seed}
    if n_models is not None:
        config['n_models'] = n_models
    best = None
    if member_list:
        top = max(member_list, key=lambda m: m['metrics']['val']['score'])
        best = {'id': top['id'], 'metrics': top['metrics']}
    ensemble = None
    if ensemble_ids is not None:
        ensemble = {
            'ids': ensemble_ids,
            'steps': [1] * len(ensemble_ids),
            'size': len(ensemble_ids),
            'n_unique': len(set(ensemble_ids)),
        }
    return {
        'schema_version': 1,
        'method': method,
        'dataset': dataset,
        'seed': seed,
        'config': config,
        'metrics': _metrics(test, val),
        'members': member_list,
        'ensemble': ensemble,
        'best_member': best,
        'n_epochs': 20,
        'n_steps': 600,
        'time_sec': time_sec,
        'env': {'device': 'cpu', 'gpu': None, 'torch': '2.x', 'git_commit': None},
        'history': [],
    }


def mlp_report(seed: int) -> dict[str, Any]:
    test, val = MLP[seed]
    return make_report('mlp', seed, test, val, time_sec=2.0 + seed)


def homogeneous_report(seed: int) -> dict[str, Any]:
    # K = 4 members, uniform average of all of them (ensemble = None).
    test, val = HOMOGENEOUS[seed]
    members = [(0.850 + 0.001 * i + 0.001 * seed, 0.84 + 0.001 * i) for i in range(4)]
    return make_report(
        'homogeneous', seed, test, val, time_sec=5.0, members=members, n_models=4
    )


def tabpack_report(seed: int) -> dict[str, Any]:
    test, val = TABPACK[seed]
    # 6 members; member 5 has the best val score.
    members = [(0.80 + 0.01 * i, 0.80 + 0.01 * i + 0.001 * seed) for i in range(6)]
    ids = [5, 4, 5, 3][: 2 + seed]
    return make_report(
        'tabpack',
        seed,
        test,
        val,
        time_sec=8.0 + seed,
        members=members,
        ensemble_ids=ids,
        n_models=6,
    )


def conservative_seed_report(seed: int) -> dict[str, Any]:
    # Deliberately different final scores than the aggregate: they must NOT be used.
    members = [(0.84, 0.85), (0.85, 0.86), (0.86, 0.87)]
    return make_report(
        'tabpack-conservative-seed',
        seed,
        0.5,
        0.5,
        time_sec=3.0,
        members=members,
        ensemble_ids=[2, 1, 2],
        n_models=3,
    )


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def conservative_aggregate(seeds: list[int]) -> dict[str, Any]:
    test = [CONSERVATIVE[s][0] for s in seeds]
    val = [CONSERVATIVE[s][1] for s in seeds]
    # Same keys as methods/conservative.py (a33) writes.
    return {
        'schema_version': 1,
        'method': 'tabpack-conservative',
        'dataset': 'churn',
        'config': {
            'method': 'tabpack-conservative',
            'source_run': 'runs/churn/tabpack/seed-0',
            'n_seeds': len(seeds),
        },
        'source_run': 'runs/churn/tabpack/seed-0',
        'source_seed': 0,
        'selected_ids': [3, 4, 5],
        'n_seeds': len(seeds),
        'seeds': seeds,
        'scores': {'val': val, 'test': test},
        'mean': {'val': statistics.mean(val), 'test': statistics.mean(test)},
        'std': {'val': _stdev(val), 'test': _stdev(test)},
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    file = path / 'report.json'
    file.write_text(json.dumps(report, indent=2))
    return file


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """The documented runs/churn layout with 3 seeds per method."""
    root = tmp_path / 'runs' / 'churn'
    for seed in range(3):
        write_report(root / 'mlp' / f'seed-{seed}', mlp_report(seed))
        write_report(root / 'homogeneous' / f'seed-{seed}', homogeneous_report(seed))
        write_report(root / 'tabpack' / f'seed-{seed}', tabpack_report(seed))
        write_report(
            root / 'tabpack-conservative' / f'seed-{seed}',
            conservative_seed_report(seed),
        )
    write_report(root / 'tabpack-conservative', conservative_aggregate([0, 1, 2]))
    return root


def _row(summary: dict[str, Any], method: str) -> dict[str, Any]:
    rows = [r for r in summary['methods'] if r['method'] == method]
    assert len(rows) == 1
    return rows[0]


def _check_block(block: dict[str, Any], values: list[float]) -> None:
    assert block['values'] == pytest.approx(values)
    assert block['n'] == len(values)
    assert block['mean'] == pytest.approx(statistics.mean(values), abs=1e-12)
    assert block['std'] == pytest.approx(_stdev(values), abs=1e-12)
    assert block['min'] == pytest.approx(min(values), abs=1e-12)
    assert block['max'] == pytest.approx(max(values), abs=1e-12)


# ---------------------------------------------------------------------------------
# collect_runs
# ---------------------------------------------------------------------------------


def test_collect_runs_loads_every_top_level_report(runs_dir: Path) -> None:
    runs = collect_runs(runs_dir)
    methods = [r['method'] for r in runs]
    assert sorted(methods) == sorted(
        ['mlp'] * 3 + ['homogeneous'] * 3 + ['tabpack'] * 3 + ['tabpack-conservative']
    )
    for run in runs:
        path = Path(run['path'])
        assert path.name == 'report.json'
        assert path.is_file()
        assert path.is_relative_to(runs_dir)
    # Deterministic (path) order.
    assert [r['path'] for r in runs] == [r['path'] for r in collect_runs(runs_dir)]
    assert [r['path'] for r in runs] == sorted(
        (r['path'] for r in runs), key=lambda p: Path(p).parts
    )


def test_collect_runs_nests_conservative_seed_reports(runs_dir: Path) -> None:
    runs = collect_runs(runs_dir)
    assert all(r['method'] != 'tabpack-conservative-seed' for r in runs)
    (aggregate,) = [r for r in runs if r['method'] == 'tabpack-conservative']
    assert aggregate['path'] == str(runs_dir / 'tabpack-conservative' / 'report.json')
    seed_runs = aggregate['seed_runs']
    assert [r['seed'] for r in seed_runs] == [0, 1, 2]
    assert all(r['method'] == 'tabpack-conservative-seed' for r in seed_runs)
    assert [r['path'] for r in seed_runs] == [
        str(runs_dir / 'tabpack-conservative' / f'seed-{s}' / 'report.json')
        for s in range(3)
    ]


def test_collect_runs_seed_runs_sorted_numerically(tmp_path: Path) -> None:
    root = tmp_path / 'cons'
    for seed in (10, 2, 1):
        write_report(root / f'seed-{seed}', conservative_seed_report(seed))
    agg = conservative_aggregate([0, 1, 2])
    agg['seeds'] = [10, 2, 1]
    write_report(root, agg)
    (aggregate,) = collect_runs(tmp_path)
    assert [r['seed'] for r in aggregate['seed_runs']] == [1, 2, 10]


def test_collect_runs_keeps_orphan_seed_reports_top_level(tmp_path: Path) -> None:
    write_report(tmp_path / 'cons' / 'seed-0', conservative_seed_report(0))
    write_report(tmp_path / 'mlp' / 'seed-0', mlp_report(0))
    runs = collect_runs(tmp_path)
    assert sorted(r['method'] for r in runs) == ['mlp', 'tabpack-conservative-seed']
    # ... but summarize never turns them into a row.
    summary = summarize(runs)
    assert [r['method'] for r in summary['methods']] == ['mlp']


def test_collect_runs_empty_and_missing_dir(tmp_path: Path) -> None:
    assert collect_runs(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        collect_runs(tmp_path / 'missing')


def test_collect_runs_rejects_other_schema_versions(tmp_path: Path) -> None:
    report = mlp_report(0)
    report['schema_version'] = 2
    write_report(tmp_path / 'mlp', report)
    with pytest.raises(ValueError, match='schema_version'):
        collect_runs(tmp_path)


def test_collect_runs_rejects_non_object_json(tmp_path: Path) -> None:
    (tmp_path / 'report.json').write_text('[1, 2]')
    with pytest.raises(TypeError, match='JSON object'):
        collect_runs(tmp_path)


# ---------------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------------


def test_summarize_top_level(runs_dir: Path) -> None:
    summary = summarize(collect_runs(runs_dir))
    assert summary['schema_version'] == 1
    assert summary['dataset'] == 'churn'
    assert summary['metric'] == 'score'
    assert summary['std_ddof'] == 1
    assert [r['method'] for r in summary['methods']] == list(METHOD_ORDER)
    assert [r['display_name'] for r in summary['methods']] == [
        'MLP',
        'MLP ensemble (homogeneous, K=4)',
        'TabPack (reduced, single run)',
        'TabPack (reduced, conservative)',
    ]


@pytest.mark.parametrize(
    ('method', 'scores'),
    [('mlp', MLP), ('homogeneous', HOMOGENEOUS), ('tabpack', TABPACK)],
)
def test_summarize_score_statistics(
    runs_dir: Path, method: str, scores: dict[int, tuple[float, float]]
) -> None:
    row = _row(summarize(collect_runs(runs_dir)), method)
    assert row['n'] == 3
    assert row['seeds'] == [0, 1, 2]
    assert row['paths'] == [
        str(runs_dir / method / f'seed-{s}' / 'report.json') for s in range(3)
    ]
    _check_block(row['test'], [scores[s][0] for s in range(3)])
    _check_block(row['val'], [scores[s][1] for s in range(3)])


def test_summarize_uses_sample_std() -> None:
    row = _row(summarize([mlp_report(0), mlp_report(2)]), 'mlp')
    # Two values 0.86 and 0.87: ddof=1 std = 0.01 / sqrt(2), not 0.005.
    assert row['test']['std'] == pytest.approx(0.01 / 2**0.5)
    assert row['test']['std'] == statistics.stdev([0.86, 0.87])


def test_summarize_single_seed_has_zero_std() -> None:
    row = _row(summarize([tabpack_report(1)]), 'tabpack')
    assert row['n'] == 1
    assert row['test'] == {
        'n': 1,
        'mean': 0.857,
        'std': 0.0,
        'min': 0.857,
        'max': 0.857,
        'values': [0.857],
    }
    assert row['member_test']['std'] == 0.0
    assert row['time_sec']['std'] == 0.0


def test_summarize_is_order_independent(runs_dir: Path) -> None:
    runs = collect_runs(runs_dir)
    expected = summarize(runs)
    for seed in range(5):
        shuffled = list(runs)
        random.Random(seed).shuffle(shuffled)
        assert summarize(shuffled) == expected


def test_summarize_sorts_seeds() -> None:
    row = _row(summarize([mlp_report(2), mlp_report(0), mlp_report(1)]), 'mlp')
    assert row['seeds'] == [0, 1, 2]
    assert row['test']['values'] == [0.86, 0.865, 0.87]
    assert row['time_sec']['values'] == [2.0, 3.0, 4.0]


def test_summarize_unknown_methods_come_last_sorted() -> None:
    runs = [
        make_report('zeta', 0, 0.8, 0.8),
        mlp_report(0),
        make_report('alpha', 0, 0.7, 0.7),
        tabpack_report(0),
    ]
    summary = summarize(runs)
    assert [r['method'] for r in summary['methods']] == [
        'mlp',
        'tabpack',
        'alpha',
        'zeta',
    ]
    assert _row(summary, 'alpha')['display_name'] == 'alpha'


def test_summarize_time_and_plain_mlp_has_no_ensemble_stats(runs_dir: Path) -> None:
    row = _row(summarize(collect_runs(runs_dir)), 'mlp')
    _check_block(row['time_sec'], [2.0, 3.0, 4.0])
    for key in (
        'n_models',
        'ensemble_size',
        'ensemble_n_unique',
        'member_test',
        'best_member_test',
    ):
        assert row[key] is None, key


def test_summarize_homogeneous_ensemble_stats(runs_dir: Path) -> None:
    row = _row(summarize(collect_runs(runs_dir)), 'homogeneous')
    _check_block(row['n_models'], [4.0, 4.0, 4.0])
    # No selection (ensemble=None): the ensemble is all K members.
    _check_block(row['ensemble_size'], [4.0, 4.0, 4.0])
    _check_block(row['ensemble_n_unique'], [4.0, 4.0, 4.0])
    member_means = [
        statistics.mean(0.850 + 0.001 * i + 0.001 * s for i in range(4))
        for s in range(3)
    ]
    _check_block(row['member_test'], member_means)
    # Best-by-val member is the last one (i = 3).
    _check_block(row['best_member_test'], [0.853 + 0.001 * s for s in range(3)])
    _check_block(row['time_sec'], [5.0, 5.0, 5.0])


def test_summarize_tabpack_ensemble_stats(runs_dir: Path) -> None:
    row = _row(summarize(collect_runs(runs_dir)), 'tabpack')
    _check_block(row['n_models'], [6.0, 6.0, 6.0])
    _check_block(row['ensemble_size'], [2.0, 3.0, 4.0])
    _check_block(row['ensemble_n_unique'], [2.0, 2.0, 3.0])
    member_mean = statistics.mean(0.80 + 0.01 * i for i in range(6))
    _check_block(row['member_test'], [member_mean] * 3)
    _check_block(row['best_member_test'], [0.85, 0.85, 0.85])
    _check_block(row['time_sec'], [8.0, 9.0, 10.0])


def test_summarize_conservative_uses_aggregate_scores(runs_dir: Path) -> None:
    summary = summarize(collect_runs(runs_dir))
    row = _row(summary, 'tabpack-conservative')
    assert row['n'] == 3
    assert row['seeds'] == [0, 1, 2]
    # The per-seed reports claim 0.5; only the aggregate's scores count.
    _check_block(row['test'], [CONSERVATIVE[s][0] for s in range(3)])
    _check_block(row['val'], [CONSERVATIVE[s][1] for s in range(3)])
    assert row['aggregate_path'] == str(
        runs_dir / 'tabpack-conservative' / 'report.json'
    )
    assert row['source_run'] == 'runs/churn/tabpack/seed-0'
    assert row['selected_ids'] == [3, 4, 5]
    assert row['paths'] == [
        str(runs_dir / 'tabpack-conservative' / f'seed-{s}' / 'report.json')
        for s in range(3)
    ]
    # Extras come from the nested per-seed reports.
    _check_block(row['time_sec'], [3.0, 3.0, 3.0])
    _check_block(row['n_models'], [3.0, 3.0, 3.0])
    _check_block(row['ensemble_size'], [3.0, 3.0, 3.0])
    _check_block(row['ensemble_n_unique'], [2.0, 2.0, 2.0])
    _check_block(row['member_test'], [0.85, 0.85, 0.85])
    _check_block(row['best_member_test'], [0.86, 0.86, 0.86])
    # No row for the per-seed reports.
    assert 'tabpack-conservative-seed' not in [r['method'] for r in summary['methods']]


def test_summarize_no_double_counting_with_flat_lists(runs_dir: Path) -> None:
    """Per-seed conservative reports passed at the top level are ignored."""
    runs = collect_runs(runs_dir)
    (aggregate,) = [r for r in runs if r['method'] == 'tabpack-conservative']
    flat = [*runs, *aggregate['seed_runs']]
    summary = summarize(flat)
    assert summary == summarize(runs)
    assert [r['n'] for r in summary['methods']] == [3, 3, 3, 3]
    # A conservative seed report never counts as a TabPack single run either.
    assert _row(summary, 'tabpack')['test']['values'] == [0.858, 0.857, 0.856]


def test_summarize_conservative_without_seed_reports() -> None:
    aggregate = conservative_aggregate([0, 1, 2])
    aggregate['path'] = 'runs/churn/tabpack-conservative/report.json'
    row = _row(summarize([aggregate]), 'tabpack-conservative')
    _check_block(row['test'], [CONSERVATIVE[s][0] for s in range(3)])
    assert row['paths'] == [None, None, None]
    assert row['time_sec'] is None
    assert row['member_test'] is None
    assert row['ensemble_size'] is None
    # K still comes from the aggregate.
    _check_block(row['n_models'], [3.0, 3.0, 3.0])


def test_summarize_conservative_sorts_unsorted_aggregate_seeds() -> None:
    aggregate = conservative_aggregate([2, 0, 1])
    row = _row(summarize([aggregate]), 'tabpack-conservative')
    assert row['seeds'] == [0, 1, 2]
    assert row['test']['values'] == [CONSERVATIVE[s][0] for s in range(3)]
    assert row['val']['values'] == [CONSERVATIVE[s][1] for s in range(3)]


def test_summarize_partial_seed_reports(runs_dir: Path) -> None:
    runs = collect_runs(runs_dir)
    (aggregate,) = [r for r in runs if r['method'] == 'tabpack-conservative']
    del aggregate['seed_runs'][1]
    row = _row(summarize(runs), 'tabpack-conservative')
    assert row['n'] == 3
    assert row['paths'][1] is None
    assert row['time_sec']['values'] == [3.0, None, 3.0]
    assert row['time_sec']['n'] == 2
    assert row['time_sec']['mean'] == 3.0


def test_summarize_empty() -> None:
    assert summarize([]) == {
        'schema_version': 1,
        'dataset': None,
        'metric': 'score',
        'std_ddof': 1,
        'methods': [],
    }


def test_summarize_k_from_data() -> None:
    runs = []
    for seed in range(2):
        report = homogeneous_report(seed)
        report['config']['n_models'] = 16
        runs.append(report)
    assert _row(summarize(runs), 'homogeneous')['display_name'] == (
        'MLP ensemble (homogeneous, K=16)'
    )
    # Without n_models in the config, K is the number of members.
    report = homogeneous_report(0)
    del report['config']['n_models']
    assert _row(summarize([report]), 'homogeneous')['display_name'] == (
        'MLP ensemble (homogeneous, K=4)'
    )


def test_summarize_prefers_top_level_n_models() -> None:
    # TabPack reports K at the top level; unfinished members are not in 'members'.
    report = tabpack_report(0)
    report['n_models'] = 32
    del report['config']['n_models']
    row = _row(summarize([report]), 'tabpack')
    assert row['n_models']['values'] == [32.0]
    member_mean = statistics.mean(0.80 + 0.01 * i for i in range(6))
    assert row['member_test']['values'] == [pytest.approx(member_mean)]


def test_summarize_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="Duplicate 'mlp' run for seed 1"):
        summarize([mlp_report(1), mlp_report(0), mlp_report(1)])


def test_summarize_rejects_mixed_datasets() -> None:
    other = mlp_report(1)
    other['dataset'] = 'adult'
    with pytest.raises(ValueError, match='different datasets'):
        summarize([mlp_report(0), other])


def test_summarize_rejects_two_conservative_aggregates() -> None:
    with pytest.raises(ValueError, match='More than one conservative'):
        summarize([conservative_aggregate([0]), conservative_aggregate([1])])


def test_summarize_rejects_malformed_reports() -> None:
    report = mlp_report(0)
    del report['seed']
    with pytest.raises(TypeError, match='seed'):
        summarize([report])
    report = mlp_report(0)
    del report['metrics']['val']['score']
    with pytest.raises(ValueError, match=r'metrics\.val\.score'):
        summarize([report])
    report = mlp_report(0)
    del report['method']
    with pytest.raises(TypeError, match='method'):
        summarize([report])
    aggregate = conservative_aggregate([0, 1])
    aggregate['scores']['test'] = [0.8]
    with pytest.raises(ValueError, match='one score per seed'):
        summarize([aggregate])
    aggregate = conservative_aggregate([0, 0])
    with pytest.raises(ValueError, match='duplicate seeds'):
        summarize([aggregate])


# ---------------------------------------------------------------------------------
# to_markdown / write_summary
# ---------------------------------------------------------------------------------

EXPECTED_MARKDOWN = """\
# Results: churn

Accuracy in %, mean ± sample std (ddof = 1) over seeds; n = number of seeds; \
time = mean wall-clock time per run.

| Method | Test acc. (%) | Val acc. (%) | n | Mean time (s) |
| :-- | --: | --: | --: | --: |
| MLP | 86.50 ± 0.50 | 86.00 ± 0.50 | 3 | 3.0 |
| MLP ensemble (homogeneous, K=4) | 86.20 ± 0.10 | 85.80 ± 0.10 | 3 | 5.0 |
| TabPack (reduced, single run) | 85.70 ± 0.10 | 86.30 ± 0.10 | 3 | 9.0 |
| TabPack (reduced, conservative) | 85.85 ± 0.10 | 86.35 ± 0.10 | 3 | 3.0 |

## Ensembles

| Method | Models | Ensemble size | Unique members | Member test acc. (%) \
| Best single member test acc. (%) |
| :-- | --: | --: | --: | --: | --: |
| MLP ensemble (homogeneous, K=4) | 4 | 4.0 | 4.0 | 85.25 ± 0.10 | 85.40 ± 0.10 |
| TabPack (reduced, single run) | 6 | 3.0 | 2.3 | 82.50 ± 0.00 | 85.00 ± 0.00 |
| TabPack (reduced, conservative) | 3 | 3.0 | 2.0 | 85.00 ± 0.00 | 86.00 ± 0.00 |

## Test accuracy per seed (%)

| Method | seed 0 | seed 1 | seed 2 |
| :-- | --: | --: | --: |
| MLP | 86.00 | 86.50 | 87.00 |
| MLP ensemble (homogeneous, K=4) | 86.10 | 86.30 | 86.20 |
| TabPack (reduced, single run) | 85.80 | 85.70 | 85.60 |
| TabPack (reduced, conservative) | 85.75 | 85.85 | 85.95 |

## Notes

* TabPack (reduced, conservative): the configs selected by \
`runs/churn/tabpack/seed-0` (ids 3, 4, 5) retrained with seeds 0, 1, 2; the std \
covers training noise only (no config sampling).
"""


def test_to_markdown_snapshot(runs_dir: Path) -> None:
    assert to_markdown(summarize(collect_runs(runs_dir))) == EXPECTED_MARKDOWN


def test_to_markdown_format_details() -> None:
    summary = summarize([mlp_report(0), mlp_report(1)])
    summary['methods'][0]['test'].update(mean=0.86124, std=0.00349)
    text = to_markdown(summary)
    assert '| MLP | 86.12 ± 0.35 | 85.75 ± 0.35 | 2 | 2.5 |' in text
    # No ensemble section for a plain MLP; per-seed table has only its seeds.
    assert '## Ensembles' not in text
    assert '| Method | seed 0 | seed 1 |' in text
    assert '## Notes' not in text


def test_to_markdown_missing_seeds_and_values() -> None:
    aggregate = conservative_aggregate([1, 2])
    summary = summarize([mlp_report(0), mlp_report(1), aggregate])
    text = to_markdown(summary)
    assert '| Method | seed 0 | seed 1 | seed 2 |' in text
    assert '| MLP | 86.00 | 86.50 | - |' in text
    assert '| TabPack (reduced, conservative) | - | 85.85 | 85.95 |' in text
    # No per-seed reports -> no time; the ensemble table still shows K.
    main_row = (
        '| TabPack (reduced, conservative) | 85.90 ± 0.07 | 86.40 ± 0.07 | 2 | - |'
    )
    assert main_row in text
    assert '| TabPack (reduced, conservative) | 3 | - | - | - | - |' in text


def test_to_markdown_empty() -> None:
    assert to_markdown(summarize([])) == '# Results\n\n_No runs found._\n'


def test_write_summary_files(runs_dir: Path, tmp_path: Path) -> None:
    summary = summarize(collect_runs(runs_dir))
    out_dir = tmp_path / 'results' / 'churn'
    write_summary(summary, out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        'summary.csv',
        'summary.json',
        'summary.md',
    ]

    loaded = json.loads((out_dir / 'summary.json').read_text())
    assert loaded == summary
    # The markdown can be regenerated from the JSON alone.
    assert to_markdown(loaded) == EXPECTED_MARKDOWN
    assert (out_dir / 'summary.md').read_text(encoding='utf-8') == EXPECTED_MARKDOWN

    with (out_dir / 'summary.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [r['method'] for r in rows] == list(METHOD_ORDER)
    assert [r['display_name'] for r in rows] == [
        r['display_name'] for r in summary['methods']
    ]
    for csv_row, row in zip(rows, summary['methods'], strict=True):
        assert int(csv_row['n']) == row['n']
        assert csv_row['seeds'] == '0;1;2'
        assert float(csv_row['test_mean']) == row['test']['mean']
        assert float(csv_row['test_std']) == row['test']['std']
        assert float(csv_row['val_max']) == row['val']['max']
        assert [float(v) for v in csv_row['test_values'].split(';')] == (
            row['test']['values']
        )
    mlp_csv = rows[0]
    assert mlp_csv['member_test_mean'] == ''
    assert mlp_csv['ensemble_size_mean'] == ''
    assert float(rows[2]['best_member_test_mean']) == pytest.approx(0.85)
    assert float(rows[3]['time_sec_mean']) == 3.0

    # Overwriting is fine and leaves no temporary files behind.
    write_summary(summary, out_dir)
    assert len(list(out_dir.iterdir())) == 3


def test_write_summary_empty(tmp_path: Path) -> None:
    write_summary(summarize([]), tmp_path / 'out')
    lines = (tmp_path / 'out' / 'summary.csv').read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith('method,display_name,n,seeds,test_mean,')
