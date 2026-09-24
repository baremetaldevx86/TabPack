"""Summaries across seeds and methods (a36).

Summary JSON (``summary.json``, the input of reporting/plots.py)::

    {
      "schema_version": 1,
      "dataset": "churn" | None,
      "metric": "score",       # the reports' metrics[part]["score"] (accuracy on Churn)
      "std_ddof": 1,
      "methods": [ROW, ...],   # canonical order, see METHOD_ORDER
    }

    ROW = {
      "method": str, "display_name": str,
      "n": int, "seeds": [int, ...] (sorted), "paths": [str | None, ...],
      "test": BLOCK, "val": BLOCK,
      "time_sec": BLOCK | None, "n_models": BLOCK | None,
      "ensemble_size": BLOCK | None, "ensemble_n_unique": BLOCK | None,
      "member_test": BLOCK | None,       # per seed: mean member test score
      "best_member_test": BLOCK | None,  # per seed: test score of the best-val member
      # 'tabpack-conservative' only:
      "aggregate_path": str, "source_run": str | None, "selected_ids": [int, ...],
    }

    BLOCK = {"n": int, "mean": float, "std": float, "min": float, "max": float,
             "values": [float | None, ...]}   # values aligned with ROW["seeds"]

Statistics ignore missing (None) values; "n" counts the present ones. The std is the
sample std (``statistics.stdev``, ddof = 1) and 0.0 for a single value. Scores are raw
fractions; only the markdown table converts accuracies to percentages.
"""

from __future__ import annotations

import csv
import io
import logging
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.utils.io import dump_json, load_json

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA_VERSION = 1
METHOD_ORDER = ('mlp', 'homogeneous', 'tabpack', 'tabpack-conservative')
CONSERVATIVE = 'tabpack-conservative'
CONSERVATIVE_SEED = 'tabpack-conservative-seed'
# Key under which collect_runs nests the per-seed reports of a conservative aggregate.
SEED_RUNS_KEY = 'seed_runs'

_STAT_KEYS = (
    'time_sec',
    'n_models',
    'ensemble_size',
    'ensemble_n_unique',
    'member_test',
    'best_member_test',
)
CSV_COLUMNS = (
    'method',
    'display_name',
    'n',
    'seeds',
    'test_mean',
    'test_std',
    'test_min',
    'test_max',
    'val_mean',
    'val_std',
    'val_min',
    'val_max',
    'time_sec_mean',
    'n_models_mean',
    'ensemble_size_mean',
    'ensemble_n_unique_mean',
    'member_test_mean',
    'member_test_std',
    'best_member_test_mean',
    'best_member_test_std',
    'test_values',
    'val_values',
)


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Load every report.json below runs_dir (recursively), adding "path".

    "path" is the path of the report.json file (``runs_dir / <relative path>``).
    Reports are returned in path order. To avoid double counting, a per-seed
    conservative report (method 'tabpack-conservative-seed') that lies below the
    directory of a conservative aggregate report (method 'tabpack-conservative') is
    not returned at the top level; it is nested, sorted by seed, in the aggregate's
    ``"seed_runs"`` list instead (the nearest aggregate above it wins). The scores of
    the conservative protocol always come from the aggregate; the nested reports only
    add times, member scores and ensemble sizes.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f'Runs directory not found: {runs_dir}')
    paths = sorted(runs_dir.rglob('report.json'), key=lambda p: p.parts)
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        report = load_json(path)
        if not isinstance(report, dict):
            raise TypeError(f'{path}: expected a JSON object, got {type(report)}')
        version = report.get('schema_version')
        if version is not None and version != RUN_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f'{path}: unsupported report schema_version {version!r}'
                f' (expected {RUN_REPORT_SCHEMA_VERSION})'
            )
        report['path'] = str(path)
        reports.append(report)

    aggregates = {
        Path(r['path']).parent: r for r in reports if r.get('method') == CONSERVATIVE
    }
    for aggregate in aggregates.values():
        aggregate[SEED_RUNS_KEY] = []
    runs = []
    for report in reports:
        if report.get('method') == CONSERVATIVE_SEED:
            owner = _find_aggregate(Path(report['path']).parent, aggregates)
            if owner is not None:
                owner[SEED_RUNS_KEY].append(report)
                continue
            logger.warning(
                '%s: per-seed conservative report without an aggregate report above'
                ' it; it is not summarized',
                report['path'],
            )
        runs.append(report)
    for aggregate in aggregates.values():
        aggregate[SEED_RUNS_KEY].sort(key=lambda r: (_seed_sort_key(r), r['path']))
    return runs


def _find_aggregate(
    directory: Path, aggregates: dict[Path, dict[str, Any]]
) -> dict[str, Any] | None:
    # The per-seed report's own directory is never the aggregate's directory (both
    # files are named report.json), so start one level up.
    for parent in directory.parents:
        if parent in aggregates:
            return aggregates[parent]
    return None


def _seed_sort_key(report: dict[str, Any]) -> tuple[int, int]:
    seed = report.get('seed')
    return (0, seed) if isinstance(seed, int) else (1, 0)


# ---------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------


def _block(values: list[float | None]) -> dict[str, Any] | None:
    """Stats over the non-None values; None if there are none."""
    values = [None if v is None else float(v) for v in values]
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {
        'n': len(present),
        'mean': statistics.mean(present),
        'std': statistics.stdev(present) if len(present) > 1 else 0.0,
        'min': min(present),
        'max': max(present),
        'values': values,
    }


def _score(metrics: Any, part: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    part_metrics = metrics.get(part)
    if not isinstance(part_metrics, dict):
        return None
    score = part_metrics.get('score')
    return None if score is None else float(score)


def _required_score(report: dict[str, Any], part: str) -> float:
    score = _score(report.get('metrics'), part)
    if score is None:
        raise ValueError(f'{_where(report)}: missing metrics.{part}.score')
    return score


def _where(report: dict[str, Any]) -> str:
    return str(report.get('path', f'report of method {report.get("method")!r}'))


def _n_models(report: dict[str, Any]) -> int | None:
    """K: report["n_models"] (TabPack), else config["n_models"], else None."""
    n_models = report.get('n_models')
    if n_models is None and isinstance(report.get('config'), dict):
        n_models = report['config'].get('n_models')
    return n_models


def _is_ensemble(report: dict[str, Any]) -> bool:
    return (
        isinstance(report.get('ensemble'), dict)
        or len(report.get('members') or []) > 1
        or _n_models(report) is not None
    )


def _run_stats(report: dict[str, Any] | None) -> dict[str, float | None]:
    """Per-run values of the optional statistics (None when unavailable)."""
    stats: dict[str, float | None] = dict.fromkeys(_STAT_KEYS)
    if report is None:
        return stats
    time_sec = report.get('time_sec')
    stats['time_sec'] = None if time_sec is None else float(time_sec)
    if not _is_ensemble(report):
        return stats

    # Members are the FINISHED members; TabPack may stop before all K finish.
    members = report.get('members') or []
    n_models = _n_models(report)
    stats['n_models'] = len(members) if n_models is None and members else n_models

    ensemble = report.get('ensemble')
    if isinstance(ensemble, dict):
        size = ensemble.get('size')
        if size is None and ensemble.get('ids') is not None:
            size = len(ensemble['ids'])
        n_unique = ensemble.get('n_unique')
        if n_unique is None and ensemble.get('ids') is not None:
            n_unique = len(set(ensemble['ids']))
        stats['ensemble_size'] = size
        stats['ensemble_n_unique'] = n_unique
    elif members:
        # No selection: the final prediction is the uniform average of all members.
        stats['ensemble_size'] = len(members)
        stats['ensemble_n_unique'] = len(members)

    member_scores = [_score(m.get('metrics'), 'test') for m in members]
    member_scores = [s for s in member_scores if s is not None]
    if member_scores:
        stats['member_test'] = statistics.mean(member_scores)

    best = report.get('best_member')
    if isinstance(best, dict):
        stats['best_member_test'] = _score(best.get('metrics'), 'test')
    return stats


# ---------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method mean/std/min/max/n of test and val scores + mean time.

    Rows (in this order when present): 'mlp', 'homogeneous', 'tabpack' (single-run
    online ensemble over seeds), 'tabpack-conservative' (paper protocol; uses the
    aggregate report's per-seed scores). Also: mean individual-member test score for
    ensembles, mean ensemble size. Uses the sample std (ddof=1), as statistics.stdev.

    The exact output schema is documented in this module's docstring. Top-level
    'tabpack-conservative-seed' reports are ignored (they only count through their
    aggregate, see collect_runs). Raises ValueError on a duplicate (method, seed),
    on reports of different datasets, on more than one conservative aggregate and on
    a missing val/test score; TypeError on a missing/non-integer seed or method.
    """
    by_method: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        method = run.get('method')
        if not isinstance(method, str):
            raise TypeError(f'{_where(run)}: missing or non-string "method"')
        if method == CONSERVATIVE_SEED:
            logger.warning(
                '%s: ignoring a per-seed conservative report outside of its aggregate',
                _where(run),
            )
            continue
        by_method.setdefault(method, []).append(run)

    known = [m for m in METHOD_ORDER if m in by_method]
    unknown = sorted(m for m in by_method if m not in METHOD_ORDER)
    rows = []
    datasets: set[str] = set()
    for method in known + unknown:
        if method == CONSERVATIVE:
            if len(by_method[method]) > 1:
                paths = ', '.join(_where(r) for r in by_method[method])
                raise ValueError(
                    f'More than one conservative aggregate report: {paths}'
                )
            row, run_datasets = _conservative_row(by_method[method][0])
        else:
            row, run_datasets = _method_row(method, by_method[method])
        rows.append(row)
        datasets |= run_datasets
    if len(datasets) > 1:
        raise ValueError(f'Reports of different datasets: {sorted(datasets)}')
    return {
        'schema_version': SUMMARY_SCHEMA_VERSION,
        'dataset': next(iter(datasets)) if datasets else None,
        'metric': 'score',
        'std_ddof': 1,
        'methods': rows,
    }


def _datasets(reports: Iterable[dict[str, Any]]) -> set[str]:
    return {r['dataset'] for r in reports if isinstance(r.get('dataset'), str)}


def _method_row(
    method: str, reports: list[dict[str, Any]]
) -> tuple[dict[str, Any], set[str]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for report in reports:
        seed = report.get('seed')
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f'{_where(report)}: missing or non-integer "seed"')
        if seed in by_seed:
            raise ValueError(
                f'Duplicate {method!r} run for seed {seed}:'
                f' {_where(by_seed[seed])} and {_where(report)}'
            )
        by_seed[seed] = report
    seeds = sorted(by_seed)
    ordered = [by_seed[s] for s in seeds]
    row = _row(
        method,
        seeds=seeds,
        paths=[r.get('path') for r in ordered],
        test=[_required_score(r, 'test') for r in ordered],
        val=[_required_score(r, 'val') for r in ordered],
        per_run=[_run_stats(r) for r in ordered],
    )
    return row, _datasets(ordered)


def _conservative_row(aggregate: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    where = _where(aggregate)
    seeds = aggregate.get('seeds')
    scores = aggregate.get('scores')
    if not isinstance(seeds, list) or not isinstance(scores, dict):
        raise TypeError(f'{where}: the aggregate needs "seeds" and "scores"')
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise TypeError(f'{where}: non-integer seeds {seeds!r}')
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'{where}: duplicate seeds {seeds!r}')
    per_part = {}
    for part in ('test', 'val'):
        values = scores.get(part)
        if not isinstance(values, list):
            raise TypeError(f'{where}: scores.{part} must be a list')
        if len(values) != len(seeds):
            raise ValueError(f'{where}: scores.{part} needs one score per seed')
        if any(v is None for v in values):
            raise ValueError(f'{where}: scores.{part} contains a missing score')
        per_part[part] = dict(zip(seeds, values, strict=True))
    n_seeds = aggregate.get('n_seeds')
    if n_seeds is not None and n_seeds != len(seeds):
        logger.warning('%s: n_seeds=%s but %d seeds', where, n_seeds, len(seeds))

    seed_runs: dict[int, dict[str, Any]] = {}
    for report in aggregate.get(SEED_RUNS_KEY) or []:
        seed = report.get('seed')
        if not isinstance(seed, int) or seed not in per_part['test']:
            logger.warning(
                '%s: seed %r is not in the aggregate %s; ignored',
                _where(report),
                seed,
                where,
            )
        elif seed in seed_runs:
            raise ValueError(
                f'Duplicate conservative run for seed {seed}:'
                f' {_where(seed_runs[seed])} and {_where(report)}'
            )
        else:
            seed_runs[seed] = report

    selected_ids = aggregate.get('selected_ids')
    ordered_seeds = sorted(seeds)
    per_run = []
    for seed in ordered_seeds:
        stats = _run_stats(seed_runs.get(seed))
        if stats['n_models'] is None and isinstance(selected_ids, list):
            stats['n_models'] = len(selected_ids)
        per_run.append(stats)
    row = _row(
        CONSERVATIVE,
        seeds=ordered_seeds,
        paths=[
            seed_runs[s].get('path') if s in seed_runs else None for s in ordered_seeds
        ],
        test=[per_part['test'][s] for s in ordered_seeds],
        val=[per_part['val'][s] for s in ordered_seeds],
        per_run=per_run,
    )
    row['aggregate_path'] = aggregate.get('path')
    row['source_run'] = aggregate.get('source_run')
    row['selected_ids'] = selected_ids
    return row, _datasets([aggregate, *seed_runs.values()])


def _row(
    method: str,
    *,
    seeds: list[int],
    paths: list[str | None],
    test: list[float],
    val: list[float],
    per_run: list[dict[str, float | None]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        'method': method,
        'display_name': method,
        'n': len(seeds),
        'seeds': seeds,
        'paths': paths,
        'test': _block(test),
        'val': _block(val),
    }
    for key in _STAT_KEYS:
        row[key] = _block([stats[key] for stats in per_run])
    row['display_name'] = _display_name(row)
    return row


def _format_count(block: dict[str, Any] | None) -> str | None:
    if block is None:
        return None
    low, high = block['min'], block['max']
    if low == high:
        return f'{low:g}'
    return f'{low:g}-{high:g}'


def _display_name(row: dict[str, Any]) -> str:
    method = row['method']
    if method == 'mlp':
        return 'MLP'
    if method == 'homogeneous':
        k = _format_count(row['n_models']) or _format_count(row['ensemble_size'])
        if k is None:
            return 'MLP ensemble (homogeneous)'
        return f'MLP ensemble (homogeneous, K={k})'
    if method == 'tabpack':
        return 'TabPack (reduced, single run)'
    if method == CONSERVATIVE:
        return 'TabPack (reduced, conservative)'
    return method


# ---------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return '-' if value is None else f'{100 * value:.2f}'


def _pct_block(block: dict[str, Any] | None) -> str:
    if block is None:
        return '-'
    return f'{100 * block["mean"]:.2f} ± {100 * block["std"]:.2f}'


def _mean(block: dict[str, Any] | None, fmt: str) -> str:
    return '-' if block is None else format(block['mean'], fmt)


def _table(header: list[str], align: list[str], rows: list[list[str]]) -> list[str]:
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(cells) + ' |' for cells in rows]
    return lines


def to_markdown(summary: dict[str, Any]) -> str:
    """A GitHub-flavored markdown table: method | test acc (mean ± std) | val acc |
    n seeds | mean time.

    Followed by an ensemble table (models, ensemble size, mean member and best single
    member test accuracy), a per-seed test accuracy table and notes. Accuracies are
    percentages with 2 decimals, mean ± sample std over seeds.
    """
    rows = summary.get('methods') or []
    dataset = summary.get('dataset')
    lines = [f'# Results: {dataset}' if dataset else '# Results', '']
    if not rows:
        return '\n'.join([*lines, '_No runs found._', ''])
    lines += [
        (
            'Accuracy in %, mean ± sample std (ddof = 1) over seeds;'
            ' n = number of seeds; time = mean wall-clock time per run.'
        ),
        '',
    ]
    lines += _table(
        ['Method', 'Test acc. (%)', 'Val acc. (%)', 'n', 'Mean time (s)'],
        [':--', '--:', '--:', '--:', '--:'],
        [
            [
                row['display_name'],
                _pct_block(row['test']),
                _pct_block(row['val']),
                str(row['n']),
                _mean(row.get('time_sec'), '.1f'),
            ]
            for row in rows
        ],
    )

    ensembles = [
        row
        for row in rows
        if any(row.get(key) is not None for key in _STAT_KEYS if key != 'time_sec')
    ]
    if ensembles:
        lines += ['', '## Ensembles', '']
        lines += _table(
            [
                'Method',
                'Models',
                'Ensemble size',
                'Unique members',
                'Member test acc. (%)',
                'Best single member test acc. (%)',
            ],
            [':--', '--:', '--:', '--:', '--:', '--:'],
            [
                [
                    row['display_name'],
                    _format_count(row.get('n_models')) or '-',
                    _mean(row.get('ensemble_size'), '.1f'),
                    _mean(row.get('ensemble_n_unique'), '.1f'),
                    _pct_block(row.get('member_test')),
                    _pct_block(row.get('best_member_test')),
                ]
                for row in ensembles
            ],
        )

    all_seeds = sorted({seed for row in rows for seed in row['seeds']})
    lines += ['', '## Test accuracy per seed (%)', '']
    per_seed_rows = []
    for row in rows:
        by_seed = dict(zip(row['seeds'], row['test']['values'], strict=True))
        per_seed_rows.append(
            [row['display_name'], *[_pct(by_seed.get(s)) for s in all_seeds]]
        )
    lines += _table(
        ['Method', *[f'seed {s}' for s in all_seeds]],
        [':--', *['--:'] * len(all_seeds)],
        per_seed_rows,
    )

    notes = []
    for row in rows:
        if row['method'] == CONSERVATIVE:
            ids = row.get('selected_ids')
            ids_text = (
                ', '.join(str(i) for i in ids) if isinstance(ids, list) else 'unknown'
            )
            source = row.get('source_run') or 'unknown'
            notes.append(
                f'* {row["display_name"]}: the configs selected by `{source}`'
                f' (ids {ids_text}) retrained with seeds'
                f' {", ".join(str(s) for s in row["seeds"])}; the std covers'
                ' training noise only (no config sampling).'
            )
    if notes:
        lines += ['', '## Notes', '', *notes]
    return '\n'.join(lines) + '\n'


def _csv_value(value: Any) -> Any:
    return '' if value is None else value


def _stat(row: dict[str, Any], key: str, stat: str) -> Any:
    block = row.get(key)
    return None if block is None else block[stat]


def _join(values: list[Any]) -> str:
    return ';'.join('' if v is None else str(v) for v in values)


def _to_csv(summary: dict[str, Any]) -> str:
    """One row per method with the raw (fraction) statistics; see CSV_COLUMNS."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(CSV_COLUMNS)
    for row in summary.get('methods') or []:
        record = {
            'method': row['method'],
            'display_name': row['display_name'],
            'n': row['n'],
            'seeds': _join(row['seeds']),
            'test_values': _join(row['test']['values']),
            'val_values': _join(row['val']['values']),
        }
        for key in ('test', 'val'):
            for stat in ('mean', 'std', 'min', 'max'):
                record[f'{key}_{stat}'] = _stat(row, key, stat)
        for key in _STAT_KEYS:
            record[f'{key}_mean'] = _stat(row, key, 'mean')
        for key in ('member_test', 'best_member_test'):
            record[f'{key}_std'] = _stat(row, key, 'std')
        writer.writerow([_csv_value(record[column]) for column in CSV_COLUMNS])
    return buffer.getvalue()


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(text, encoding='utf-8', newline='')
    tmp.replace(path)


def write_summary(summary: dict[str, Any], out_dir: str | Path) -> None:
    """Write summary.json, summary.md and summary.csv into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / 'summary.json', summary)
    _write_text(out_dir / 'summary.md', to_markdown(summary))
    _write_text(out_dir / 'summary.csv', _to_csv(summary))
