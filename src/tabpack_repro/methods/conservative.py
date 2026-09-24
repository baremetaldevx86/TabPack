"""The paper's conservative evaluation protocol for TabPack (a33).

Read ``<source_run>/report.json`` (a TabPack main run), take the sorted unique ids of
its final online ensemble, look up their configs in report["members"], and for each
seed s in range(n_seeds) run methods.tabpack.run with ``configs=selected``,
``n_models=len(selected)``, ``seed=s`` (all other settings copied from the source
run's config) into ``<output_dir>/seed-<s>``. Aggregate as described in
methods/report.py. Must be resumable: skip seeds whose report.json exists.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import statistics
from pathlib import Path
from typing import Any

from tabpack_repro.config import (
    ConservativeEvalConfig,
    TabPackConfig,
    config_from_dict,
    config_to_dict,
    dump_config,
    load_config,
)
from tabpack_repro.methods import tabpack as _tabpack
from tabpack_repro.methods.common import (
    CONFIG_FILENAME,
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
)
from tabpack_repro.methods.report import RUN_REPORT_SCHEMA_VERSION
from tabpack_repro.utils.io import dump_json, load_json, to_jsonable

_METHOD = 'tabpack-conservative'
_SEED_METHOD = 'tabpack-conservative-seed'
_SOURCE_METHOD = 'tabpack'
_PARTS = ('val', 'test')


def run(config: ConservativeEvalConfig, output_dir: str | Path) -> dict[str, Any]:
    """Retrain the configs selected by a TabPack main run with seeds 0..n_seeds-1.

    Writes ``<output_dir>/seed-<s>/`` (a full TabPack run whose report has method
    "tabpack-conservative-seed") for every seed, then the aggregate
    ``<output_dir>/report.json`` (and ``config.toml``), and returns the aggregate.
    Member i of every retraining run is the source member ``selected_ids[i]``.
    Seeds that already have a finished report are not retrained, only aggregated;
    a finished seed whose config differs from the expected one raises ValueError.
    """
    if not isinstance(config, ConservativeEvalConfig):
        raise TypeError(
            f'expected a ConservativeEvalConfig, got {type(config).__name__}'
        )
    n_seeds = config.n_seeds
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int) or n_seeds < 1:
        raise ValueError(f'n_seeds must be a positive int, got {n_seeds!r}')
    if not config.source_run:
        raise ValueError('source_run must be the directory of a finished TabPack run')

    source_dir = Path(config.source_run)
    output_dir = Path(output_dir)
    _check_output_dir(output_dir, source_dir)
    source_report, source_config = _load_source(source_dir)
    selected_ids = _selected_ids(source_report)
    selected_configs = _member_configs(source_report, selected_ids)

    seeds = list(range(n_seeds))
    seed_reports = [
        _run_seed(
            _seed_config(source_config, selected_configs, seed),
            output_dir / f'seed-{seed}',
        )
        for seed in seeds
    ]

    scores = {
        part: [_score(r, part, s) for r, s in zip(seed_reports, seeds, strict=True)]
        for part in _PARTS
    }
    report = {
        'schema_version': RUN_REPORT_SCHEMA_VERSION,
        'method': _METHOD,
        'dataset': source_report.get('dataset'),
        'config': config_to_dict(config),
        'source_run': str(config.source_run),
        'source_seed': source_report.get('seed'),
        'selected_ids': selected_ids,
        'n_seeds': n_seeds,
        'seeds': seeds,
        'scores': scores,
        'mean': {part: statistics.fmean(v) for part, v in scores.items()},
        # Sample std (ddof=1), as statistics.stdev; a single seed has no spread.
        'std': {
            part: statistics.stdev(v) if len(v) > 1 else 0.0
            for part, v in scores.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / REPORT_FILENAME, report)
    dump_config(config, output_dir / CONFIG_FILENAME)
    return report


def _check_output_dir(output_dir: Path, source_dir: Path) -> None:
    """Refuse output directories whose report.json belongs to another run."""
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError('output_dir must differ from source_run')
    report_path = output_dir / REPORT_FILENAME
    if report_path.is_file():
        report = load_json(report_path)
        method = report.get('method') if isinstance(report, dict) else None
        if method != _METHOD:
            raise ValueError(
                f'{report_path} is not a conservative-evaluation report'
                f' (method {method!r}); refusing to overwrite it'
            )


def _load_source(source_dir: Path) -> tuple[dict[str, Any], TabPackConfig]:
    """The source run's report and its TabPackConfig.

    The config comes from report["config"] (lossless JSON, and in the same file as
    the members and the ensemble, so the three cannot disagree); config.toml is only
    a fallback for reports without a config.
    """
    report_path = source_dir / REPORT_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(
            f'the source run {source_dir} has no {REPORT_FILENAME}'
            ' (run the TabPack main run first)'
        )
    report = load_json(report_path)
    method = report.get('method') if isinstance(report, dict) else None
    if method != _SOURCE_METHOD:
        raise ValueError(
            f'the source run must be a {_SOURCE_METHOD!r} run,'
            f' but {report_path} has method {method!r}'
        )
    if report.get('config') is not None:
        source_config = config_from_dict(copy.deepcopy(report['config']))
    elif (source_dir / CONFIG_FILENAME).is_file():
        source_config = load_config(source_dir / CONFIG_FILENAME)
    else:
        raise ValueError(
            f'the source run {source_dir} has neither report["config"]'
            f' nor {CONFIG_FILENAME}'
        )
    if source_config.method != _SOURCE_METHOD:
        raise ValueError(
            'the source run config must be a TabPackConfig,'
            f' got {type(source_config).__name__}'
        )
    return report, source_config


def _selected_ids(report: dict[str, Any]) -> list[int]:
    """Sorted unique member ids of the source run's final online ensemble."""
    ensemble = report.get('ensemble')
    ids = ensemble.get('ids') if isinstance(ensemble, dict) else None
    if not isinstance(ids, list) or not ids:
        raise ValueError('the source run report has no (non-empty) ensemble ids')
    not_ints = [i for i in ids if not _is_int(i)]
    if not_ints:
        raise ValueError(f'ensemble ids must be ints, got {not_ints!r}')
    return sorted(set(ids))


def _member_configs(
    report: dict[str, Any], selected_ids: list[int]
) -> list[dict[str, Any]]:
    """The member configs of `selected_ids` (in that order) from report["members"]."""
    members = report.get('members')
    if not isinstance(members, list) or not members:
        raise ValueError('the source run report has no "members" list')
    configs: dict[int, Any] = {}
    for member in members:
        member_id = member.get('id') if isinstance(member, dict) else None
        if not _is_int(member_id):
            raise ValueError(f'malformed member entry in the source report: {member!r}')
        if member_id in configs:
            raise ValueError(
                f'member id {member_id} appears twice in the source report'
            )
        configs[member_id] = member.get('config')
    missing = [i for i in selected_ids if i not in configs]
    if missing:
        raise ValueError(
            f'ensemble member ids {missing} are missing from the source report members'
        )
    no_config = [i for i in selected_ids if not isinstance(configs[i], dict)]
    if no_config:
        raise ValueError(
            f'ensemble member ids {no_config} have no config in the source report'
        )
    return [copy.deepcopy(configs[i]) for i in selected_ids]


def _seed_config(
    source: TabPackConfig, configs: list[dict[str, Any]], seed: int
) -> TabPackConfig:
    """The source config with the official evaluation overrides.

    As the official ``_evaluate_ensemble``: n_models and configs are replaced by the
    selected ones, the optimizer uses a shared step, and the seed is the evaluation
    seed. The sampler is dropped: ``space`` is kept for provenance but is ignored by
    TabPack when ``configs`` is set. Everything else (data, d_block, training, online
    ensemble) is copied unchanged.
    """
    base = copy.deepcopy(source)
    return dataclasses.replace(
        base,
        seed=seed,
        n_models=len(configs),
        configs=copy.deepcopy(configs),
        optimizer=dataclasses.replace(base.optimizer, shared_step=True),
    )


def _finished_report(seed_dir: Path) -> dict[str, Any] | None:
    """The report of a finished seed run, or None if the seed must be (re)trained.

    A report with method "tabpack" means TabPack finished but the method field was
    not rewritten yet (e.g. an interruption right after tabpack.run). It counts as
    finished only when TabPack also wrote its other artifacts, because report.json
    is the first file it writes.
    """
    report_path = seed_dir / REPORT_FILENAME
    if not report_path.is_file():
        return None
    report = load_json(report_path)
    method = report.get('method') if isinstance(report, dict) else None
    if method == _SEED_METHOD:
        return report
    if method == _SOURCE_METHOD:
        artifacts = (PREDICTIONS_FILENAME, CONFIG_FILENAME)
        return report if all((seed_dir / f).is_file() for f in artifacts) else None
    raise ValueError(
        f'{report_path} is not a conservative-evaluation seed report'
        f' (method {method!r}); refusing to overwrite it'
    )


def _run_seed(seed_config: TabPackConfig, seed_dir: Path) -> dict[str, Any]:
    report = _finished_report(seed_dir)
    if report is None:
        _tabpack.run(seed_config, seed_dir)
        report = load_json(seed_dir / REPORT_FILENAME)
    # Guards resuming into a directory made from another source run or settings.
    expected = to_jsonable(config_to_dict(seed_config))
    if report.get('config') != expected:
        raise ValueError(
            f'{seed_dir / REPORT_FILENAME} was trained with a different config than'
            ' the one this conservative evaluation expects (another source run or'
            ' settings?); use a fresh output directory'
        )
    if report.get('method') != _SEED_METHOD:
        report['method'] = _SEED_METHOD
        dump_json(seed_dir / REPORT_FILENAME, report)
    return report


def _score(report: dict[str, Any], part: str, seed: int) -> float:
    try:
        score = report['metrics'][part]['score']
    except (KeyError, TypeError) as err:
        raise ValueError(
            f'the seed-{seed} report has no metrics[{part!r}]["score"]'
        ) from err
    if not (_is_real(score) and math.isfinite(score)):
        raise ValueError(
            f'the seed-{seed} {part} score is not a finite number: {score!r}'
        )
    return float(score)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
