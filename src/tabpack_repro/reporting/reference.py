"""Official Churn numbers for comparison (a38).

Reads the report.json files shipped in the official repository (cloned outside this
repo; path from $TABPACK_REFERENCE_DIR) and extracts Churn results, e.g.
experiments/tabpack/churn/main/report.json (64 models, A100):
online_ensembles.greedy.report.metrics.test.score, best single model, mean member
score, n_models, time. Writes ``results/reference/churn_official.json`` so the
comparison works without the clone.

Layout of the official ``experiments`` directory (official README, "Overview")::

    experiments/<variant>/churn/main/report.json            the "main" run
    experiments/<variant>/churn/eval-<online|offline>-ensembles/<ensemble>/
        evaluation/report.json                              the "evaluation" run

The evaluation run is the paper's *conservative* protocol: the configs of the unique
members of the main run's final ensemble are retrained with ``n_seeds`` seeds. The
paper's number is the mean and the sample std (``statistics.stdev``) of the ensemble's
test score over those seeds (official README, "Metrics"). For TabPack on Churn this
gives 0.8575 +- 0.0014, which is the value in Table 14 (Appendix G) of the paper.

Everything here is read-only JSON parsing; nothing from the official code is
imported or copied. Regenerate the committed file with::

    tools/dev/py -m tabpack_repro.reporting.reference \\
        --output results/reference/churn_official.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

REFERENCE_URL = 'https://github.com/yandex-research/tabpack'
REFERENCE_COMMIT = '05a89e21b955f12de84889d662e15ca534019aaa'
SCHEMA_VERSION = 1
DATASET = 'churn'
DEFAULT_OUTPUT = 'results/reference/churn_official.json'

# Variant directory -> model name in the paper (official README, "Models").
_PAPER_NAMES = {
    'tabpack': 'TabPack',
    'tabpack-cosine': 'TabPack†',
    'tabpack-cosine-macbook': 'TabPack†_MacBook',
    'tabpack-cosine-offline': 'TabPack†_Offline',
}

# Churn block of Table 14 (Appendix G, "Per-Dataset Results") of arXiv:2607.05380v1,
# transcribed by a38 from https://arxiv.org/html/2607.05380 (retrieved 2026-09-24).
# Not part of the official clone; used only to cross-check the evaluation reports.
_PAPER_SOURCE = (
    'arXiv:2607.05380v1, Appendix G "Per-Dataset Results", Table 14, Churn block '
    '(test accuracy, mean ± std); transcribed from https://arxiv.org/html/2607.05380 '
    'on 2026-09-24; not part of the official clone'
)
_PAPER_CAPTION = (
    'Per-dataset performance for the methods. For each dataset, we report the mean '
    'and standard deviation of test metric across ten random seeds.'
)
_PAPER_TABLE14_CHURN: dict[str, tuple[float, float]] = {
    'MLP': (0.8562, 0.0018),
    'XGBoost': (0.8605, 0.0020),
    'TabICLv2': (0.8658, 0.0008),
    'TabPFN-3': (0.8655, 0.0009),
    'ModernNCA†': (0.8591, 0.0027),
    'RealMLP': (0.8599, 0.0023),
    'TabM': (0.8638, 0.0017),
    'MLP†': (0.8633, 0.0029),
    'TabM†': (0.8609, 0.0014),
    'TabPack': (0.8575, 0.0014),
    'MLP†_HPE': (0.8624, 0.0013),
    'TabPack†_Offline': (0.8562, 0.0048),
    'TabPack†_MacBook': (0.8588, 0.0052),
    'TabPack†': (0.8623, 0.0028),
}

_EVAL_DIR_RE = re.compile(r'^eval-(online|offline)-ensembles$')
_HEX40_RE = re.compile(r'^[0-9a-f]{40}$')


# ---------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------
def get_reference_dir() -> Path | None:
    """$TABPACK_REFERENCE_DIR if it exists, else <repo>/.reference/tabpack if it
    exists, else None."""
    env = os.environ.get('TABPACK_REFERENCE_DIR', '').strip()
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path
    default = _project_root() / '.reference' / 'tabpack'
    return default if default.is_dir() else None


def extract_churn_reference(reference_dir: str | Path) -> dict[str, Any]:
    """Collect Churn numbers from every experiments/*/churn/**/report.json."""
    root = Path(reference_dir).expanduser()
    experiments_dir = root / 'experiments'
    if not experiments_dir.is_dir():
        raise FileNotFoundError(
            f'{experiments_dir} does not exist; is {root} a clone of {REFERENCE_URL}?'
        )

    variants: dict[str, Any] = {}
    other_reports: list[str] = []
    for dataset_dir in sorted(experiments_dir.glob(f'*/{DATASET}')):
        if not dataset_dir.is_dir():
            continue
        name = dataset_dir.parent.name
        variant, unclassified = _extract_variant(root, dataset_dir)
        variants[name] = variant
        other_reports.extend(unclassified)

    commit = _git_head(root)
    notes = [
        'Scores are accuracies (the official "score" for binclass tasks).',
        (
            'main: the single official run (seed 0). n_models in the report counts '
            'the members that FINISHED; the run stops early when the online ensemble '
            'runs out of patience, so n_models_configured - n_models members were '
            'still running and are not in the member statistics.'
        ),
        (
            'evaluations: the conservative protocol of the paper. The configs of the '
            "unique members of the main run's final ensemble (sorted by id) are "
            'retrained for n_seeds seeds; the reported number is the mean and the '
            'sample std (ddof=1) of the ensemble test score over the seeds '
            '(provenance.conservative_recipe).'
        ),
        (
            'headline is the TabPack conservative number (docs/EXPERIMENT.md quotes '
            'it as "85.75 ± 0.14"); paper_crosscheck compares every conservative '
            'number with paper_table14 rounded to 4 decimals.'
        ),
        (
            'The Table 14 caption says "ten random seeds", but the shipped Churn '
            'evaluation reports have the seed counts in paper_crosscheck, and those '
            'seeds alone reproduce the paper values.'
        ),
        'Member and seed statistics use the sample std (ddof=1), as statistics.stdev.',
        (
            'Inside evaluations, member ids (per_seed[*].ensemble.ids, '
            'per_seed[*].best_single.id) index the retrained configs; '
            'selected_main_ids maps them back to main-run member ids.'
        ),
    ]
    result: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'dataset': DATASET,
        'metric': 'accuracy',
        'provenance': {
            'repository': REFERENCE_URL,
            'commit': commit,
            'pinned_commit': REFERENCE_COMMIT,
            'commit_matches_pinned': commit == REFERENCE_COMMIT,
            'paths': 'relative to the root of the official clone',
            'extracted_by': 'tabpack_repro.reporting.reference.extract_churn_reference',
            'conservative_recipe': (
                'README.md "Metrics": for x in report["experiments"] of '
                'experiments/<variant>/<dataset>/eval-online-ensembles/greedy/'
                'evaluation/report.json take x["report"]["online_ensembles"]'
                '["greedy"]["report"]["metrics"]["test"]["score"]; report '
                'statistics.mean and statistics.stdev'
            ),
        },
        'variants': variants,
        'other_reports': sorted(other_reports),
        'summary': {name: _summary_row(name, v) for name, v in variants.items()},
        'headline': _headline(variants),
        'paper_crosscheck': {
            name: {
                'paper_name': v['paper_name'],
                'n_seeds': v['conservative'].get('n_seeds'),
                'matches_paper_table14': v['conservative'].get('matches_paper_table14'),
            }
            for name, v in variants.items()
        },
        'paper_table14': {
            'source': _PAPER_SOURCE,
            'caption': _PAPER_CAPTION,
            'churn': {
                method: {'mean': mean, 'std': std}
                for method, (mean, std) in _PAPER_TABLE14_CHURN.items()
            },
        },
        'notes': notes,
    }
    return _sort_keys(result)


def load_saved_reference(
    path: str | Path = 'results/reference/churn_official.json',
) -> dict[str, Any]:
    """Load the committed reference JSON.

    A relative ``path`` that does not exist relative to the working directory is
    also looked up relative to the repository root.
    """
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = _project_root() / path
        if candidate.exists():
            path = candidate
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict) or 'variants' not in data:
        raise ValueError(f'{path} is not a reference file (no "variants" key)')
    return data


# ---------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------
def _extract_variant(root: Path, dataset_dir: Path) -> tuple[dict[str, Any], list]:
    name = dataset_dir.parent.name
    main: dict[str, Any] | None = None
    member_configs: dict[int, Any] = {}
    evaluations: dict[str, Any] = {}
    unclassified: list[str] = []
    for report_path in sorted(dataset_dir.rglob('report.json')):
        rel_parts = report_path.relative_to(dataset_dir).parts
        eval_match = _EVAL_DIR_RE.match(rel_parts[0])
        if rel_parts == ('main', 'report.json'):
            main, member_configs = _extract_main(root, report_path)
        elif eval_match and len(rel_parts) == 4 and rel_parts[2] == 'evaluation':
            key = f'{rel_parts[0]}/{rel_parts[1]}'
            evaluations[key] = _extract_evaluation(
                root, report_path, kind=eval_match.group(1), ensemble_name=rel_parts[1]
            )
        else:
            unclassified.append(_rel(root, report_path))

    for evaluation in evaluations.values():
        _link_evaluation_to_main(evaluation, main, member_configs)

    primary = _primary_evaluation(evaluations)
    paper_name = _PAPER_NAMES.get(name)
    paper = _PAPER_TABLE14_CHURN.get(paper_name) if paper_name else None
    conservative: dict[str, Any]
    if primary is None:
        conservative = {
            'available': False,
            'note': (
                'no experiments/'
                f'{name}/{DATASET}/eval-*-ensembles/*/evaluation/report.json '
                'in the clone'
            ),
        }
    else:
        key, evaluation = primary
        test = evaluation['test']
        conservative = {
            'available': True,
            'evaluation': key,
            'report_path': evaluation['report_path'],
            'n_seeds': test['n'],
            'test_mean': test['mean'],
            'test_std': test['std'],
            'text': _pm_text(test['mean'], test['std']),
        }
        if paper is not None:
            conservative['matches_paper_table14'] = _matches(test, paper)

    variant = {
        'paper_name': paper_name,
        'paper_table14': (
            None if paper is None else {'mean': paper[0], 'std': paper[1]}
        ),
        'main': main,
        'evaluations': evaluations,
        'evaluation_available': bool(evaluations),
        'conservative': conservative,
    }
    return variant, unclassified


def _primary_evaluation(
    evaluations: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """The evaluation the paper reports: online greedy first, then any other."""
    if not evaluations:
        return None
    for key in ('eval-online-ensembles/greedy', 'eval-offline-ensembles/greedy'):
        if key in evaluations:
            return key, evaluations[key]
    key = min(evaluations)
    return key, evaluations[key]


# ---------------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------------
def _extract_main(
    root: Path, report_path: Path
) -> tuple[dict[str, Any], dict[int, Any]]:
    """The main-run summary and {member id: sampled config} of finished members."""
    report = _read_json(report_path)
    config_path = report_path.parent / 'config.json'
    config = _read_json(config_path) if config_path.exists() else None

    epoch_size = _int_or_none(report.get('epoch_size'))
    step = _int_or_none(report.get('step'))
    experiments = report.get('experiments') or []
    finished_ids = [int(x['report']['id']) for x in experiments]
    n_models = _int_or_none(report.get('n_models'))
    configured = None if config is None else _int_or_none(config.get('n_models'))

    unfinished_ids = None
    if configured is not None and set(finished_ids) <= set(range(configured)):
        unfinished_ids = sorted(set(range(configured)) - set(finished_ids))

    member_configs = {int(x['report']['id']): x.get('config') for x in experiments}
    summary = {
        'report_path': _rel(root, report_path),
        'config_path': None if config is None else _rel(root, config_path),
        'function': report.get('function'),
        'seed': None if config is None else config.get('seed'),
        'gpu': report.get('gpu'),
        'n_models': n_models,
        'n_trials': _int_or_none(report.get('n_trials')),
        'n_models_configured': configured,
        'n_unfinished': (
            None if configured is None or n_models is None else configured - n_models
        ),
        'unfinished_ids': unfinished_ids,
        'time_sec': _float_or_none(report.get('time')),
        'epoch_size': epoch_size,
        'step': step,
        'n_epochs': _epochs(step, epoch_size),
        'eval_batch_size': _int_or_none(report.get('eval_batch_size')),
        'prediction_type': report.get('prediction_type'),
        'online_ensembles': {
            ens_name: _ensemble_summary(ens.get('report') or {}, epoch_size)
            for ens_name, ens in sorted((report.get('online_ensembles') or {}).items())
        },
        'best_single': _best_single(report, experiments),
        'members': _member_stats(experiments),
        'config': None if config is None else _config_highlights(config),
    }
    return summary, member_configs


def _ensemble_summary(ens: dict[str, Any], epoch_size: int | None) -> dict[str, Any]:
    ids = [int(i) for i in ens.get('ids') or []]
    metrics = ens.get('metrics') or {}
    step = _int_or_none(ens.get('step'))
    history = ens.get('history')
    return {
        'ids': ids,
        'unique_ids': sorted(set(ids)),
        'n_unique': len(set(ids)),
        'size': len(ids),
        'steps': [int(s) for s in ens.get('steps') or []],
        'step': step,
        'epoch': _epochs(step, epoch_size),
        'time_sec': _float_or_none(ens.get('time')),
        'n_updates': None if history is None else len(history),
        'val': _scalar_metrics(metrics.get('val')),
        'test': _scalar_metrics(metrics.get('test')),
    }


def _best_single(
    report: dict[str, Any], experiments: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The official report['best'] (first member with the strictly highest val
    score, in finishing order); recomputed that way when the key is missing."""
    best = report.get('best')
    source = "report['best']"
    if best is None:
        source = 'recomputed: first member with the highest val score'
        best = None
        for experiment in experiments:
            if best is None or _val_score(experiment) > _val_score(best):
                best = experiment
    if best is None:
        return None
    best_report = best['report']
    best_val = _val_score(best)
    metrics = best_report.get('metrics') or {}
    return {
        'id': int(best_report['id']),
        'best_step': _int_or_none(best_report.get('best_step')),
        'config': best.get('config'),
        'source': source,
        'n_val_ties': sum(1 for x in experiments if _val_score(x) == best_val),
        'train_score': _float_or_none((metrics.get('train') or {}).get('score')),
        'val_score': _float_or_none(best_val),
        'test_score': _float_or_none((metrics.get('test') or {}).get('score')),
    }


def _member_stats(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    by_part = {
        part: [
            x['report']['metrics'][part]['score']
            for x in experiments
            if part in x['report'].get('metrics', {})
        ]
        for part in ('val', 'test')
    }
    return {
        'n': len(experiments),
        'ids': sorted(int(x['report']['id']) for x in experiments),
        'val': _stats(by_part['val']),
        'test': _stats(by_part['test']),
    }


def _config_highlights(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get('model') or {}
    optimizer = config.get('optimizer') or {}
    sampler = config.get('sampler') or {}
    ensembles = {}
    for ens_name, ens in sorted((config.get('online_ensembles') or {}).items()):
        options = ens.get('options') or {}
        ensembles[ens_name] = {
            'type': ens.get('type'),
            'update_type': ens.get('update_type'),
            'include_current_ensemble_in_pool': ens.get(
                'include_current_ensemble_in_pool'
            ),
            'patience': ens.get('patience'),
            'max_ensemble_size': options.get('max_ensemble_size'),
        }
    return {
        'n_models': config.get('n_models'),
        'batch_size': config.get('batch_size'),
        'n_epochs': config.get('n_epochs'),
        'patience': config.get('patience'),
        'amp_dtype': config.get('amp_dtype'),
        'd_block': model.get('d_block'),
        'activation': model.get('activation'),
        'num_embeddings': model.get('num_embeddings'),
        'optimizer_type': optimizer.get('type'),
        'optimizer_shared_step': optimizer.get('shared_step'),
        'online_ensembles': ensembles,
        'max_ensemble_size': _first_ensemble_option(ensembles, 'max_ensemble_size'),
        'ensemble_patience': _first_ensemble_option(ensembles, 'patience'),
        'data': config.get('data'),
        'sampler_type': sampler.get('type'),
        'search_space': sampler.get('space'),
        'n_fixed_configs': (
            None if config.get('configs') is None else len(config['configs'])
        ),
    }


def _first_ensemble_option(ensembles: dict[str, Any], key: str) -> Any:
    if 'greedy' in ensembles:
        return ensembles['greedy'][key]
    return next(iter(ensembles.values()))[key] if ensembles else None


# ---------------------------------------------------------------------------------
# Evaluation (conservative) runs
# ---------------------------------------------------------------------------------
def _extract_evaluation(
    root: Path, report_path: Path, *, kind: str, ensemble_name: str
) -> dict[str, Any]:
    report = _read_json(report_path)
    config_path = report_path.parent / 'config.json'
    config = _read_json(config_path) if config_path.exists() else {}
    base_config = config.get('base_config') or {}

    per_seed = []
    for experiment in report.get('experiments') or []:
        run = experiment['report']
        epoch_size = _int_or_none(run.get('epoch_size'))
        ens = ((run.get('online_ensembles') or {}).get(ensemble_name) or {}).get(
            'report'
        ) or {}
        ens_summary = _ensemble_summary(ens, epoch_size)
        members = run.get('experiments') or []
        best = _best_single(run, members)
        per_seed.append(
            {
                'seed': (experiment.get('config') or {}).get('seed'),
                'gpu': run.get('gpu'),
                'n_models': _int_or_none(run.get('n_models')),
                'time_sec': _float_or_none(run.get('time')),
                'step': _int_or_none(run.get('step')),
                'ensemble': {
                    key: ens_summary[key]
                    for key in ('ids', 'n_unique', 'size', 'step', 'epoch')
                }
                | {
                    'val_score': ens_summary['val'].get('score'),
                    'test_score': ens_summary['test'].get('score'),
                },
                'best_single': None
                if best is None
                else {
                    'id': best['id'],
                    'val_score': best['val_score'],
                    'test_score': best['test_score'],
                },
                'members': {
                    'n': len(members),
                    'test_mean': _stats(
                        [x['report']['metrics']['test']['score'] for x in members]
                    )['mean'],
                },
            }
        )

    test_scores = [s['ensemble']['test_score'] for s in per_seed]
    val_scores = [s['ensemble']['val_score'] for s in per_seed]
    times = [s['time_sec'] for s in per_seed]
    return {
        'report_path': _rel(root, report_path),
        'config_path': _rel(root, config_path) if config else None,
        'protocol': 'conservative',
        'ensemble_kind': kind,
        'ensemble_name': ensemble_name,
        'function': report.get('function'),
        'run_function': config.get('function'),
        'n_seeds': config.get('n_seeds'),
        'seeds': [s['seed'] for s in per_seed],
        'time_sec_total': _float_or_none(report.get('time')),
        'time_sec_per_seed': _stats([t for t in times if t is not None]),
        'base_config': _config_highlights(base_config) if base_config else None,
        'selected_configs': base_config.get('configs'),
        'per_seed': per_seed,
        'val': _stats([v for v in val_scores if v is not None]),
        'test': _stats([t for t in test_scores if t is not None]),
        'text': _pm_text(*_mean_std(test_scores)),
    }


def _link_evaluation_to_main(
    evaluation: dict[str, Any],
    main: dict[str, Any] | None,
    member_configs: dict[int, Any],
) -> None:
    """Map the evaluation's fixed configs back to main-run member ids."""
    selected = evaluation.pop('selected_configs')
    evaluation['selected_main_ids'] = None
    evaluation['selected_ids_match_main_ensemble'] = None
    if main is None or selected is None:
        return
    ids = []
    for config in selected:
        matches = [i for i, c in member_configs.items() if c == config]
        ids.append(matches[0] if len(matches) == 1 else None)
    evaluation['selected_main_ids'] = ids
    ens = main['online_ensembles'].get(evaluation['ensemble_name'])
    if ens is not None:
        evaluation['selected_ids_match_main_ensemble'] = ids == ens['unique_ids']


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def _headline(variants: dict[str, Any]) -> dict[str, Any] | None:
    """The TabPack (no embeddings) conservative number, the one this repository
    reproduces; None when the clone has no evaluation report for it."""
    variant = variants.get('tabpack')
    if variant is None or not variant['conservative']['available']:
        return None
    conservative = variant['conservative']
    ensemble_name = conservative['evaluation'].split('/')[-1]
    return {
        'variant': 'tabpack',
        'paper_name': variant['paper_name'],
        'protocol': 'conservative',
        'metric': 'test accuracy, mean ± sample std over seeds',
        'n_seeds': conservative['n_seeds'],
        'test_mean': conservative['test_mean'],
        'test_std': conservative['test_std'],
        'text': conservative['text'],
        'source': (
            f'{conservative["report_path"]}: experiments[*].report.online_ensembles.'
            f'{ensemble_name}.report.metrics.test.score'
        ),
        'paper_table14': variant['paper_table14'],
        'matches_paper_table14': conservative.get('matches_paper_table14'),
    }


def _summary_row(name: str, variant: dict[str, Any]) -> dict[str, Any]:
    main = variant['main'] or {}
    ens = (main.get('online_ensembles') or {}).get('greedy') or {}
    best = main.get('best_single') or {}
    members = main.get('members') or {}
    conservative = variant['conservative']
    return {
        'paper_name': variant['paper_name'],
        'main_gpu': main.get('gpu'),
        'main_time_sec': main.get('time_sec'),
        'main_n_models_configured': main.get('n_models_configured'),
        'main_n_finished': main.get('n_models'),
        'main_ensemble_n_unique': ens.get('n_unique'),
        'main_ensemble_size': ens.get('size'),
        'main_ensemble_val': (ens.get('val') or {}).get('score'),
        'main_ensemble_test': (ens.get('test') or {}).get('score'),
        'main_best_single_val': best.get('val_score'),
        'main_best_single_test': best.get('test_score'),
        'main_member_test_mean': (members.get('test') or {}).get('mean'),
        'main_member_test_std': (members.get('test') or {}).get('std'),
        'conservative_available': conservative['available'],
        'conservative_n_seeds': conservative.get('n_seeds'),
        'conservative_test_mean': conservative.get('test_mean'),
        'conservative_test_std': conservative.get('test_std'),
        'conservative_text': conservative.get('text'),
        'paper_table14': variant['paper_table14'],
    }


def _matches(stats: dict[str, Any], paper: tuple[float, float]) -> bool:
    mean, std = stats['mean'], stats['std']
    if mean is None or std is None:
        return False
    return round(mean, 4) == paper[0] and round(std, 4) == paper[1]


def _stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(v)]
    mean, std = _mean_std(clean)
    return {
        'n': len(clean),
        'mean': mean,
        'std': std,
        'min': min(clean) if clean else None,
        'max': max(clean) if clean else None,
    }


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(v)]
    mean = statistics.mean(clean) if clean else None
    std = statistics.stdev(clean) if len(clean) >= 2 else None
    return mean, std


def _pm_text(mean: float | None, std: float | None) -> str | None:
    """'85.75 ± 0.14' (percent, two decimals)."""
    if mean is None:
        return None
    if std is None:
        return f'{100 * mean:.2f}'
    return f'{100 * mean:.2f} ± {100 * std:.2f}'


def _scalar_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Top-level numeric metrics (score, accuracy, roc-auc, cross-entropy, ...);
    per-class dicts of the classification report are dropped."""
    if not metrics:
        return {}
    return {
        key: _float_or_none(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _val_score(experiment: dict[str, Any]) -> float:
    score = experiment['report']['metrics']['val']['score']
    return -math.inf if score is None or math.isnan(score) else float(score)


def _epochs(step: int | None, epoch_size: int | None) -> int | None:
    if step is None or not epoch_size:
        return None
    return step // epoch_size


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sort_keys(obj: Any) -> Any:
    """Recursively rebuild dicts with sorted keys (lists keep their order), so any
    JSON writer emits the same key order as ``sort_keys=True``."""
    if isinstance(obj, dict):
        return {key: _sort_keys(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list | tuple):
        return [_sort_keys(x) for x in obj]
    return obj


def _project_root() -> Path:
    # src/tabpack_repro/reporting/reference.py -> repository root
    return Path(__file__).resolve().parents[3]


def _git_head(path: Path) -> str | None:
    """HEAD commit of the git checkout at ``path`` (None if it is not one)."""
    if not (path / '.git').exists():
        return None
    try:
        out = subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if _HEX40_RE.match(out) else None


def _dumps_reference(data: dict[str, Any]) -> str:
    """Pretty JSON with sorted keys and a trailing newline (the committed format)."""
    return (
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + '\n'
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reference-dir', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    reference_dir = args.reference_dir or get_reference_dir()
    if reference_dir is None:
        parser.error('official clone not found; set TABPACK_REFERENCE_DIR')
    data = extract_churn_reference(reference_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_dumps_reference(data), encoding='utf-8')
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
