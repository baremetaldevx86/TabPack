"""Tests for tabpack_repro.reporting.reference (a38).

Most tests run on a fabricated mini clone of the official repository in tmp_path.
The parity test re-extracts from the real clone ($TABPACK_REFERENCE_DIR) and checks
that the result equals the committed results/reference/churn_official.json.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from tabpack_repro.reporting import reference
from tabpack_repro.reporting.reference import (
    extract_churn_reference,
    get_reference_dir,
    load_saved_reference,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
COMMITTED = PROJECT_DIR / 'results' / 'reference' / 'churn_official.json'

# The official TabPack Churn evaluation: 5 seeds, online greedy ensemble test scores.
EVAL_TEST = [0.8585, 0.858, 0.858, 0.855, 0.858]
EVAL_VAL = [0.871875, 0.871875, 0.8725, 0.871875, 0.8725]


# ---------------------------------------------------------------------------------
# Mini clone
# ---------------------------------------------------------------------------------
def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def _member(mid: int, val: float, test: float, config: dict | None = None) -> dict:
    return {
        'config': config or {'model': {'n_blocks': 1 + mid % 4, 'dropout': 0.0}},
        'report': {
            'id': mid,
            'best_step': 25 * (mid + 1),
            'metrics': {
                'train': {'accuracy': 0.9, 'score': 0.9},
                'val': {'accuracy': val, 'score': val},
                'test': {'accuracy': test, 'score': test},
            },
            'time': 1.0 + mid,
        },
    }


def _ens_metrics(val: float, test: float) -> dict:
    per_class = {'precision': 0.9, 'recall': 0.9, 'f1-score': 0.9, 'support': 10.0}
    return {
        part: {
            '0': per_class,
            'accuracy': score,
            'roc-auc': 0.86,
            'cross-entropy': 0.34,
            'score': score,
        }
        for part, score in (('val', val), ('test', test))
    }


def _main_config(n_models: int, **model: Any) -> dict:
    return {
        'seed': 0,
        'n_models': n_models,
        'batch_size': 256,
        'n_epochs': -1,
        'patience': 16,
        'amp_dtype': 'bfloat16',
        'data': {'path': 'data/churn', 'cat_policy': 'ordinal'},
        'model': {'activation': 'ReLU', 'd_block': 384, **model},
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': True},
        'online_ensembles': {
            'greedy': {
                'type': 'greedy',
                'update_type': 'latest',
                'include_current_ensemble_in_pool': True,
                'patience': 32,
                'options': {'max_ensemble_size': 32},
            }
        },
        'sampler': {
            'type': 'RandomSampler',
            'space': {'model': {'n_blocks': ['_tune_', 'int', 1, 4]}},
        },
    }


# Finished members of the fabricated main run (id -> (val, test)); ids 3 and 5 of the
# 7 configured members are still running when the run stops.
MEMBERS = {0: (0.85, 0.84), 1: (0.87, 0.855), 2: (0.87, 0.86), 4: (0.86, 0.85),
           6: (0.84, 0.83)}  # fmt: skip
MEMBER_CONFIGS = {
    mid: {'model': {'n_blocks': 1 + mid % 4, 'dropout': 0.1 * mid}} for mid in range(7)
}


def _main_report(ensemble_ids: list[int], **extra: Any) -> dict:
    experiments = [
        _member(mid, val, test, MEMBER_CONFIGS[mid])
        for mid, (val, test) in MEMBERS.items()
    ]
    report = {
        'function': 'project.tabpack.main',
        'gpu': 'FAKE-GPU',
        'n_models': len(experiments),
        'n_trials': len(experiments),
        'prediction_type': 'probs',
        'epoch_size': 25,
        'eval_batch_size': 32768,
        'step': 510,
        'time': 12.5,
        'online_ensembles': {
            'greedy': {
                'report': {
                    'ids': ensemble_ids,
                    'steps': [100] * len(ensemble_ids),
                    'step': 260,
                    'time': 3.0,
                    'ensemble_time': 0.1,
                    'metrics': _ens_metrics(0.88, 0.857),
                    'history': [{}, {}, {}],
                }
            }
        },
        'experiments': experiments,
        'best': experiments[1],
    }
    report.update(extra)
    return report


def _eval_seed_report(seed: int, val: float, test: float, n_models: int) -> dict:
    members = [_member(i, 0.86 + 0.001 * i, 0.85 + 0.002 * i) for i in range(n_models)]
    return {
        'config': {'seed': seed, 'n_models': n_models},
        'report': {
            'function': 'project.tabpack.main',
            'gpu': 'FAKE-GPU',
            'n_models': n_models,
            'epoch_size': 25,
            'step': 300 + seed,
            'time': 4.0 + seed,
            'online_ensembles': {
                'greedy': {
                    'report': {
                        'ids': [0, 1, 0],
                        'steps': [50, 75, 50],
                        'step': 100,
                        'metrics': _ens_metrics(val, test),
                    }
                }
            },
            'experiments': members,
            'best': members[-1],
        },
    }


@pytest.fixture
def mini_clone(tmp_path: Path) -> Path:
    root = tmp_path / 'clone'
    churn = root / 'experiments' / 'tabpack' / 'churn'
    ensemble_ids = [2, 1, 2]  # unique: [1, 2]
    _write(churn / 'main' / 'config.json', _main_config(7))
    _write(churn / 'main' / 'report.json', _main_report(ensemble_ids))

    evaluation = churn / 'eval-online-ensembles' / 'greedy' / 'evaluation'
    base = _main_config(2)
    base.pop('sampler')
    base['configs'] = [MEMBER_CONFIGS[1], MEMBER_CONFIGS[2]]
    _write(
        evaluation / 'config.json',
        {'function': 'project.tabpack.main', 'n_seeds': 5, 'base_config': base},
    )
    _write(
        evaluation / 'report.json',
        {
            'function': 'lib.tools.evaluate.main',
            'time': 30.0,
            'experiments': [
                _eval_seed_report(seed, val, test, 2 if seed != 3 else 1)
                for seed, (val, test) in enumerate(
                    zip(EVAL_VAL, EVAL_TEST, strict=True)
                )
            ],
        },
    )

    # A variant with a main run only (no evaluation) and cosine embeddings.
    cosine = root / 'experiments' / 'tabpack-cosine' / 'churn' / 'main'
    _write(
        cosine / 'config.json',
        _main_config(7, num_embeddings={'type': 'CosineEmbeddingsPack'}),
    )
    report = _main_report([0], gpu=None)
    report.pop('gpu')
    report.pop('best')
    _write(cosine / 'report.json', report)

    # Noise: another dataset, a stray churn report, a directory without churn.
    _write(root / 'experiments' / 'tabpack' / 'adult' / 'main' / 'report.json', {})
    _write(churn / 'scratch' / 'report.json', {})
    (root / 'experiments' / 'examples' / 'demo').mkdir(parents=True)
    (root / 'experiments' / 'tabpack' / 'make.py').write_text('')
    return root


# ---------------------------------------------------------------------------------
# extract_churn_reference on the mini clone
# ---------------------------------------------------------------------------------
def test_extract_top_level(mini_clone: Path) -> None:
    ref = extract_churn_reference(mini_clone)
    assert ref['schema_version'] == 1
    assert ref['dataset'] == 'churn'
    assert ref['metric'] == 'accuracy'
    assert sorted(ref['variants']) == ['tabpack', 'tabpack-cosine']
    assert ref['other_reports'] == ['experiments/tabpack/churn/scratch/report.json']
    provenance = ref['provenance']
    assert provenance['commit'] is None  # not a git checkout
    assert provenance['pinned_commit'] == '05a89e21b955f12de84889d662e15ca534019aaa'
    assert provenance['commit_matches_pinned'] is False
    assert 'statistics.stdev' in provenance['conservative_recipe']
    assert ref['paper_table14']['churn']['TabPack'] == {'mean': 0.8575, 'std': 0.0014}
    assert set(ref['summary']) == set(ref['variants'])


def test_extract_main_run(mini_clone: Path) -> None:
    main = extract_churn_reference(mini_clone)['variants']['tabpack']['main']
    assert main['report_path'] == 'experiments/tabpack/churn/main/report.json'
    assert main['config_path'] == 'experiments/tabpack/churn/main/config.json'
    assert main['function'] == 'project.tabpack.main'
    assert main['seed'] == 0
    assert main['gpu'] == 'FAKE-GPU'
    assert main['n_models'] == 5
    assert main['n_trials'] == 5
    assert main['n_models_configured'] == 7
    assert main['n_unfinished'] == 2
    assert main['unfinished_ids'] == [3, 5]
    assert main['time_sec'] == 12.5
    assert main['epoch_size'] == 25
    assert main['step'] == 510
    assert main['n_epochs'] == 20
    assert main['prediction_type'] == 'probs'

    ens = main['online_ensembles']['greedy']
    assert ens['ids'] == [2, 1, 2]
    assert ens['unique_ids'] == [1, 2]
    assert ens['n_unique'] == 2
    assert ens['size'] == 3
    assert ens['step'] == 260
    assert ens['epoch'] == 10
    assert ens['n_updates'] == 3
    assert ens['time_sec'] == 3.0
    # Per-class dicts are dropped, scalar metrics kept.
    assert ens['test'] == {
        'accuracy': 0.857,
        'cross-entropy': 0.34,
        'roc-auc': 0.86,
        'score': 0.857,
    }
    assert ens['val']['score'] == 0.88

    best = main['best_single']
    assert best['id'] == 1
    assert best['source'] == "report['best']"
    assert best['val_score'] == 0.87
    assert best['test_score'] == 0.855
    assert best['train_score'] == 0.9
    assert best['best_step'] == 50
    assert best['n_val_ties'] == 2
    assert best['config'] == MEMBER_CONFIGS[1]

    members = main['members']
    tests = [test for _, test in MEMBERS.values()]
    assert members['n'] == 5
    assert members['ids'] == sorted(MEMBERS)
    assert members['test']['mean'] == statistics.mean(tests)
    assert members['test']['std'] == statistics.stdev(tests)
    assert members['test']['min'] == min(tests)
    assert members['test']['max'] == max(tests)
    assert members['val']['n'] == 5

    config = main['config']
    assert config['n_models'] == 7
    assert config['d_block'] == 384
    assert config['activation'] == 'ReLU'
    assert config['optimizer_type'] == 'MuonAdamWPack'
    assert config['optimizer_shared_step'] is True
    assert config['patience'] == 16
    assert config['max_ensemble_size'] == 32
    assert config['ensemble_patience'] == 32
    assert config['num_embeddings'] is None
    assert config['online_ensembles']['greedy']['update_type'] == 'latest'
    assert config['sampler_type'] == 'RandomSampler'
    assert config['search_space'] == {'model': {'n_blocks': ['_tune_', 'int', 1, 4]}}
    assert config['n_fixed_configs'] is None


def test_extract_evaluation(mini_clone: Path) -> None:
    variant = extract_churn_reference(mini_clone)['variants']['tabpack']
    assert variant['paper_name'] == 'TabPack'
    assert variant['evaluation_available'] is True
    assert list(variant['evaluations']) == ['eval-online-ensembles/greedy']
    ev = variant['evaluations']['eval-online-ensembles/greedy']
    assert ev['report_path'] == (
        'experiments/tabpack/churn/eval-online-ensembles/greedy/evaluation/report.json'
    )
    assert ev['protocol'] == 'conservative'
    assert ev['ensemble_kind'] == 'online'
    assert ev['ensemble_name'] == 'greedy'
    assert ev['function'] == 'lib.tools.evaluate.main'
    assert ev['run_function'] == 'project.tabpack.main'
    assert ev['n_seeds'] == 5
    assert ev['seeds'] == [0, 1, 2, 3, 4]
    assert ev['time_sec_total'] == 30.0
    assert ev['time_sec_per_seed']['mean'] == statistics.mean([4.0, 5, 6, 7, 8])
    assert ev['selected_main_ids'] == [1, 2]
    assert ev['selected_ids_match_main_ensemble'] is True
    assert ev['base_config']['n_models'] == 2
    assert ev['base_config']['n_fixed_configs'] == 2
    assert 'selected_configs' not in ev

    assert ev['test']['n'] == 5
    assert ev['test']['mean'] == statistics.mean(EVAL_TEST)
    assert ev['test']['std'] == statistics.stdev(EVAL_TEST)
    assert ev['val']['mean'] == statistics.mean(EVAL_VAL)
    assert ev['text'] == '85.75 ± 0.14'

    seed3 = ev['per_seed'][3]
    assert seed3['seed'] == 3
    assert seed3['n_models'] == 1
    assert seed3['time_sec'] == 7.0
    assert seed3['ensemble'] == {
        'ids': [0, 1, 0],
        'n_unique': 2,
        'size': 3,
        'step': 100,
        'epoch': 4,
        'val_score': EVAL_VAL[3],
        'test_score': EVAL_TEST[3],
    }
    assert seed3['best_single'] == {'id': 0, 'val_score': 0.86, 'test_score': 0.85}
    assert seed3['members'] == {'n': 1, 'test_mean': 0.85}

    conservative = variant['conservative']
    assert conservative['available'] is True
    assert conservative['evaluation'] == 'eval-online-ensembles/greedy'
    assert conservative['n_seeds'] == 5
    assert conservative['text'] == '85.75 ± 0.14'
    assert conservative['matches_paper_table14'] is True


def test_headline_and_crosscheck(mini_clone: Path) -> None:
    ref = extract_churn_reference(mini_clone)
    headline = ref['headline']
    assert headline['variant'] == 'tabpack'
    assert headline['text'] == '85.75 ± 0.14'
    assert headline['n_seeds'] == 5
    assert headline['matches_paper_table14'] is True
    assert headline['source'].startswith(
        'experiments/tabpack/churn/eval-online-ensembles/greedy/evaluation/report.json'
    )
    assert headline['source'].endswith(
        'online_ensembles.greedy.report.metrics.test.score'
    )
    assert ref['paper_crosscheck']['tabpack']['matches_paper_table14'] is True
    assert ref['paper_crosscheck']['tabpack-cosine'] == {
        'paper_name': 'TabPack†',
        'n_seeds': None,
        'matches_paper_table14': None,
    }
    row = ref['summary']['tabpack']
    assert row['main_ensemble_test'] == 0.857
    assert row['main_best_single_test'] == 0.855
    assert row['main_n_finished'] == 5
    assert row['conservative_test_mean'] == statistics.mean(EVAL_TEST)


def test_paper_mismatch_is_reported(mini_clone: Path) -> None:
    path = (
        mini_clone / 'experiments/tabpack/churn/eval-online-ensembles/greedy'
        '/evaluation/report.json'
    )
    report = json.loads(path.read_text())
    ens = report['experiments'][0]['report']['online_ensembles']['greedy']['report']
    ens['metrics']['test']['score'] = 0.8
    path.write_text(json.dumps(report))
    ref = extract_churn_reference(mini_clone)
    assert ref['variants']['tabpack']['conservative']['matches_paper_table14'] is False
    assert ref['headline']['matches_paper_table14'] is False


def test_variant_without_evaluation(mini_clone: Path) -> None:
    variant = extract_churn_reference(mini_clone)['variants']['tabpack-cosine']
    assert variant['paper_name'] == 'TabPack†'
    assert variant['paper_table14'] == {'mean': 0.8623, 'std': 0.0028}
    assert variant['evaluation_available'] is False
    assert variant['evaluations'] == {}
    assert variant['conservative']['available'] is False
    assert 'eval-*-ensembles' in variant['conservative']['note']
    main = variant['main']
    assert main['gpu'] is None
    assert main['config']['num_embeddings'] == {'type': 'CosineEmbeddingsPack'}
    # No report['best']: first member with the highest val score (finishing order).
    assert main['best_single']['id'] == 1
    assert main['best_single']['source'].startswith('recomputed')
    assert main['best_single']['n_val_ties'] == 2


def test_no_evaluation_anywhere_means_no_headline(mini_clone: Path) -> None:
    import shutil

    shutil.rmtree(mini_clone / 'experiments/tabpack/churn/eval-online-ensembles')
    ref = extract_churn_reference(mini_clone)
    assert ref['headline'] is None
    assert ref['variants']['tabpack']['conservative']['available'] is False
    assert ref['summary']['tabpack']['conservative_text'] is None


def test_nan_metrics_become_null(mini_clone: Path) -> None:
    path = mini_clone / 'experiments/tabpack/churn/main/report.json'
    report = json.loads(path.read_text())
    metrics = report['online_ensembles']['greedy']['report']['metrics']
    metrics['test']['cross-entropy'] = math.nan
    report['experiments'][0]['report']['metrics']['val']['score'] = math.nan
    path.write_text(json.dumps(report))  # json writes NaN, as Python's json reads it
    ref = extract_churn_reference(mini_clone)
    main = ref['variants']['tabpack']['main']
    assert main['online_ensembles']['greedy']['test']['cross-entropy'] is None
    assert main['members']['val']['n'] == 4
    json.dumps(ref, allow_nan=False)  # strict JSON


def test_keys_sorted_and_deterministic(mini_clone: Path) -> None:
    ref = extract_churn_reference(mini_clone)

    def check(obj: Any) -> None:
        if isinstance(obj, dict):
            assert list(obj) == sorted(obj)
            for value in obj.values():
                check(value)
        elif isinstance(obj, list):
            for value in obj:
                check(value)

    check(ref)
    assert extract_churn_reference(str(mini_clone)) == ref


def test_missing_experiments_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='experiments'):
        extract_churn_reference(tmp_path)


def test_empty_clone(tmp_path: Path) -> None:
    (tmp_path / 'experiments').mkdir()
    ref = extract_churn_reference(tmp_path)
    assert ref['variants'] == {}
    assert ref['headline'] is None
    assert ref['other_reports'] == []


def test_main_writes_committed_format(mini_clone: Path, tmp_path: Path) -> None:
    out = tmp_path / 'out' / 'ref.json'
    assert (
        reference._main(['--reference-dir', str(mini_clone), '--output', str(out)]) == 0
    )
    text = out.read_text(encoding='utf-8')
    assert text.endswith('}\n')
    assert '85.75 ± 0.14' in text  # ensure_ascii=False
    assert json.loads(text) == extract_churn_reference(mini_clone)
    assert text == reference._dumps_reference(extract_churn_reference(mini_clone))


# ---------------------------------------------------------------------------------
# get_reference_dir
# ---------------------------------------------------------------------------------
def test_get_reference_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clone = tmp_path / 'somewhere'
    clone.mkdir()
    monkeypatch.setattr(reference, '_project_root', lambda: tmp_path / 'repo')
    monkeypatch.setenv('TABPACK_REFERENCE_DIR', str(clone))
    assert get_reference_dir() == clone


def test_get_reference_dir_env_missing_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default = tmp_path / 'repo' / '.reference' / 'tabpack'
    default.mkdir(parents=True)
    monkeypatch.setattr(reference, '_project_root', lambda: tmp_path / 'repo')
    monkeypatch.setenv('TABPACK_REFERENCE_DIR', str(tmp_path / 'does-not-exist'))
    assert get_reference_dir() == default
    monkeypatch.setenv('TABPACK_REFERENCE_DIR', '')
    assert get_reference_dir() == default
    monkeypatch.delenv('TABPACK_REFERENCE_DIR')
    assert get_reference_dir() == default


def test_get_reference_dir_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(reference, '_project_root', lambda: tmp_path / 'repo')
    monkeypatch.delenv('TABPACK_REFERENCE_DIR', raising=False)
    assert get_reference_dir() is None
    monkeypatch.setenv('TABPACK_REFERENCE_DIR', str(tmp_path / 'nope'))
    assert get_reference_dir() is None
    # A file is not a clone directory.
    (tmp_path / 'file').write_text('')
    monkeypatch.setenv('TABPACK_REFERENCE_DIR', str(tmp_path / 'file'))
    assert get_reference_dir() is None


def test_get_reference_dir_default_is_repo_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert reference._project_root() == PROJECT_DIR
    monkeypatch.delenv('TABPACK_REFERENCE_DIR', raising=False)
    expected = PROJECT_DIR / '.reference' / 'tabpack'
    assert get_reference_dir() == (expected if expected.is_dir() else None)


# ---------------------------------------------------------------------------------
# load_saved_reference and the committed file
# ---------------------------------------------------------------------------------
def test_committed_reference_loads() -> None:
    ref = load_saved_reference(COMMITTED)
    assert 'tabpack' in ref['variants']
    test = ref['variants']['tabpack']['main']['online_ensembles']['greedy']['test']
    assert 0.8 < test['score'] < 0.9
    assert 0.8 < ref['headline']['test_mean'] < 0.9
    assert ref['headline']['text'] == '85.75 ± 0.14'
    assert ref['provenance']['commit'] == reference.REFERENCE_COMMIT
    assert all(v['matches_paper_table14'] for v in ref['paper_crosscheck'].values())
    assert ref['other_reports'] == []


def test_committed_reference_is_formatted() -> None:
    text = COMMITTED.read_text(encoding='utf-8')
    assert text == reference._dumps_reference(json.loads(text))


def test_load_saved_reference_default_path_from_any_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_saved_reference() == json.loads(COMMITTED.read_text(encoding='utf-8'))


def test_load_saved_reference_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_saved_reference(tmp_path / 'missing.json')
    bad = tmp_path / 'bad.json'
    bad.write_text('[1, 2]')
    with pytest.raises(ValueError, match='variants'):
        load_saved_reference(bad)


@pytest.mark.parity
def test_parity_committed_equals_fresh_extraction() -> None:
    clone = get_reference_dir()
    if clone is None or not (clone / 'experiments').is_dir():
        pytest.skip('Official TabPack clone not found (set TABPACK_REFERENCE_DIR)')
    fresh = extract_churn_reference(clone)
    assert fresh == load_saved_reference(COMMITTED)
    assert COMMITTED.read_text(encoding='utf-8') == reference._dumps_reference(fresh)
