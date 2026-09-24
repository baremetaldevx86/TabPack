"""Tests for tabpack_repro.methods.conservative (a33).

Unit tests replace methods.tabpack.run with a fake that writes a run like the real
one (report.json first, then predictions.npz and config.toml).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tabpack_repro.config import (
    ConservativeEvalConfig,
    DataConfig,
    MLPMethodConfig,
    MuonAdamWConfig,
    OnlineEnsembleConfig,
    TabPackConfig,
    TrainingConfig,
    config_to_dict,
    dump_config,
    load_config,
)
from tabpack_repro.methods import conservative, tabpack
from tabpack_repro.utils.io import dump_json, load_json, to_jsonable

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _member_config(i: int) -> dict[str, Any]:
    return {
        'model': {'n_blocks': i % 4 + 1, 'dropout': 0.05 * i},
        'optimizer': {
            'lr': 1e-4 * (i + 1),
            'weight_decay': 1e-3 * (i + 1),
            'muon_lr': 1e-3 * (i + 2),
        },
    }


def _source_config() -> TabPackConfig:
    # Non-default values everywhere, so that "copied from the source" is checked.
    return TabPackConfig(
        seed=7,
        n_models=6,
        d_block=32,
        activation='ReLU',
        optimizer=MuonAdamWConfig(shared_step=False, muon_ns_steps=4, beta2=0.99),
        training=TrainingConfig(
            batch_size=64, patience=3, max_epochs=5, amp_dtype=None, device='cpu'
        ),
        online_ensemble=OnlineEnsembleConfig(patience=5, max_ensemble_size=4),
    )


def _write_source(
    run_dir: Path,
    *,
    ids: list[int] | None = None,
    member_ids: list[int] | None = None,
    member_configs: list[Any] | None = None,
    config: TabPackConfig | None = None,
) -> dict[str, Any]:
    """Write a fake finished TabPack main run into run_dir."""
    config = _source_config() if config is None else config
    ids = [3, 1, 3, 5] if ids is None else ids
    # Members are reported in stopping order, not in id order.
    member_ids = [4, 1, 0, 5, 2, 3] if member_ids is None else member_ids
    members = [
        {
            'id': i,
            'best_step': 10 * i,
            'config': _member_config(i),
            'metrics': {p: {'score': 0.5 + 0.01 * i} for p in ('train', 'val', 'test')},
        }
        for i in member_ids
    ]
    report = {
        'schema_version': 1,
        'method': 'tabpack',
        'dataset': 'churn',
        'seed': config.seed,
        'config': config_to_dict(config),
        'metrics': {'val': {'score': 0.9}, 'test': {'score': 0.8}},
        'members': members,
        'ensemble': {
            'ids': ids,
            'steps': [0] * len(ids),
            'size': len(ids),
            'n_unique': len(set(ids)),
        },
        'best_member': None,
        'n_epochs': 3,
        'n_steps': 30,
        'time_sec': 1.0,
        'env': {'device': 'cpu', 'gpu': None, 'torch': 'x', 'git_commit': None},
        'history': [],
    }
    if member_configs is not None:
        report['member_configs'] = member_configs
    dump_json(run_dir / 'report.json', report)
    return report


def _scores(seed: int) -> dict[str, float]:
    return {'val': 0.80 + 0.01 * seed + 0.003 * seed**2, 'test': 0.85 - 0.002 * seed}


class FakeTabPack:
    """Records calls; writes report.json (method "tabpack"), predictions, config."""

    def __init__(self) -> None:
        self.calls: list[tuple[TabPackConfig, Path]] = []

    def __call__(self, config: TabPackConfig, output_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        self.calls.append((copy.deepcopy(config), output_dir))
        scores = _scores(config.seed)
        report = {
            'schema_version': 1,
            'method': 'tabpack',
            'dataset': 'churn',
            'seed': config.seed,
            'config': config_to_dict(config),
            'metrics': {p: {'score': s, 'roc-auc': 0.5} for p, s in scores.items()},
            'members': [
                {'id': i, 'best_step': 1, 'config': c, 'metrics': {}}
                for i, c in enumerate(config.configs)
            ],
            'member_configs': config.configs,
            'ensemble': {'ids': [0], 'steps': [1], 'size': 1, 'n_unique': 1},
            'best_member': None,
            'n_epochs': 1,
            'n_steps': 1,
            'time_sec': 0.5,
            'env': {'device': 'cpu', 'gpu': None, 'torch': 'x', 'git_commit': None},
            'history': [],
        }
        dump_json(output_dir / 'report.json', report)
        np.savez(output_dir / 'predictions.npz', val=np.zeros(2), test=np.zeros(2))
        dump_config(config, output_dir / 'config.toml')
        return report

    @property
    def seeds(self) -> list[int]:
        return [c.seed for c, _ in self.calls]


@pytest.fixture
def fake_tabpack(monkeypatch) -> FakeTabPack:
    fake = FakeTabPack()
    monkeypatch.setattr(tabpack, 'run', fake)
    return fake


@pytest.fixture
def source_run(tmp_path) -> Path:
    run_dir = tmp_path / 'tabpack' / 'seed-7'
    _write_source(run_dir)
    return run_dir


def _run(source_run: Path, out: Path, n_seeds: int = 3) -> dict[str, Any]:
    config = ConservativeEvalConfig(source_run=str(source_run), n_seeds=n_seeds)
    return conservative.run(config, out)


# ---------------------------------------------------------------------------
# selection and per-seed configs
# ---------------------------------------------------------------------------


def test_retrains_sorted_unique_ensemble_configs_per_seed(
    fake_tabpack, source_run, tmp_path
):
    out = tmp_path / 'conservative'
    _run(source_run, out, n_seeds=3)

    assert fake_tabpack.seeds == [0, 1, 2]
    assert [d for _, d in fake_tabpack.calls] == [out / f'seed-{s}' for s in range(3)]
    source = _source_config()
    for seed_config, _ in fake_tabpack.calls:
        assert isinstance(seed_config, TabPackConfig)
        # ids [3, 1, 3, 5] -> [1, 3, 5], looked up by id (members are not in id order).
        assert seed_config.n_models == 3
        assert seed_config.configs == [_member_config(i) for i in (1, 3, 5)]
        # Official overrides: shared step on; everything else copied from the source.
        assert seed_config.optimizer == dataclasses.replace(
            source.optimizer, shared_step=True
        )
        expected = dataclasses.replace(
            source,
            seed=seed_config.seed,
            n_models=3,
            configs=seed_config.configs,
            optimizer=seed_config.optimizer,
        )
        assert seed_config == expected


def test_seed_configs_are_independent_copies(fake_tabpack, source_run, tmp_path):
    _run(source_run, tmp_path / 'out', n_seeds=2)
    (c0, _), (c1, _) = fake_tabpack.calls
    assert c0.configs == c1.configs
    assert c0.configs is not c1.configs


def test_per_seed_reports_are_marked_conservative(fake_tabpack, source_run, tmp_path):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    for seed in range(2):
        seed_report = load_json(out / f'seed-{seed}' / 'report.json')
        assert seed_report['method'] == 'tabpack-conservative-seed'
        assert seed_report['seed'] == seed
        assert seed_report['metrics']['test']['score'] == _scores(seed)['test']
        assert (out / f'seed-{seed}' / 'predictions.npz').is_file()


def test_source_run_is_not_modified(fake_tabpack, source_run, tmp_path):
    before = (source_run / 'report.json').read_bytes()
    _run(source_run, tmp_path / 'out', n_seeds=1)
    assert (source_run / 'report.json').read_bytes() == before
    assert sorted(p.name for p in source_run.iterdir()) == ['report.json']


# ---------------------------------------------------------------------------
# aggregate report
# ---------------------------------------------------------------------------


def test_aggregate_report(fake_tabpack, source_run, tmp_path):
    out = tmp_path / 'out'
    report = _run(source_run, out, n_seeds=4)

    assert load_json(out / 'report.json') == to_jsonable(report)
    assert report['method'] == 'tabpack-conservative'
    assert report['schema_version'] == 1
    assert report['dataset'] == 'churn'
    assert report['source_run'] == str(source_run)
    assert report['source_seed'] == 7
    assert report['selected_ids'] == [1, 3, 5]
    assert report['n_seeds'] == 4
    assert report['seeds'] == [0, 1, 2, 3]
    assert report['config'] == config_to_dict(
        ConservativeEvalConfig(source_run=str(source_run), n_seeds=4)
    )
    for part in ('val', 'test'):
        values = [_scores(s)[part] for s in range(4)]
        assert report['scores'][part] == values
        assert report['mean'][part] == pytest.approx(statistics.fmean(values))
        assert report['std'][part] == pytest.approx(statistics.stdev(values))
        # Sample std (ddof=1), not the population std.
        assert report['std'][part] == pytest.approx(np.std(values, ddof=1))
    assert (out / 'config.toml').is_file()


def test_single_seed_has_zero_std(fake_tabpack, source_run, tmp_path):
    report = _run(source_run, tmp_path / 'out', n_seeds=1)
    assert report['scores'] == {k: [v] for k, v in _scores(0).items()}
    assert report['mean'] == _scores(0)
    assert report['std'] == {'val': 0.0, 'test': 0.0}


def test_aggregate_config_toml_round_trips(fake_tabpack, source_run, tmp_path):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    assert load_config(out / 'config.toml') == ConservativeEvalConfig(
        source_run=str(source_run), n_seeds=2
    )


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------


def test_resume_skips_finished_seeds_and_aggregates_them(
    fake_tabpack, source_run, tmp_path
):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    assert fake_tabpack.seeds == [0, 1]
    first = load_json(out / 'seed-0' / 'report.json')

    report = _run(source_run, out, n_seeds=4)
    assert fake_tabpack.seeds == [0, 1, 2, 3]
    assert report['seeds'] == [0, 1, 2, 3]
    assert report['scores']['test'] == [_scores(s)['test'] for s in range(4)]
    assert load_json(out / 'seed-0' / 'report.json') == first

    # Everything finished: nothing is retrained, the aggregate is rebuilt.
    again = _run(source_run, out, n_seeds=4)
    assert fake_tabpack.seeds == [0, 1, 2, 3]
    assert again == report


def test_fewer_seeds_aggregates_only_the_requested_ones(
    fake_tabpack, source_run, tmp_path
):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=3)
    report = _run(source_run, out, n_seeds=2)
    assert fake_tabpack.seeds == [0, 1, 2]
    assert report['seeds'] == [0, 1]
    assert len(report['scores']['val']) == 2


def test_finished_but_unmarked_seed_is_marked_not_retrained(
    fake_tabpack, source_run, tmp_path
):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    # Simulate an interruption right after tabpack.run returned for seed 1.
    path = out / 'seed-1' / 'report.json'
    seed_report = load_json(path)
    seed_report['method'] = 'tabpack'
    dump_json(path, seed_report)

    _run(source_run, out, n_seeds=2)
    assert fake_tabpack.seeds == [0, 1]
    assert load_json(path)['method'] == 'tabpack-conservative-seed'


@pytest.mark.parametrize('missing', ['predictions.npz', 'config.toml'])
def test_incomplete_seed_is_retrained(fake_tabpack, source_run, tmp_path, missing):
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    # report.json is the first file TabPack writes: without the other artifacts
    # (and without the conservative mark) the seed did not finish.
    path = out / 'seed-1' / 'report.json'
    seed_report = load_json(path)
    seed_report['method'] = 'tabpack'
    seed_report['metrics']['test']['score'] = 0.0
    dump_json(path, seed_report)
    (out / 'seed-1' / missing).unlink()

    report = _run(source_run, out, n_seeds=2)
    assert fake_tabpack.seeds == [0, 1, 1]
    assert report['scores']['test'][1] == _scores(1)['test']
    assert (out / 'seed-1' / missing).is_file()


def test_marked_seed_without_predictions_counts_as_finished(
    fake_tabpack, source_run, tmp_path
):
    # predictions.npz is git-ignored: committed runs have only report.json and
    # config.toml, and must not be retrained.
    out = tmp_path / 'out'
    _run(source_run, out, n_seeds=2)
    (out / 'seed-0' / 'predictions.npz').unlink()
    _run(source_run, out, n_seeds=2)
    assert fake_tabpack.seeds == [0, 1]


def test_resume_with_another_source_run_raises(fake_tabpack, tmp_path):
    out = tmp_path / 'out'
    first = tmp_path / 'src-a'
    second = tmp_path / 'src-b'
    _write_source(first, ids=[1, 3])
    _write_source(second, ids=[2, 4])
    _run(first, out, n_seeds=1)
    with pytest.raises(ValueError, match='different config'):
        _run(second, out, n_seeds=1)
    assert fake_tabpack.seeds == [0]


def test_refuses_to_overwrite_foreign_seed_report(fake_tabpack, source_run, tmp_path):
    out = tmp_path / 'out'
    dump_json(out / 'seed-0' / 'report.json', {'method': 'mlp'})
    with pytest.raises(ValueError, match='refusing to overwrite'):
        _run(source_run, out, n_seeds=1)
    assert fake_tabpack.calls == []


def test_refuses_to_overwrite_foreign_aggregate(fake_tabpack, source_run, tmp_path):
    out = tmp_path / 'out'
    dump_json(out / 'report.json', {'method': 'tabpack'})
    with pytest.raises(ValueError, match='refusing to overwrite'):
        _run(source_run, out, n_seeds=1)
    assert fake_tabpack.calls == []


def test_output_dir_must_differ_from_source(fake_tabpack, source_run):
    with pytest.raises(ValueError, match='differ'):
        _run(source_run, source_run, n_seeds=1)


# ---------------------------------------------------------------------------
# source loading and validation
# ---------------------------------------------------------------------------


def test_config_toml_is_a_fallback_for_reports_without_config(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    report = _write_source(run_dir)
    del report['config']
    dump_json(run_dir / 'report.json', report)
    dump_config(_source_config(), run_dir / 'config.toml')

    _run(run_dir, tmp_path / 'out', n_seeds=1)
    ((seed_config, _),) = fake_tabpack.calls
    assert seed_config.d_block == _source_config().d_block
    assert seed_config.training == _source_config().training


def test_missing_source_report_raises(fake_tabpack, tmp_path):
    with pytest.raises(FileNotFoundError, match=r'report\.json'):
        _run(tmp_path / 'nope', tmp_path / 'out', n_seeds=1)


def test_missing_member_raises(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    _write_source(run_dir, ids=[1, 3, 9, 8], member_ids=[0, 1, 2, 3])
    with pytest.raises(ValueError, match=r'\[8, 9\] are missing'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)
    assert fake_tabpack.calls == []


def test_member_without_config_raises(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    report = _write_source(run_dir)
    report['members'][1]['config'] = None  # id 1
    dump_json(run_dir / 'report.json', report)
    with pytest.raises(ValueError, match=r'\[1\] have no config'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)


def test_unfinished_ensemble_members_use_member_configs(fake_tabpack, tmp_path):
    # Ensemble ids 3 and 5 never finished: they are only in member_configs.
    run_dir = tmp_path / 'src'
    _write_source(
        run_dir,
        member_ids=[0, 1, 2, 4],
        member_configs=[_member_config(i) for i in range(6)],
    )
    _run(run_dir, tmp_path / 'out', n_seeds=1)
    ((seed_config, _),) = fake_tabpack.calls
    assert seed_config.configs == [_member_config(i) for i in (1, 3, 5)]


def test_member_configs_alone_are_enough(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    report = _write_source(
        run_dir, member_configs=[_member_config(i) for i in range(6)]
    )
    del report['members']
    dump_json(run_dir / 'report.json', report)
    _run(run_dir, tmp_path / 'out', n_seeds=1)
    ((seed_config, _),) = fake_tabpack.calls
    assert seed_config.configs == [_member_config(i) for i in (1, 3, 5)]


def test_member_configs_disagreeing_with_members_raise(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    configs = [_member_config(i) for i in range(6)]
    configs[4] = _member_config(0)
    _write_source(run_dir, member_configs=configs)
    with pytest.raises(ValueError, match='member 4 has different configs'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)
    assert fake_tabpack.calls == []


def test_duplicate_member_ids_raise(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    _write_source(run_dir, member_ids=[1, 3, 5, 3])
    with pytest.raises(ValueError, match='appears twice'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)


@pytest.mark.parametrize('ensemble', [None, {'ids': []}, {}])
def test_missing_ensemble_raises(fake_tabpack, tmp_path, ensemble):
    run_dir = tmp_path / 'src'
    report = _write_source(run_dir)
    report['ensemble'] = ensemble
    dump_json(run_dir / 'report.json', report)
    with pytest.raises(ValueError, match='ensemble ids'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)


def test_non_int_ensemble_ids_raise(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    _write_source(run_dir, ids=[1, 3.0])
    with pytest.raises(ValueError, match='must be ints'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)


def test_source_must_be_a_tabpack_run(fake_tabpack, tmp_path):
    run_dir = tmp_path / 'src'
    report = _write_source(run_dir)
    report['method'] = 'homogeneous'
    dump_json(run_dir / 'report.json', report)
    with pytest.raises(ValueError, match="must be a 'tabpack' run"):
        _run(run_dir, tmp_path / 'out', n_seeds=1)

    report['method'] = 'tabpack'
    report['config'] = config_to_dict(MLPMethodConfig())
    dump_json(run_dir / 'report.json', report)
    with pytest.raises(ValueError, match='TabPackConfig'):
        _run(run_dir, tmp_path / 'out', n_seeds=1)


@pytest.mark.parametrize('n_seeds', [0, -1, True, 2.0])
def test_invalid_n_seeds_raises(fake_tabpack, source_run, tmp_path, n_seeds):
    with pytest.raises(ValueError, match='n_seeds'):
        _run(source_run, tmp_path / 'out', n_seeds=n_seeds)


def test_empty_source_run_raises(fake_tabpack, tmp_path):
    with pytest.raises(ValueError, match='source_run'):
        conservative.run(ConservativeEvalConfig(n_seeds=1), tmp_path / 'out')


def test_wrong_config_type_raises(fake_tabpack, tmp_path):
    with pytest.raises(TypeError, match='ConservativeEvalConfig'):
        conservative.run(_source_config(), tmp_path / 'out')


def test_missing_seed_score_raises(monkeypatch, source_run, tmp_path):
    fake = FakeTabPack()

    def no_test_score(config, output_dir):
        report = fake(config, output_dir)
        del report['metrics']['test']
        dump_json(Path(output_dir) / 'report.json', report)
        return report

    monkeypatch.setattr(tabpack, 'run', no_test_score)
    with pytest.raises(ValueError, match=r"seed-0 report has no metrics\['test'\]"):
        _run(source_run, tmp_path / 'out', n_seeds=1)


# ---------------------------------------------------------------------------
# end to end: a real TabPack source run on a tiny on-disk dataset
# ---------------------------------------------------------------------------


def _write_tiny_dataset(path: Path, n: int = 150) -> None:
    """A learnable binclass dataset in the official on-disk format."""
    rng = np.random.default_rng(0)
    x_num = rng.normal(size=(n, 3)).astype(np.float32)
    x_cat = rng.choice(np.array(['a', 'b', 'c']), size=(n, 1))
    logits = x_num[:, 0] - x_num[:, 1] + (x_cat[:, 0] == 'a')
    y = (logits + 0.3 * rng.normal(size=n) > 0.3).astype(np.int64)
    path.mkdir(parents=True)
    (path / 'info.json').write_text(
        json.dumps({'task': {'type': 'binclass', 'score': 'accuracy'}})
    )
    np.save(path / 'x_num.npy', x_num)
    np.save(path / 'x_cat.npy', x_cat)
    np.save(path / 'y.npy', y)
    split_dir = path / 'splits' / 'default'
    split_dir.mkdir(parents=True)
    perm = rng.permutation(n).astype(np.int64)
    for part, idx in zip(
        ('train', 'val', 'test'), np.split(perm, [90, 120]), strict=True
    ):
        np.save(split_dir / f'{part}.npy', np.sort(idx))


@pytest.mark.slow
def test_end_to_end_on_a_tiny_dataset(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data' / 'tiny'
    _write_tiny_dataset(data_dir)
    source_config = TabPackConfig(
        seed=0,
        n_models=4,
        data=DataConfig(path=str(data_dir)),
        d_block=8,
        optimizer=MuonAdamWConfig(shared_step=False),
        training=TrainingConfig(
            batch_size=16,
            patience=1,
            max_epochs=3,
            eval_batch_size=1024,
            amp_dtype=None,
            device='cpu',
        ),
        online_ensemble=OnlineEnsembleConfig(patience=4, max_ensemble_size=4),
    )
    source_dir = tmp_path / 'tabpack' / 'seed-0'
    source = tabpack.run(source_config, source_dir)
    selected = sorted(set(source['ensemble']['ids']))

    real_run = tabpack.run
    seeds: list[int] = []

    def spy(config, output_dir):
        seeds.append(config.seed)
        return real_run(config, output_dir)

    monkeypatch.setattr(tabpack, 'run', spy)
    config = ConservativeEvalConfig(source_run=str(source_dir), n_seeds=2)
    out = tmp_path / 'conservative'
    report = conservative.run(config, out)

    assert seeds == [0, 1]
    assert report['method'] == 'tabpack-conservative'
    assert report['selected_ids'] == selected
    assert report['seeds'] == [0, 1]
    selected_configs = [source['member_configs'][i] for i in selected]
    for seed in (0, 1):
        seed_dir = out / f'seed-{seed}'
        seed_report = load_json(seed_dir / 'report.json')
        assert seed_report['method'] == 'tabpack-conservative-seed'
        assert seed_report['seed'] == seed
        assert seed_report['n_models'] == len(selected)
        assert seed_report['member_configs'] == selected_configs
        for part in ('val', 'test'):
            score = seed_report['metrics'][part]['score']
            assert report['scores'][part][seed] == score
            assert 0.0 <= score <= 1.0
        assert load_config(seed_dir / 'config.toml') == dataclasses.replace(
            source_config,
            seed=seed,
            n_models=len(selected),
            configs=selected_configs,
            optimizer=MuonAdamWConfig(shared_step=True),
        )
        assert (seed_dir / 'predictions.npz').is_file()
    for part in ('val', 'test'):
        assert report['mean'][part] == pytest.approx(
            statistics.fmean(report['scores'][part])
        )
        assert report['std'][part] == pytest.approx(
            statistics.stdev(report['scores'][part])
        )
    assert load_json(out / 'report.json') == to_jsonable(report)
    assert load_config(out / 'config.toml') == config

    # Resuming a finished evaluation retrains nothing and rebuilds the same report.
    assert conservative.run(config, out) == report
    assert seeds == [0, 1]
