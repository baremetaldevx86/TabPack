"""Tests of the experiment orchestration in scripts/run_churn.py (a35).

No real training: step functions and method runners are replaced with fakes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tabpack_repro.config import ConservativeEvalConfig, MLPMethodConfig, TabPackConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'run_churn.py'
WRAPPER = REPO_ROOT / 'scripts' / 'run_churn.sh'


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location('run_churn', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['run_churn'] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


rc = _load_script()


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        '--runs-dir',
        str(tmp_path / 'runs'),
        '--results-dir',
        str(tmp_path / 'results'),
        '--configs-dir',
        str(tmp_path / 'configs'),
        '--reference-file',
        str(tmp_path / 'reference.json'),
        *extra,
    ]


def _write_report(path: Path, val: float = 0.86, test: float = 0.85) -> None:
    path.mkdir(parents=True, exist_ok=True)
    report = {'metrics': {'val': {'score': val}, 'test': {'score': test}}}
    (path / 'report.json').write_text(json.dumps(report))


class Recorder:
    """Fake step functions: record the calls, optionally fail on some steps."""

    def __init__(self, fail: set[str] = frozenset()) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, step: Any, ctx: Any) -> str:
        self.calls.append(step.name)
        if step.name in self.fail:
            raise RuntimeError(f'boom in {step.name}')
        if step.kind in ('run', 'conservative'):
            _write_report(step.output)
        return f'fake {step.name}'


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    for name in (
        'step_download',
        'step_run',
        'step_conservative',
        'step_summarize',
        'step_plots',
        'step_reference',
    ):
        monkeypatch.setattr(rc, name, rec)
    return rec


# ----------------------------------------------------------------------------------
# Arguments and plan
# ----------------------------------------------------------------------------------


def test_defaults_are_repo_paths() -> None:
    args = rc.parse_args([])
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.methods == ['mlp', 'homogeneous', 'tabpack', 'tabpack-conservative']
    assert args.runs_dir.resolve() == REPO_ROOT / 'runs' / 'churn'
    assert args.results_dir.resolve() == REPO_ROOT / 'results' / 'churn'
    assert args.configs_dir.resolve() == REPO_ROOT / 'configs' / 'churn'
    assert args.reference_file.resolve() == (
        REPO_ROOT / 'results' / 'reference' / 'churn_official.json'
    )
    assert args.device is None and args.conservative_seeds is None
    assert not (args.force or args.dry_run or args.skip_download)


def test_default_paths_are_relative_from_the_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    args = rc.parse_args([])
    assert args.runs_dir == Path('runs/churn')
    assert args.configs_dir == Path('configs/churn')


def test_seeds_and_methods_are_normalized() -> None:
    args = rc.parse_args(['--seeds', '3', '1', '3', '--methods', 'tabpack', 'mlp'])
    assert args.seeds == [3, 1]
    # Canonical method order, whatever the order on the command line.
    assert args.methods == ['mlp', 'tabpack']


@pytest.mark.parametrize(
    'argv',
    [
        ['--seeds', '-1'],
        ['--seeds', 'x'],
        ['--methods', 'catboost'],
        ['--conservative-seeds', '0'],
    ],
)
def test_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        rc.parse_args(argv)
    assert info.value.code == 2


def test_plan_order(tmp_path: Path) -> None:
    args = rc.parse_args(_args(tmp_path, '--seeds', '0', '1'))
    plan = rc.build_plan(args)
    assert [s.name for s in plan] == [
        'download',
        'mlp/seed-0',
        'homogeneous/seed-0',
        'tabpack/seed-0',
        'mlp/seed-1',
        'homogeneous/seed-1',
        'tabpack/seed-1',
        'tabpack-conservative',
        'summarize',
        'plots',
        'reference',
    ]
    runs = tmp_path / 'runs'
    by_name = {s.name: s for s in plan}
    step = by_name['homogeneous/seed-1']
    assert (step.kind, step.method, step.seed) == ('run', 'homogeneous', 1)
    assert step.output == runs / 'homogeneous' / 'seed-1'
    assert by_name['tabpack-conservative'].kind == 'conservative'
    assert by_name['tabpack-conservative'].output == runs / 'tabpack-conservative'
    assert by_name['summarize'].output == tmp_path / 'results'
    assert by_name['plots'].output == tmp_path / 'results' / 'figures'
    assert by_name['reference'].output == tmp_path / 'reference.json'


def test_plan_methods_filter_and_skip_download(tmp_path: Path) -> None:
    args = rc.parse_args(
        _args(tmp_path, '--seeds', '2', '--methods', 'tabpack', '--skip-download')
    )
    names = [s.name for s in rc.build_plan(args)]
    assert names == ['tabpack/seed-2', 'summarize', 'plots', 'reference']

    args = rc.parse_args(_args(tmp_path, '--methods', 'tabpack-conservative'))
    names = [s.name for s in rc.build_plan(args)]
    assert names == [
        'download',
        'tabpack-conservative',
        'summarize',
        'plots',
        'reference',
    ]


# ----------------------------------------------------------------------------------
# Execution: order, resume, force, failures
# ----------------------------------------------------------------------------------


def test_main_runs_every_step_in_order(
    tmp_path: Path, recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    code = rc.main(_args(tmp_path, '--seeds', '0', '1'))
    assert code == 0
    plan = rc.build_plan(rc.parse_args(_args(tmp_path, '--seeds', '0', '1')))
    assert recorder.calls == [s.name for s in plan]
    for method in ('mlp', 'homogeneous', 'tabpack'):
        for seed in (0, 1):
            assert (
                tmp_path / 'runs' / method / f'seed-{seed}' / 'report.json'
            ).exists()
    out = capsys.readouterr().out
    assert 'run_churn: summary' in out
    assert '11 ok, 0 skipped, 0 failed' in out
    assert 'Failures' not in out


def test_resume_skips_completed_runs(
    tmp_path: Path, recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(tmp_path / 'runs' / 'mlp' / 'seed-0', val=0.9, test=0.875)
    _write_report(tmp_path / 'runs' / 'tabpack-conservative')
    code = rc.main(_args(tmp_path, '--seeds', '0', '--skip-download'))
    assert code == 0
    # The conservative step is always called: conservative.run resumes by itself.
    assert recorder.calls == [
        'homogeneous/seed-0',
        'tabpack/seed-0',
        'tabpack-conservative',
        'summarize',
        'plots',
        'reference',
    ]
    out = capsys.readouterr().out
    assert '6 ok, 1 skipped, 0 failed' in out
    # The table shows the scores of the skipped (previously finished) run.
    mlp_row = next(line for line in out.splitlines() if line.startswith('mlp/seed-0'))
    assert 'skipped' in mlp_row and 'val 0.9000, test 0.8750' in mlp_row


def test_force_reruns_completed_runs(tmp_path: Path, recorder: Recorder) -> None:
    _write_report(tmp_path / 'runs' / 'mlp' / 'seed-0')
    code = rc.main(
        _args(
            tmp_path, '--seeds', '0', '--methods', 'mlp', '--force', '--skip-download'
        )
    )
    assert code == 0
    assert recorder.calls == ['mlp/seed-0', 'summarize', 'plots', 'reference']


def test_failures_are_counted_and_other_steps_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rec = Recorder(fail={'homogeneous/seed-1', 'plots'})
    for name in ('step_run', 'step_conservative', 'step_summarize', 'step_plots'):
        monkeypatch.setattr(rc, name, rec)
    monkeypatch.setattr(rc, 'step_reference', rec)
    code = rc.main(_args(tmp_path, '--seeds', '0', '1', '--skip-download'))
    assert code == 1
    # Everything was attempted, in order, despite the failures.
    assert rec.calls[-4:] == ['tabpack-conservative', 'summarize', 'plots', 'reference']
    assert 'tabpack/seed-1' in rec.calls
    assert not (tmp_path / 'runs' / 'homogeneous' / 'seed-1' / 'report.json').exists()
    out = capsys.readouterr().out
    assert '8 ok, 0 skipped, 2 failed' in out
    failures = out.split('Failures:')[1]
    assert 'homogeneous/seed-1: RuntimeError: boom in homogeneous/seed-1' in failures
    assert 'plots: RuntimeError: boom in plots' in failures


def test_step_skipped_is_not_a_failure(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def skip(step: Any, ctx: Any) -> str:
        raise rc.StepSkipped('nothing to plot')

    monkeypatch.setattr(rc, 'step_plots', skip)
    ctx = rc.Context(rc.parse_args(_args(tmp_path, '--methods', 'mlp', '--seeds', '0')))
    plan = rc.build_plan(ctx.args)
    results = rc.execute_plan(plan, ctx)
    statuses = {r.step.name: (r.status, r.detail) for r in results}
    assert statuses['plots'] == ('skipped', 'nothing to plot')
    assert all(r.status == 'ok' for r in results if r.step.name != 'plots')


def test_keyboard_interrupt_stops_the_plan(
    tmp_path: Path,
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(step: Any, ctx: Any) -> str:
        if step.name == 'homogeneous/seed-0':
            raise KeyboardInterrupt
        return recorder(step, ctx)

    monkeypatch.setattr(rc, 'step_run', interrupt)
    code = rc.main(_args(tmp_path, '--seeds', '0', '--skip-download'))
    assert code == 130
    assert recorder.calls == ['mlp/seed-0']
    assert 'not run' in capsys.readouterr().out


# ----------------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------------


def test_dry_run_prints_the_plan_and_runs_nothing(
    tmp_path: Path,
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded: list[Path] = []

    def fake_load(path: Path) -> Any:
        loaded.append(path)
        method = path.stem
        return {
            'mlp': MLPMethodConfig(),
            'tabpack': TabPackConfig(),
            'tabpack-conservative': ConservativeEvalConfig(),
        }[method]

    monkeypatch.setattr(rc, 'load_method_config', fake_load)
    _write_report(tmp_path / 'runs' / 'mlp' / 'seed-0')
    _write_report(tmp_path / 'runs' / 'tabpack-conservative')
    argv = _args(
        tmp_path,
        '--dry-run',
        '--seeds',
        '0',
        '1',
        '--methods',
        'mlp',
        'tabpack',
        'tabpack-conservative',
        '--device',
        'cpu',
    )
    assert rc.main(argv) == 0
    assert recorder.calls == []
    assert not (tmp_path / 'results').exists()
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith('Plan (9 steps; dry run')
    mlp0 = next(line for line in lines if 'mlp/seed-0' in line)
    assert ' skip ' in mlp0
    conservative = next(line for line in lines if 'tabpack-conservative ' in line)
    assert ' resume ' in conservative
    mlp1 = next(line for line in lines if 'mlp/seed-1' in line)
    assert ' run ' in mlp1 and 'seed=1, device=cpu' in mlp1
    assert str(tmp_path / 'runs' / 'mlp' / 'seed-1') in mlp1
    assert 'source_run=' + str(tmp_path / 'runs' / 'tabpack' / 'seed-0') in out
    assert [p.name for p in loaded] == [
        'mlp.toml',
        'tabpack.toml',
        'tabpack-conservative.toml',
    ]
    assert out.count(': ok') == 3
    assert '(produced by this plan)' in out


def test_dry_run_force_and_config_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(path: Path) -> Any:
        raise ValueError(f'unknown key in {path.name}')

    monkeypatch.setattr(rc, 'load_method_config', broken)
    _write_report(tmp_path / 'runs' / 'mlp' / 'seed-0')
    argv = _args(tmp_path, '--dry-run', '--seeds', '0', '--methods', 'mlp', '--force')
    assert rc.main(argv) == 1
    out = capsys.readouterr().out
    assert 'rerun (--force)' in out
    assert 'ERROR ValueError: unknown key in mlp.toml' in out


def test_dry_run_notes_a_missing_conservative_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rc, 'load_method_config', lambda p: ConservativeEvalConfig())
    argv = _args(tmp_path, '--dry-run', '--methods', 'tabpack-conservative')
    assert rc.main(argv) == 0
    assert 'NOT produced by this plan' in capsys.readouterr().out


# ----------------------------------------------------------------------------------
# The real step functions, with fake method runners / reporting functions
# ----------------------------------------------------------------------------------


def test_prepare_run_config_overrides_seed_and_device() -> None:
    config = MLPMethodConfig()
    new = rc.prepare_run_config(config, method='mlp', seed=3, device='cpu')
    assert new.seed == 3 and new.training.device == 'cpu'
    assert config.seed == 0 and config.training.device == 'auto'  # not mutated
    assert new.data == config.data and new.model == config.model
    kept = rc.prepare_run_config(config, method='mlp', seed=1, device=None)
    assert kept.training.device == 'auto'
    with pytest.raises(ValueError, match='expected'):
        rc.prepare_run_config(config, method='tabpack', seed=0, device=None)


@pytest.mark.parametrize('method', ['mlp', 'homogeneous', 'tabpack'])
def test_real_configs_can_be_prepared(method: str) -> None:
    args = rc.parse_args([])
    config = rc.load_method_config(rc.config_path(args, method))
    new = rc.prepare_run_config(config, method=method, seed=2, device='cpu')
    assert new.method == method and new.seed == 2 and new.training.device == 'cpu'


def test_real_conservative_config_can_be_prepared(tmp_path: Path) -> None:
    args = rc.parse_args([])
    config = rc.load_method_config(rc.config_path(args, 'tabpack-conservative'))
    new = rc.prepare_conservative_config(config, source_run=tmp_path, n_seeds=None)
    assert new.source_run == str(tmp_path) and new.n_seeds == config.n_seeds
    assert (
        rc.prepare_conservative_config(config, source_run=tmp_path, n_seeds=2).n_seeds
        == 2
    )
    with pytest.raises(ValueError, match='expected'):
        rc.prepare_conservative_config(
            MLPMethodConfig(), source_run=tmp_path, n_seeds=None
        )


def test_step_run_calls_the_method_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, Path]] = []

    def fake_runner(config: Any, output_dir: Path) -> dict[str, Any]:
        calls.append((config, output_dir))
        return {'metrics': {'val': {'score': 0.5}, 'test': {'score': 0.25}}}

    monkeypatch.setattr(rc, 'load_method_config', lambda path: MLPMethodConfig())
    monkeypatch.setattr(rc, 'get_method_runner', lambda method: fake_runner)
    args = rc.parse_args(_args(tmp_path, '--device', 'cpu', '--force'))
    step = next(s for s in rc.build_plan(args) if s.name == 'mlp/seed-4')
    old = step.output / 'report.json'
    _write_report(step.output)
    detail = rc.step_run(step, rc.Context(args))
    assert detail == 'val 0.5000, test 0.2500'
    ((config, output_dir),) = calls
    assert output_dir == tmp_path / 'runs' / 'mlp' / 'seed-4'
    assert config.seed == 4 and config.training.device == 'cpu'
    assert not old.exists()  # --force removes the stale report before rerunning


def test_get_method_runner_maps_methods_to_modules() -> None:
    from tabpack_repro.methods import conservative, homogeneous, mlp, tabpack

    assert rc.get_method_runner('mlp') is mlp.run
    assert rc.get_method_runner('homogeneous') is homogeneous.run
    assert rc.get_method_runner('tabpack') is tabpack.run
    assert rc.get_method_runner('tabpack-conservative') is conservative.run


def test_step_conservative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, Path]] = []

    def fake_runner(config: Any, output_dir: Path) -> dict[str, Any]:
        calls.append((config, output_dir))
        return {'mean': {'val': 0.8, 'test': 0.75}}

    monkeypatch.setattr(
        rc, 'load_method_config', lambda path: ConservativeEvalConfig(n_seeds=5)
    )
    monkeypatch.setattr(rc, 'get_method_runner', lambda method: fake_runner)
    args = rc.parse_args(
        _args(
            tmp_path, '--methods', 'tabpack-conservative', '--conservative-seeds', '2'
        )
    )
    step = next(s for s in rc.build_plan(args) if s.kind == 'conservative')
    ctx = rc.Context(args)

    # The source run (TabPack seed 0) is required.
    with pytest.raises(FileNotFoundError, match='seed-0'):
        rc.step_conservative(step, ctx)
    assert calls == []

    _write_report(tmp_path / 'runs' / 'tabpack' / 'seed-0')
    assert rc.step_conservative(step, ctx) == 'val 0.8000, test 0.7500'
    ((config, output_dir),) = calls
    assert config.source_run == str(tmp_path / 'runs' / 'tabpack' / 'seed-0')
    assert config.n_seeds == 2
    assert output_dir == tmp_path / 'runs' / 'tabpack-conservative'


def test_step_conservative_force_removes_seed_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rc, 'load_method_config', lambda path: ConservativeEvalConfig(n_seeds=2)
    )
    seen: list[list[str]] = []

    def fake_runner(config: Any, output_dir: Path) -> dict[str, Any]:
        seen.append(
            sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob('*.json'))
        )
        return {}

    monkeypatch.setattr(rc, 'get_method_runner', lambda method: fake_runner)
    out = tmp_path / 'runs' / 'tabpack-conservative'
    for sub in ('', 'seed-0', 'seed-1', 'seed-2'):
        _write_report(out / sub)
    _write_report(tmp_path / 'runs' / 'tabpack' / 'seed-0')
    args = rc.parse_args(
        _args(tmp_path, '--methods', 'tabpack-conservative', '--force')
    )
    step = next(s for s in rc.build_plan(args) if s.kind == 'conservative')
    rc.step_conservative(step, rc.Context(args))
    # Only the aggregate and the seeds of this evaluation (0..n_seeds-1) are removed.
    assert seen == [['seed-2/report.json']]


def test_step_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tabpack_repro.data import download

    names: list[str] = []

    def fake_download(name: str = 'churn', **kwargs: Any) -> Path:
        names.append(name)
        return tmp_path / name

    monkeypatch.setattr(download, 'download_dataset', fake_download)
    args = rc.parse_args(_args(tmp_path))
    assert rc.step_download(rc.build_plan(args)[0], rc.Context(args)) == str(
        tmp_path / 'churn'
    )
    assert names == ['churn']


def _patch_reporting(monkeypatch: pytest.MonkeyPatch, calls: list[Any]) -> None:
    from tabpack_repro.reporting import plots, summarize

    monkeypatch.setattr(
        summarize, 'collect_runs', lambda d: calls.append(('collect', d)) or [{'r': 1}]
    )
    monkeypatch.setattr(
        summarize,
        'summarize',
        lambda runs: calls.append(('summarize', runs)) or {'s': 1},
    )
    monkeypatch.setattr(
        summarize, 'write_summary', lambda s, d: calls.append(('write', s, d))
    )
    monkeypatch.setattr(summarize, 'to_markdown', lambda s: '| method |\n')
    for name in (
        'plot_method_comparison',
        'plot_online_ensemble_history',
        'plot_member_scores',
    ):
        monkeypatch.setattr(
            plots, name, lambda obj, path, name=name: calls.append((name, obj, path))
        )


def test_step_summarize_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    _patch_reporting(monkeypatch, calls)
    args = rc.parse_args(_args(tmp_path))
    plan = {s.name: s for s in rc.build_plan(args)}
    ctx = rc.Context(args)

    rc.step_summarize(plan['summarize'], ctx)
    assert ctx.summary == {'s': 1}
    assert calls == [
        ('collect', tmp_path / 'runs'),
        ('summarize', [{'r': 1}]),
        ('write', {'s': 1}, tmp_path / 'results'),
    ]

    # Without the TabPack seed-0 report only the comparison is plotted.
    calls.clear()
    assert rc.step_plots(plan['plots'], ctx).startswith('1 figures')
    figures = tmp_path / 'results' / 'figures'
    assert calls == [
        ('plot_method_comparison', {'s': 1}, figures / 'method_comparison.png')
    ]

    calls.clear()
    _write_report(tmp_path / 'runs' / 'tabpack' / 'seed-0', test=0.5)
    rc.step_plots(plan['plots'], ctx)
    assert [c[0] for c in calls] == [
        'plot_method_comparison',
        'plot_online_ensemble_history',
        'plot_member_scores',
    ]
    assert calls[1][1]['metrics']['test']['score'] == 0.5
    assert calls[1][2] == figures / 'tabpack_seed0_online_ensemble.png'
    assert calls[2][2] == figures / 'tabpack_seed0_member_scores.png'


def test_step_summarize_and_plots_skip_without_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    _patch_reporting(monkeypatch, calls)
    from tabpack_repro.reporting import summarize

    monkeypatch.setattr(summarize, 'collect_runs', lambda d: [])
    args = rc.parse_args(_args(tmp_path))
    plan = {s.name: s for s in rc.build_plan(args)}
    ctx = rc.Context(args)
    with pytest.raises(rc.StepSkipped, match=r'no report\.json'):
        rc.step_summarize(plan['summarize'], ctx)
    with pytest.raises(rc.StepSkipped, match='nothing to plot'):
        rc.step_plots(plan['plots'], ctx)
    assert calls == []


def test_step_reference_refreshes_from_the_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tabpack_repro.reporting import reference

    clone = tmp_path / 'clone'
    monkeypatch.setattr(reference, 'get_reference_dir', lambda: clone)
    monkeypatch.setattr(
        reference, 'extract_churn_reference', lambda d: {'from': str(d), 'score': 0.86}
    )
    monkeypatch.setattr(
        reference, 'load_saved_reference', lambda p: pytest.fail('must not be called')
    )
    args = rc.parse_args(_args(tmp_path))
    step = rc.build_plan(args)[-1]
    ctx = rc.Context(args)
    assert 'refreshed' in rc.step_reference(step, ctx)
    saved = json.loads((tmp_path / 'reference.json').read_text())
    assert saved == {'from': str(clone), 'score': 0.86} == ctx.reference


def test_step_reference_uses_the_saved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tabpack_repro.reporting import reference

    loaded: list[Path] = []
    monkeypatch.setattr(
        reference, 'load_saved_reference', lambda p: loaded.append(p) or {'saved': 1}
    )
    args = rc.parse_args(_args(tmp_path))
    step = rc.build_plan(args)[-1]
    ctx = rc.Context(args)

    # No clone.
    monkeypatch.setattr(reference, 'get_reference_dir', lambda: None)
    assert rc.step_reference(step, ctx) == f'loaded {tmp_path / "reference.json"}'
    assert ctx.reference == {'saved': 1}

    # A broken clone falls back to the saved file too.
    def broken(d: Path) -> dict[str, Any]:
        raise OSError('no reports')

    monkeypatch.setattr(reference, 'get_reference_dir', lambda: tmp_path)
    monkeypatch.setattr(reference, 'extract_churn_reference', broken)
    assert rc.step_reference(step, ctx).startswith('loaded')
    assert loaded == [tmp_path / 'reference.json'] * 2
    assert not (tmp_path / 'reference.json').exists()


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('report', 'expected'),
    [
        ({'metrics': {'val': {'score': 0.5}, 'test': {'score': 0.25}}}, (0.5, 0.25)),
        ({'mean': {'val': 0.5, 'test': 0.25}}, (0.5, 0.25)),
        ({'mean': {'val': {'score': 0.5}}}, (0.5, None)),
        ({'metrics': {}}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
)
def test_report_scores(report: Any, expected: tuple[Any, Any]) -> None:
    assert rc.report_scores(report) == expected


def test_format_reference() -> None:
    data = {
        'headline': {'text': '85.75 ± 0.14', 'protocol': 'conservative', 'n_seeds': 5},
        'paper_table14': {
            'churn': {
                'MLP': {'mean': 0.8562, 'std': 0.0018},
                'TabPack': {'mean': 0.8575},
                'XGBoost': {'mean': 0.8605, 'std': 0.002},
            }
        },
    }
    assert rc.format_reference(data) == [
        'official TabPack: 85.75 ± 0.14 (conservative protocol, 5 seeds)',
        'paper Table 14, MLP: 85.62 ± 0.18',
        'paper Table 14, TabPack: 85.75',
    ]
    assert rc.format_reference({'headline': {'text': 'x'}}) == ['official TabPack: x']
    for bad in (None, [], {}, {'headline': 1, 'paper_table14': {'churn': [1]}}):
        assert rc.format_reference(bad) == []


def test_format_reference_of_the_committed_file() -> None:
    path = REPO_ROOT / 'results' / 'reference' / 'churn_official.json'
    if not path.is_file():
        pytest.skip('results/reference/churn_official.json is not available')
    lines = rc.format_reference(json.loads(path.read_text()))
    assert lines[0].startswith('official TabPack: ')
    assert any(line.startswith('paper Table 14, MLP: ') for line in lines)


def test_summary_shows_the_reference(
    tmp_path: Path,
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_reference(step: Any, ctx: Any) -> str:
        ctx.reference = {'headline': {'text': '85.75 ± 0.14'}}
        return 'loaded'

    monkeypatch.setattr(rc, 'step_reference', fake_reference)
    assert rc.main(_args(tmp_path, '--methods', 'mlp', '--seeds', '0')) == 0
    out = capsys.readouterr().out
    assert 'Official Churn numbers (test accuracy):\n  official TabPack: 85.75' in out


@pytest.mark.parametrize(
    ('seconds', 'text'),
    [(0.04, '0.0s'), (12.34, '12.3s'), (125, '2m05s'), (3725, '1h02m05s')],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert rc.format_duration(seconds) == text


# ----------------------------------------------------------------------------------
# scripts/run_churn.sh
# ----------------------------------------------------------------------------------


def _fake_uv(tmp_path: Path, exit_code: int) -> dict[str, str]:
    fake = tmp_path / 'fake-uv'
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'echo "cwd=$PWD"\n'
        'echo "args=$*"\n'
        'echo "to stderr" >&2\n'
        f'exit {exit_code}\n'
    )
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != 'RUN_CHURN_LOG'}
    return {**env, 'UV': str(fake)}


def test_wrapper_runs_from_the_repo_root_and_logs(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    result = subprocess.run(
        ['bash', str(WRAPPER), '--seeds', '0', '--runs-dir', str(runs)],
        cwd=tmp_path,
        env=_fake_uv(tmp_path, 0),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f'cwd={REPO_ROOT}' in result.stdout
    assert f'args=run python -u scripts/run_churn.py --seeds 0 --runs-dir {runs}' in (
        result.stdout
    )
    log = (runs / 'run.log').read_text()
    assert log.startswith('==== ')
    assert 'to stderr' in log and f'cwd={REPO_ROOT}' in log

    # A second invocation appends to the log.
    subprocess.run(
        ['bash', str(WRAPPER), f'--runs-dir={runs}'],
        env=_fake_uv(tmp_path, 0),
        capture_output=True,
        check=True,
    )
    assert (runs / 'run.log').read_text().count('==== ') == 2


def test_wrapper_propagates_the_exit_code(tmp_path: Path) -> None:
    log = tmp_path / 'custom.log'
    result = subprocess.run(
        ['bash', str(WRAPPER), '--runs-dir', str(tmp_path / 'runs')],
        env={**_fake_uv(tmp_path, 3), 'RUN_CHURN_LOG': str(log)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert 'args=' in log.read_text()
    assert not (tmp_path / 'runs').exists()


def test_wrapper_does_not_log_dry_runs(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    result = subprocess.run(
        ['bash', str(WRAPPER), '--dry-run', '--runs-dir', str(runs)],
        env=_fake_uv(tmp_path, 0),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert 'args=run python -u scripts/run_churn.py --dry-run' in result.stdout
    assert not runs.exists()
