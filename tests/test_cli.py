"""Tests for the command-line interface (a34).

The CLI is tested in-process through ``main(argv)``. Every heavy target (the
methods, download, reporting) is replaced by a fake via monkeypatch, so these tests
check argument handling and dispatch, not the methods themselves.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tabpack_repro import cli
from tabpack_repro import config as config_lib
from tabpack_repro.config import (
    ConservativeEvalConfig,
    HomogeneousEnsembleConfig,
    MLPMethodConfig,
    TabPackConfig,
)

SRC_DIR = Path(cli.__file__).resolve().parents[1]
CONFIGS_DIR = Path(__file__).resolve().parents[1] / 'configs' / 'churn'
SUBCOMMANDS = ['download', 'run', 'conservative', 'summarize', 'reference']
RUN_METHODS = {
    'mlp': ('tabpack_repro.methods.mlp', MLPMethodConfig),
    'homogeneous': ('tabpack_repro.methods.homogeneous', HomogeneousEnsembleConfig),
    'tabpack': ('tabpack_repro.methods.tabpack', TabPackConfig),
}


@pytest.fixture(autouse=True)
def _restore_package_log_level():
    logger = logging.getLogger('tabpack_repro')
    level = logger.level
    yield
    logger.setLevel(level)


def _report(method: str, seed: int = 0, test: float = 0.857, time_sec: float = 1.5):
    return {
        'schema_version': 1,
        'method': method,
        'dataset': 'churn',
        'seed': seed,
        'metrics': {'val': {'score': 0.87625}, 'test': {'score': test}},
        'time_sec': time_sec,
    }


def _source_run(tmp_path: Path) -> Path:
    source = tmp_path / 'runs' / 'tabpack' / 'seed-0'
    source.mkdir(parents=True)
    (source / 'report.json').write_text(json.dumps(_report('tabpack')))
    return source


class _FakeRun:
    """Records every call and returns a report built from the config."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.error = error

    def __call__(self, config, output_dir):
        self.calls.append((config, output_dir))
        if self.error is not None:
            raise self.error
        if config.method == 'tabpack-conservative':
            return {
                'method': 'tabpack-conservative',
                'source_run': config.source_run,
                'n_seeds': config.n_seeds,
                'seeds': list(range(config.n_seeds)),
                'scores': {
                    'val': [0.875] * config.n_seeds,
                    'test': [0.855 + 0.001 * s for s in range(config.n_seeds)],
                },
            }
        return _report(config.method, seed=config.seed)


@pytest.fixture
def fake_load_config(monkeypatch, tmp_path):
    """Make load_config return a given config for an (empty) existing file."""
    loaded: list[Path] = []

    def install(config) -> Path:
        path = tmp_path / f'{config.method}.toml'
        path.write_text('# fake\n')

        def load_config(p):
            loaded.append(Path(p))
            return config

        monkeypatch.setattr(config_lib, 'load_config', load_config)
        return path

    install.loaded = loaded
    return install


def _install_fake_runs(monkeypatch) -> dict[str, _FakeRun]:
    fakes = {}
    for method, (module, _) in RUN_METHODS.items():
        fakes[method] = _FakeRun()
        monkeypatch.setattr(f'{module}.run', fakes[method])
    fakes['tabpack-conservative'] = _FakeRun()
    monkeypatch.setattr(
        'tabpack_repro.methods.conservative.run', fakes['tabpack-conservative']
    )
    return fakes


# ----------------------------------------------------------------------------------
# Help and usage errors
# ----------------------------------------------------------------------------------


def test_top_level_help(capsys) -> None:
    assert cli.main(['--help']) == 0
    out = capsys.readouterr().out
    assert out.startswith('usage: tabpack-repro')
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize('name', SUBCOMMANDS)
def test_subcommand_help(name: str, capsys) -> None:
    assert cli.main([name, '--help']) == 0
    out = capsys.readouterr().out
    assert out.startswith(f'usage: tabpack-repro {name}')


@pytest.mark.parametrize(
    ('name', 'options'),
    [
        ('download', ['--name', '--data-dir', '--cache-dir', '--force']),
        ('run', ['--config', '--seed', '--output', '--device']),
        ('conservative', ['--source-run', '--n-seeds', '--output']),
        ('summarize', ['--runs-dir', '--output']),
        ('reference', ['--output', '--reference-dir']),
    ],
)
def test_subcommand_help_lists_the_contract_options(name, options, capsys) -> None:
    assert cli.main([name, '--help']) == 0
    out = capsys.readouterr().out
    for option in options:
        assert option in out


@pytest.mark.parametrize(
    'argv',
    [
        [],
        ['frobnicate'],
        ['run'],
        ['run', '--config', 'x.toml'],
        ['run', '--output', 'out'],
        ['conservative', '--output', 'out'],
        ['conservative', '--source-run', 'src'],
        ['summarize', '--runs-dir', 'runs'],
        ['summarize', '--output', 'out'],
        ['reference'],
        ['download', '--bogus'],
    ],
)
def test_missing_or_unknown_arguments_exit_2(argv, capsys) -> None:
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'error:' in captured.err
    assert 'usage: tabpack-repro' in captured.err


@pytest.mark.parametrize(
    ('extra', 'message'),
    [
        (['--seed', '-1'], 'invalid seed -1'),
        (['--seed', 'abc'], "invalid seed 'abc'"),
        (['--seed', str(2**32)], 'must be in [0, 4294967296)'),
        (['--device', 'gpu'], "invalid device 'gpu'"),
        (['--device', 'cuda:x'], "invalid device 'cuda:x'"),
    ],
)
def test_invalid_run_values_exit_2(extra, message, monkeypatch, capsys) -> None:
    fakes = _install_fake_runs(monkeypatch)
    argv = ['run', '--config', 'x.toml', '--output', 'out', *extra]
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err
    assert not any(f.calls for f in fakes.values())


@pytest.mark.parametrize('value', ['0', '-2', 'five'])
def test_invalid_n_seeds_exit_2(value, capsys) -> None:
    argv = ['conservative', '--source-run', 'src', '--output', 'o', '--n-seeds', value]
    assert cli.main(argv) == 2
    assert 'invalid value' in capsys.readouterr().err


def test_verbose_and_quiet_are_exclusive(capsys) -> None:
    assert cli.main(['download', '-v', '-q']) == 2
    assert 'not allowed with argument' in capsys.readouterr().err


def test_python_dash_m_help_is_fast_and_does_not_import_torch() -> None:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(SRC_DIR)])}
    result = subprocess.run(
        [sys.executable, '-m', 'tabpack_repro', '--help'],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('usage: tabpack-repro')

    code = (
        'import sys\n'
        'from tabpack_repro.cli import main\n'
        "for name in ['download', 'run', 'conservative', 'summarize', 'reference']:\n"
        "    assert main([name, '--help']) == 0\n"
        "heavy = sorted({'torch', 'numpy', 'optuna', 'pandas'} & set(sys.modules))\n"
        'assert not heavy, heavy\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize('method', list(RUN_METHODS))
def test_run_dispatches_on_config_method(
    method, monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    fakes = _install_fake_runs(monkeypatch)
    config = RUN_METHODS[method][1]()
    path = fake_load_config(config)
    output = tmp_path / 'runs' / method / 'seed-0'

    assert cli.main(['run', '--config', str(path), '--output', str(output)]) == 0

    assert fake_load_config.loaded == [path]
    assert len(fakes[method].calls) == 1
    got_config, got_output = fakes[method].calls[0]
    assert got_config == config  # no override requested -> unchanged
    assert Path(got_output) == output
    for other, fake in fakes.items():
        if other != method:
            assert fake.calls == []

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    result, out_line = lines
    assert result.split()[0] == method
    assert 'seed=0' in result
    assert 'test_score=0.85700' in result
    assert 'val_score=0.87625' in result
    assert 'time=1.5s' in result
    assert out_line == f'output: {output}'


@pytest.mark.parametrize('method', list(RUN_METHODS))
def test_run_seed_and_device_overrides_reach_the_config(
    method, monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    fakes = _install_fake_runs(monkeypatch)
    config = RUN_METHODS[method][1]()
    assert (config.seed, config.training.device) == (0, 'auto')
    path = fake_load_config(config)
    argv = ['run', '--config', str(path), '--output', str(tmp_path / 'o')]

    assert cli.main([*argv, '--seed', '3', '--device', 'cpu']) == 0
    got, _ = fakes[method].calls[-1]
    assert got.seed == 3
    assert got.training.device == 'cpu'
    # Nothing else changes, and the loaded config is not mutated.
    assert got == dataclasses.replace(
        config, seed=3, training=dataclasses.replace(config.training, device='cpu')
    )
    assert (config.seed, config.training.device) == (0, 'auto')
    assert 'seed=3' in capsys.readouterr().out

    assert cli.main([*argv, '--seed', '4']) == 0
    got, _ = fakes[method].calls[-1]
    assert (got.seed, got.training.device) == (4, 'auto')

    assert cli.main([*argv, '--device', 'cuda:1']) == 0
    got, _ = fakes[method].calls[-1]
    assert (got.seed, got.training.device) == (0, 'cuda:1')


def test_run_bad_config_path_exits_1(monkeypatch, tmp_path, capsys) -> None:
    fakes = _install_fake_runs(monkeypatch)
    missing = tmp_path / 'nope.toml'
    assert cli.main(['run', '--config', str(missing), '--output', 'out']) == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err.strip() == (
        f'tabpack-repro run: error: config file not found: {missing}'
    )
    assert not any(f.calls for f in fakes.values())


def test_run_invalid_config_exits_1_with_the_loader_message(
    monkeypatch, tmp_path, capsys
) -> None:
    fakes = _install_fake_runs(monkeypatch)
    path = tmp_path / 'bad.toml'
    path.write_text('[model]\nbogus = 1\n')

    def load_config(p):
        raise ValueError("unknown key 'model.bogus'")

    monkeypatch.setattr(config_lib, 'load_config', load_config)
    assert cli.main(['run', '--config', str(path), '--output', 'out']) == 1
    err = capsys.readouterr().err
    assert f'invalid config file {path}' in err
    assert "ValueError: unknown key 'model.bogus'" in err
    assert 'Traceback' not in err
    assert not any(f.calls for f in fakes.values())


def test_run_method_failure_exits_1(
    monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    _install_fake_runs(monkeypatch)
    failing = _FakeRun(error=RuntimeError('CUDA out of memory'))
    monkeypatch.setattr('tabpack_repro.methods.tabpack.run', failing)
    path = fake_load_config(TabPackConfig())
    argv = ['run', '--config', str(path), '--output', str(tmp_path / 'o')]

    assert cli.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err.strip() == (
        'tabpack-repro run: error: RuntimeError: CUDA out of memory'
    )

    # --verbose adds the traceback.
    assert cli.main([*argv, '-v']) == 1
    err = capsys.readouterr().err
    assert 'Traceback (most recent call last)' in err
    assert err.rstrip().endswith('RuntimeError: CUDA out of memory')


def test_run_keyboard_interrupt_exits_130(
    monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    _install_fake_runs(monkeypatch)
    monkeypatch.setattr(
        'tabpack_repro.methods.mlp.run', _FakeRun(error=KeyboardInterrupt())
    )
    path = fake_load_config(MLPMethodConfig())
    assert cli.main(['run', '--config', str(path), '--output', str(tmp_path)]) == 130
    assert 'interrupted' in capsys.readouterr().err


def test_run_unknown_method_exits_1(
    monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    @dataclasses.dataclass
    class Weird:
        method: str = 'xgboost'
        seed: int = 0

    path = fake_load_config(Weird())
    assert cli.main(['run', '--config', str(path), '--output', str(tmp_path)]) == 1
    assert "unknown method 'xgboost'" in capsys.readouterr().err


def test_run_conservative_config(
    monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    fakes = _install_fake_runs(monkeypatch)
    source = _source_run(tmp_path)
    config = ConservativeEvalConfig(source_run=str(source), n_seeds=3)
    path = fake_load_config(config)
    argv = ['run', '--config', str(path), '--output', str(tmp_path / 'c')]

    assert cli.main(argv) == 0
    assert fakes['tabpack-conservative'].calls == [(config, tmp_path / 'c')]
    out = capsys.readouterr().out
    assert out.startswith('tabpack-conservative  n_seeds=3')
    assert 'test_score=0.85600±0.00100' in out

    for extra in (['--seed', '1'], ['--device', 'cpu']):
        assert cli.main([*argv, *extra]) == 2
        err = capsys.readouterr().err
        assert f'{extra[0]} does not apply to a tabpack-conservative config' in err
    assert len(fakes['tabpack-conservative'].calls) == 1


def test_run_handles_missing_report_fields(
    monkeypatch, fake_load_config, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        'tabpack_repro.methods.mlp.run',
        lambda config, output_dir: {'method': 'mlp', 'metrics': {'test': {}}},
    )
    path = fake_load_config(MLPMethodConfig())
    assert cli.main(['run', '--config', str(path), '--output', str(tmp_path)]) == 0
    result = capsys.readouterr().out.splitlines()[0]
    assert result.startswith('mlp  seed=n/a  val_score=n/a  test_score=n/a  time=')


# ----------------------------------------------------------------------------------
# conservative
# ----------------------------------------------------------------------------------


def test_conservative_builds_the_config(monkeypatch, tmp_path, capsys) -> None:
    fakes = _install_fake_runs(monkeypatch)
    source = _source_run(tmp_path)
    output = tmp_path / 'runs' / 'tabpack-conservative'
    argv = ['conservative', '--source-run', str(source), '--output', str(output)]

    assert cli.main(argv) == 0
    config, got_output = fakes['tabpack-conservative'].calls[-1]
    assert config == ConservativeEvalConfig(source_run=str(source), n_seeds=5)
    assert Path(got_output) == output
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith('tabpack-conservative  n_seeds=5  val_score=0.87500±0')
    assert 'test_score=0.85700±0.00158' in lines[0]
    assert lines[1] == f'output: {output}'

    assert cli.main([*argv, '--n-seeds', '2']) == 0
    config, _ = fakes['tabpack-conservative'].calls[-1]
    assert config.n_seeds == 2
    assert fakes['tabpack'].calls == []


def test_conservative_missing_source_run_exits_1(monkeypatch, tmp_path, capsys):
    fakes = _install_fake_runs(monkeypatch)
    source = tmp_path / 'missing'
    argv = ['conservative', '--source-run', str(source), '--output', str(tmp_path)]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert f'source run report not found: {source / "report.json"}' in err
    assert fakes['tabpack-conservative'].calls == []

    # The same check applies to `run` with a tabpack-conservative config.
    path = tmp_path / 'c.toml'
    path.write_text('')
    config = ConservativeEvalConfig(source_run=str(source))
    monkeypatch.setattr(config_lib, 'load_config', lambda p: config)
    assert cli.main(['run', '--config', str(path), '--output', str(tmp_path)]) == 1
    assert 'source run report not found' in capsys.readouterr().err
    assert fakes['tabpack-conservative'].calls == []


# ----------------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------------


def test_download_passes_the_options(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def download_dataset(name='churn', *, data_dir=None, cache_dir=None, force=False):
        calls.append((name, data_dir, cache_dir, force))
        return Path(data_dir or '/default') / name

    monkeypatch.setattr(
        'tabpack_repro.data.download.download_dataset', download_dataset
    )
    assert cli.main(['download']) == 0
    assert calls[-1] == ('churn', None, None, False)
    assert capsys.readouterr().out.strip() == (
        f"dataset 'churn' is ready at {Path('/default/churn')}"
    )

    argv = ['download', '--name', 'adult', '--data-dir', str(tmp_path)]
    argv += ['--cache-dir', str(tmp_path / 'c'), '--force']
    assert cli.main(argv) == 0
    assert calls[-1] == ('adult', tmp_path, tmp_path / 'c', True)
    assert str(tmp_path / 'adult') in capsys.readouterr().out


def test_download_failure_exits_1(monkeypatch, capsys) -> None:
    def download_dataset(*args, **kwargs):
        raise RuntimeError('The data bundle is corrupt')

    monkeypatch.setattr(
        'tabpack_repro.data.download.download_dataset', download_dataset
    )
    assert cli.main(['download']) == 1
    assert capsys.readouterr().err.strip() == (
        'tabpack-repro download: error: RuntimeError: The data bundle is corrupt'
    )


# ----------------------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------------------


def test_summarize(monkeypatch, tmp_path, capsys) -> None:
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir()
    output = tmp_path / 'results'
    runs = [_report('mlp', seed=s) for s in range(2)]
    summary = {'rows': ['mlp']}
    calls = {}
    monkeypatch.setattr(
        'tabpack_repro.reporting.summarize.collect_runs',
        lambda d: calls.setdefault('collect', d) and runs,
    )
    monkeypatch.setattr(
        'tabpack_repro.reporting.summarize.summarize',
        lambda r: calls.setdefault('summarize', r) and summary,
    )
    monkeypatch.setattr(
        'tabpack_repro.reporting.summarize.write_summary',
        lambda s, d: calls.setdefault('write', (s, d)),
    )
    monkeypatch.setattr(
        'tabpack_repro.reporting.summarize.to_markdown',
        lambda s: '| method |\n| :-- |\n| mlp |\n',
    )

    argv = ['summarize', '--runs-dir', str(runs_dir), '--output', str(output)]
    assert cli.main(argv) == 0
    assert Path(calls['collect']) == runs_dir
    assert calls['summarize'] is runs
    assert calls['write'][0] is summary
    assert Path(calls['write'][1]) == output
    out = capsys.readouterr().out
    assert out.startswith('| method |\n| :-- |\n| mlp |\n\n')
    assert 'summarized 2 run reports' in out
    assert f'output: {output}' in out


def test_summarize_errors_exit_1(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr('tabpack_repro.reporting.summarize.collect_runs', lambda d: [])
    missing = tmp_path / 'missing'
    argv = ['summarize', '--output', str(tmp_path / 'r'), '--runs-dir']
    assert cli.main([*argv, str(missing)]) == 1
    assert f'runs directory not found: {missing}' in capsys.readouterr().err

    assert cli.main([*argv, str(tmp_path)]) == 1
    assert f'no report.json found below {tmp_path}' in capsys.readouterr().err
    assert not (tmp_path / 'r').exists()


# ----------------------------------------------------------------------------------
# reference
# ----------------------------------------------------------------------------------


def test_reference_writes_the_extracted_json(monkeypatch, tmp_path, capsys) -> None:
    clone = tmp_path / 'clone'
    clone.mkdir()
    extracted = {'main': {'test_score': 0.857, 'n_models': 64}}
    seen = []
    monkeypatch.setattr(
        'tabpack_repro.reporting.reference.get_reference_dir', lambda: clone
    )
    monkeypatch.setattr(
        'tabpack_repro.reporting.reference.extract_churn_reference',
        lambda d: seen.append(Path(d)) or extracted,
    )
    output = tmp_path / 'results' / 'reference' / 'churn_official.json'

    assert cli.main(['reference', '--output', str(output)]) == 0
    assert seen == [clone]
    assert json.loads(output.read_text()) == extracted
    assert f'output: {output}' in capsys.readouterr().out

    other = tmp_path / 'other'
    other.mkdir()
    argv = ['reference', '--output', str(output), '--reference-dir', str(other)]
    assert cli.main(argv) == 0
    assert seen[-1] == other


def test_reference_without_clone_exits_1(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        'tabpack_repro.reporting.reference.get_reference_dir', lambda: None
    )
    output = tmp_path / 'ref.json'
    assert cli.main(['reference', '--output', str(output)]) == 1
    assert 'TABPACK_REFERENCE_DIR' in capsys.readouterr().err

    argv = ['reference', '--output', str(output), '--reference-dir']
    assert cli.main([*argv, str(tmp_path / 'missing')]) == 1
    assert 'reference directory not found' in capsys.readouterr().err
    assert not output.exists()


# ----------------------------------------------------------------------------------
# Real config files (a28's load_config and configs/churn/*.toml)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize('method', list(RUN_METHODS))
def test_run_real_config_files(method, monkeypatch, tmp_path, capsys) -> None:
    fakes = _install_fake_runs(monkeypatch)
    path = CONFIGS_DIR / f'{method}.toml'
    expected = config_lib.load_config(path)
    assert isinstance(expected, RUN_METHODS[method][1])
    output = tmp_path / method / 'seed-2'

    argv = ['run', '--config', str(path), '--output', str(output)]
    assert cli.main([*argv, '--seed', '2', '--device', 'cpu']) == 0
    (got, got_output), *_ = fakes[method].calls
    assert type(got) is type(expected)
    assert got == dataclasses.replace(
        expected, seed=2, training=dataclasses.replace(expected.training, device='cpu')
    )
    assert Path(got_output) == output
    assert 'seed=2' in capsys.readouterr().out

    assert cli.main(argv) == 0
    assert fakes[method].calls[-1][0] == expected


def test_run_real_conservative_config(monkeypatch, tmp_path, capsys) -> None:
    # Its source_run (runs/churn/tabpack/seed-0) is relative to the working dir.
    fakes = _install_fake_runs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    path = CONFIGS_DIR / 'tabpack-conservative.toml'
    argv = ['run', '--config', str(path), '--output', 'runs/churn/cons']

    assert cli.main(argv) == 1
    assert 'runs/churn/tabpack/seed-0/report.json' in capsys.readouterr().err

    source = tmp_path / 'runs' / 'churn' / 'tabpack' / 'seed-0'
    source.mkdir(parents=True)
    (source / 'report.json').write_text('{}')
    assert cli.main(argv) == 0
    ((config, _),) = fakes['tabpack-conservative'].calls
    assert config == ConservativeEvalConfig(
        source_run='runs/churn/tabpack/seed-0', n_seeds=5
    )
    assert capsys.readouterr().out.startswith('tabpack-conservative  n_seeds=5')


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('method = "mlp"\n[training]\npatiance = 3\n', 'patiance'),
        ('method = "mlp"\nseed = "zero"\n', 'seed'),
        ('method = "lightgbm"\n', 'lightgbm'),
        ('method = "mlp"\n[training\n', 'TOMLDecodeError'),
    ],
)
def test_run_real_invalid_config_exits_1(
    text, message, monkeypatch, tmp_path, capsys
) -> None:
    fakes = _install_fake_runs(monkeypatch)
    path = tmp_path / 'bad.toml'
    path.write_text(text)
    assert cli.main(['run', '--config', str(path), '--output', str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f'tabpack-repro run: error: invalid config file {path}')
    assert message in lines[0]
    assert not any(f.calls for f in fakes.values())


# ----------------------------------------------------------------------------------
# End to end with the real MLP method (a30) on the real Churn data
# ----------------------------------------------------------------------------------


@pytest.mark.data
def test_run_real_mlp_on_churn(churn_dir, tmp_path, capsys) -> None:
    path = tmp_path / 'mlp-tiny.toml'
    path.write_text(
        'method = "mlp"\n'
        f'[data]\npath = "{churn_dir.as_posix()}"\n'
        '[model]\nn_blocks = 1\nd_block = 16\n'
        '[training]\nmax_epochs = 1\n'
    )
    output = tmp_path / 'runs' / 'mlp' / 'seed-1'
    argv = ['run', '--config', str(path), '--output', str(output)]
    assert cli.main([*argv, '--seed', '1', '--device', 'cpu']) == 0

    report = json.loads((output / 'report.json').read_text())
    assert report['method'] == 'mlp'
    assert report['seed'] == report['config']['seed'] == 1
    assert report['config']['training']['device'] == 'cpu'
    assert report['config']['model']['d_block'] == 16
    assert (output / 'predictions.npz').is_file()
    assert (output / 'config.toml').is_file()
    result, out_line = capsys.readouterr().out.splitlines()
    test_score = report['metrics']['test']['score']
    assert result.startswith('mlp  seed=1  val_score=')
    assert f'test_score={test_score:.5f}' in result
    assert out_line == f'output: {output}'
