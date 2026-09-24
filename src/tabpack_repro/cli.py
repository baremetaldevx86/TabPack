"""Command-line interface (a34): ``tabpack-repro`` / ``python -m tabpack_repro``.

Subcommands:
  download [--name churn] [--data-dir DIR] [--force]
  run --config configs/churn/<method>.toml [--seed S] --output DIR [--device D]
  conservative --source-run DIR [--n-seeds 5] --output DIR
  summarize --runs-dir runs/churn --output results/churn
  reference --output results/reference/churn_official.json
Uses argparse only. Exit code 0 on success, 2 on usage errors.

Exit codes: 0 success, 1 runtime error (bad config file, failed download, missing
source run, ...), 2 usage error (argparse errors and options that do not apply to
the chosen config), 130 interrupted. Results go to stdout, errors to stderr.

Everything heavy (torch, the methods, reporting) is imported inside the subcommand
handlers, so ``--help`` stays fast. The handlers look up the functions they call at
call time (``module.function``), so tests can monkeypatch those module attributes.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import math
import re
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PROG = 'tabpack-repro'

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_USAGE = 2
_EXIT_INTERRUPTED = 130

# config.method -> module whose `run(config, output_dir)` executes it.
_METHOD_MODULES: dict[str, str] = {
    'mlp': 'tabpack_repro.methods.mlp',
    'homogeneous': 'tabpack_repro.methods.homogeneous',
    'tabpack': 'tabpack_repro.methods.tabpack',
    'tabpack-conservative': 'tabpack_repro.methods.conservative',
}

_DEVICE_RE = re.compile(r'auto|cpu|cuda(:\d+)?|mps')
_MAX_SEED = 2**32  # numpy's global seeding accepts [0, 2**32).


class _CLIError(Exception):
    """A runtime error with a user-facing message (exit code 1, no traceback)."""


# ----------------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------------


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'invalid seed {text!r}: not an integer'
        ) from None
    if not 0 <= value < _MAX_SEED:
        raise argparse.ArgumentTypeError(
            f'invalid seed {value}: must be in [0, {_MAX_SEED})'
        )
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'invalid value {text!r}: not an integer'
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(f'invalid value {value}: must be >= 1')
    return value


def _device(text: str) -> str:
    if _DEVICE_RE.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(
            f'invalid device {text!r}: expected auto, cpu, cuda, cuda:<index> or mps'
        )
    return text


def _build_parser() -> argparse.ArgumentParser:
    """The argument parser (no heavy imports; used by main and by the tests)."""
    common = argparse.ArgumentParser(add_help=False)
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        help='debug logging, and full tracebacks on errors',
    )
    verbosity.add_argument(
        '-q',
        '--quiet',
        action='store_const',
        dest='log_level',
        const=logging.WARNING,
        help='only log warnings and errors',
    )
    common.set_defaults(log_level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            'From-scratch reproduction of TabPack (ICML 2026) on Churn: ordinary MLP '
            'vs homogeneous MLP ensemble vs reduced heterogeneous TabPack.'
        ),
        epilog=f'Run "{_PROG} COMMAND --help" for the options of a command.',
    )
    commands = parser.add_subparsers(
        dest='command', metavar='COMMAND', required=True, title='commands'
    )

    def add(name: str, handler: Callable[[argparse.Namespace], int], help_: str):
        sub = commands.add_parser(
            name, parents=[common], help=help_, description=help_ + '.'
        )
        sub.set_defaults(handler=handler, subparser=sub)
        return sub

    p = add(
        'download',
        _cmd_download,
        'Download the official data bundle and extract one dataset',
    )
    p.add_argument(
        '--name',
        default='churn',
        metavar='NAME',
        help='dataset directory name (default: churn)',
    )
    p.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help='data root (default: $TABPACK_DATA_DIR, else <repo>/data)',
    )
    p.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help='where the bundle is cached (default: <data-dir>/.cache)',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='re-extract the dataset even if it is already complete',
    )

    p = add('run', _cmd_run, 'Run one method from a TOML config file')
    p.add_argument(
        '--config',
        type=Path,
        required=True,
        metavar='PATH',
        help='TOML config, e.g. configs/churn/tabpack.toml',
    )
    p.add_argument(
        '--output',
        type=Path,
        required=True,
        metavar='DIR',
        help='run directory (report.json, predictions.npz, config.toml)',
    )
    p.add_argument(
        '--seed',
        type=_seed,
        default=None,
        metavar='S',
        help='override the seed of the config',
    )
    p.add_argument(
        '--device',
        type=_device,
        default=None,
        metavar='D',
        help='override training.device: auto, cpu, cuda or cuda:<index>',
    )

    p = add(
        'conservative',
        _cmd_conservative,
        'Retrain the ensemble selected by a TabPack run with fresh seeds '
        '(conservative protocol)',
    )
    p.add_argument(
        '--source-run',
        type=Path,
        required=True,
        metavar='DIR',
        help='directory of a finished TabPack run (contains report.json)',
    )
    p.add_argument(
        '--n-seeds',
        type=_positive_int,
        default=5,
        metavar='N',
        help='number of retraining seeds 0..N-1 (default: 5)',
    )
    p.add_argument(
        '--output',
        type=Path,
        required=True,
        metavar='DIR',
        help='output directory (seed-<s>/ subdirectories and an aggregate report)',
    )

    p = add(
        'summarize',
        _cmd_summarize,
        'Aggregate every report.json below a runs directory over seeds',
    )
    p.add_argument(
        '--runs-dir',
        type=Path,
        required=True,
        metavar='DIR',
        help='e.g. runs/churn (searched recursively)',
    )
    p.add_argument(
        '--output',
        type=Path,
        required=True,
        metavar='DIR',
        help='directory for summary.json, summary.md and summary.csv',
    )

    p = add(
        'reference',
        _cmd_reference,
        'Extract the official Churn results from the official TabPack clone',
    )
    p.add_argument(
        '--output',
        type=Path,
        required=True,
        metavar='PATH',
        help='JSON file to write, e.g. results/reference/churn_official.json',
    )
    p.add_argument(
        '--reference-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help=(
            'official TabPack clone '
            '(default: $TABPACK_REFERENCE_DIR, else <repo>/.reference/tabpack)'
        ),
    )
    return parser


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def _out(line: str = '') -> None:
    print(line, file=sys.stdout, flush=True)


def _configure_logging(level: int) -> None:
    # No-op when the root logger already has handlers (an embedding application,
    # pytest); the package level is always applied.
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )
    logging.getLogger('tabpack_repro').setLevel(level)


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, _CLIError):
        return str(exc)
    if isinstance(exc, OSError) and exc.strerror and exc.filename is not None:
        return f'{exc.strerror}: {exc.filename}'
    message = str(exc)
    return f'{type(exc).__name__}: {message}' if message else type(exc).__name__


def _fmt_score(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 'n/a'
    return f'{value:.5f}' if math.isfinite(value) else 'n/a'


def _fmt_time(seconds: Any) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        return 'n/a'
    return f'{seconds:.1f}s'


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _load_config(path: Path) -> Any:
    from tabpack_repro import config as config_lib

    if not path.is_file():
        raise _CLIError(f'config file not found: {path}')
    try:
        return config_lib.load_config(path)
    except Exception as exc:
        raise _CLIError(f'invalid config file {path}: {_describe_error(exc)}') from exc


def _method_runner(method: str) -> Callable[[Any, Path], dict[str, Any]]:
    try:
        module_name = _METHOD_MODULES[method]
    except KeyError:
        raise _CLIError(
            f'unknown method {method!r} (expected one of: {", ".join(_METHOD_MODULES)})'
        ) from None
    return importlib.import_module(module_name).run


def _print_run_result(report: Any, *, fallback_time: float) -> None:
    method = _get(report, 'method') or '?'
    seed = _get(report, 'seed')
    elapsed = _get(report, 'time_sec')
    _out(
        f'{method}  seed={"n/a" if seed is None else seed}  '
        f'val_score={_fmt_score(_get(report, "metrics", "val", "score"))}  '
        f'test_score={_fmt_score(_get(report, "metrics", "test", "score"))}  '
        f'time={_fmt_time(fallback_time if elapsed is None else elapsed)}'
    )


def _mean_std(values: Any) -> str:
    if not isinstance(values, list):
        return 'n/a'
    finite = [
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ]
    if not finite:
        return 'n/a'
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return f'{mean:.5f}±{std:.5f}'


def _print_conservative_result(report: Any, *, elapsed: float) -> None:
    seeds = _get(report, 'seeds')
    n_seeds = len(seeds) if isinstance(seeds, list) else _get(report, 'n_seeds')
    _out(
        f'{_get(report, "method") or "tabpack-conservative"}  '
        f'n_seeds={"n/a" if n_seeds is None else n_seeds}  '
        f'val_score={_mean_std(_get(report, "scores", "val"))}  '
        f'test_score={_mean_std(_get(report, "scores", "test"))}  '
        f'time={_fmt_time(elapsed)}'
    )


def _execute(config: Any, output: Path) -> int:
    """Run `config` with its method's run(), print the result and output path."""
    method = getattr(config, 'method', None)
    run = _method_runner(method)
    if method == 'tabpack-conservative':
        source_report = Path(config.source_run) / 'report.json'
        if not source_report.is_file():
            raise _CLIError(
                f'source run report not found: {source_report} (run TabPack first: '
                f'{_PROG} run --config configs/churn/tabpack.toml --seed 0 '
                f'--output {config.source_run})'
            )

    start = time.perf_counter()
    report = run(config, output)
    elapsed = time.perf_counter() - start
    if method == 'tabpack-conservative':
        _print_conservative_result(report, elapsed=elapsed)
    else:
        _print_run_result(report, fallback_time=elapsed)
    _out(f'output: {output}')
    return _EXIT_OK


# ----------------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------------


def _cmd_download(args: argparse.Namespace) -> int:
    from tabpack_repro.data import download

    path = download.download_dataset(
        args.name, data_dir=args.data_dir, cache_dir=args.cache_dir, force=args.force
    )
    _out(f'dataset {args.name!r} is ready at {path}')
    return _EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    method = getattr(config, 'method', None)
    if method == 'tabpack-conservative':
        # ConservativeEvalConfig has no seed and no training section: the seeds are
        # 0..n_seeds-1 and the settings come from the source run.
        for flag, value in (('--seed', args.seed), ('--device', args.device)):
            if value is not None:
                args.subparser.error(
                    f'{flag} does not apply to a tabpack-conservative config'
                )
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    if args.device is not None:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, device=args.device)
        )
    return _execute(config, args.output)


def _cmd_conservative(args: argparse.Namespace) -> int:
    from tabpack_repro import config as config_lib

    config = config_lib.ConservativeEvalConfig(
        source_run=str(args.source_run), n_seeds=args.n_seeds
    )
    return _execute(config, args.output)


def _cmd_summarize(args: argparse.Namespace) -> int:
    from tabpack_repro.reporting import summarize as summarize_lib

    if not args.runs_dir.is_dir():
        raise _CLIError(f'runs directory not found: {args.runs_dir}')
    runs = summarize_lib.collect_runs(args.runs_dir)
    if not runs:
        raise _CLIError(f'no report.json found below {args.runs_dir}')
    summary = summarize_lib.summarize(runs)
    summarize_lib.write_summary(summary, args.output)
    _out(summarize_lib.to_markdown(summary).rstrip('\n'))
    _out()
    _out(f'summarized {len(runs)} run reports')
    _out(f'output: {args.output} (summary.json, summary.md, summary.csv)')
    return _EXIT_OK


def _cmd_reference(args: argparse.Namespace) -> int:
    from tabpack_repro.reporting import reference as reference_lib
    from tabpack_repro.utils import io as io_lib

    reference_dir = args.reference_dir
    if reference_dir is None:
        reference_dir = reference_lib.get_reference_dir()
        if reference_dir is None:
            raise _CLIError(
                'the official TabPack clone was not found: set $TABPACK_REFERENCE_DIR '
                'or pass --reference-dir (`make reference` clones it into '
                '.reference/tabpack)'
            )
    elif not reference_dir.is_dir():
        raise _CLIError(f'reference directory not found: {reference_dir}')
    data = reference_lib.extract_churn_reference(reference_dir)
    io_lib.dump_json(args.output, data)
    _out(f'extracted the official Churn results from {reference_dir}')
    _out(f'output: {args.output}')
    return _EXIT_OK


# ----------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------


def _exit_code(code: object) -> int:
    # SystemExit.code: None -> 0, int -> itself, anything else (a message) -> 1.
    if code is None:
        return _EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return _EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (default: sys.argv[1:]), run the subcommand, return the exit
    code. Never raises SystemExit itself (``--help`` returns 0, usage errors 2)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return _exit_code(exc.code)

    _configure_logging(args.log_level)
    prefix = f'{_PROG} {args.command}: error:'
    try:
        return args.handler(args)
    except SystemExit as exc:  # subparser.error() for usage errors found late
        return _exit_code(exc.code)
    except KeyboardInterrupt:
        print(f'{_PROG} {args.command}: interrupted', file=sys.stderr)
        return _EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 (top level: report, exit 1)
        if args.log_level <= logging.DEBUG:
            traceback.print_exc(file=sys.stderr)
        print(f'{prefix} {_describe_error(exc)}', file=sys.stderr)
        return _EXIT_ERROR
