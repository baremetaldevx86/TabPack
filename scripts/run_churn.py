"""Run the whole Churn experiment with one command (a35).

Steps, in order (each one is a function below, so tests can replace it):

1. ``download``: make sure ``data/churn`` exists (``data.download_dataset``);
2. for every seed (default 0..4) and method (mlp, homogeneous, tabpack): run
   ``configs/churn/<method>.toml`` with the seed overridden into
   ``<runs-dir>/<method>/seed-<s>/``;
3. ``tabpack-conservative``: the paper's conservative protocol for the ensemble
   selected by ``<runs-dir>/tabpack/seed-0``
   (``configs/churn/tabpack-conservative.toml``) into
   ``<runs-dir>/tabpack-conservative/``. Its seeds are 0..n_seeds-1 and are
   handled inside ``methods.conservative.run`` (``--seeds`` does not apply to it);
4. ``summarize``: ``reporting.summarize`` over ``<runs-dir>`` into ``<results-dir>``;
5. ``plots``: method comparison + online-ensemble history and member scores of
   TabPack seed 0 into ``<results-dir>/figures/``;
6. ``reference``: refresh ``results/reference/churn_official.json`` from the official
   clone when it is available, otherwise load the saved file.

Runs whose ``report.json`` already exists are skipped (resume) unless ``--force``.
A failing step is logged and the remaining steps still run; the script exits with 1
if any step failed. Default paths are relative to the repository root.

Usage: ``bash scripts/run_churn.sh [ARGS]`` or ``uv run python scripts/run_churn.py
[ARGS]``; ``--dry-run`` prints the plan without running anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import logging
import os
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET = 'churn'
RUN_METHODS = ('mlp', 'homogeneous', 'tabpack')
CONSERVATIVE = 'tabpack-conservative'
ALL_METHODS = (*RUN_METHODS, CONSERVATIVE)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# The TabPack run whose ensemble the conservative protocol retrains, and whose
# online-ensemble history / member scores are plotted.
SOURCE_SEED = 0
REPORT = 'report.json'
# methods.<module>.run for every method name.
METHOD_MODULES = {
    'mlp': 'mlp',
    'homogeneous': 'homogeneous',
    'tabpack': 'tabpack',
    CONSERVATIVE: 'conservative',
}
FIGURES = {
    'comparison': 'method_comparison.png',
    'history': f'tabpack_seed{SOURCE_SEED}_online_ensemble.png',
    'members': f'tabpack_seed{SOURCE_SEED}_member_scores.png',
}

logger = logging.getLogger('run_churn')


class StepSkipped(Exception):
    """Raised by a step function when there is nothing to do (reason = message)."""


@dataclass(frozen=True)
class Step:
    kind: str  # download | run | conservative | summarize | plots | reference
    name: str
    method: str | None = None
    seed: int | None = None
    output: Path | None = None


@dataclass
class StepResult:
    step: Step
    status: str  # ok | skipped | failed | interrupted
    seconds: float = 0.0
    detail: str = ''


@dataclass
class Context:
    """State shared between the steps of one invocation."""

    args: argparse.Namespace
    summary: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    results: list[StepResult] = field(default_factory=list)


# ----------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------


def _default_path(relative: str) -> Path:
    """`relative` (to the repository root) expressed relative to the cwd if possible,
    so that paths stored in reports stay short when run from the repository root."""
    path = REPO_ROOT / relative
    try:
        return Path(os.path.relpath(path))
    except ValueError:  # e.g. a different drive on Windows
        return path


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid seed: {text!r}') from None
    if value < 0:
        raise argparse.ArgumentTypeError(f'seeds must be >= 0, got {value}')
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid integer: {text!r}') from None
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='run_churn.py',
        description=(
            'Run the whole Churn experiment: MLP, homogeneous ensemble and TabPack '
            'for every seed, the conservative TabPack evaluation, then summary '
            'tables, figures and the official reference numbers.'
        ),
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=_seed,
        default=list(DEFAULT_SEEDS),
        metavar='S',
        help='seeds of the mlp/homogeneous/tabpack runs (default: 0 1 2 3 4)',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=ALL_METHODS,
        default=list(ALL_METHODS),
        metavar='M',
        help=f'subset of {", ".join(ALL_METHODS)} (default: all)',
    )
    parser.add_argument('--runs-dir', type=Path, default=_default_path('runs/churn'))
    parser.add_argument(
        '--results-dir', type=Path, default=_default_path('results/churn')
    )
    parser.add_argument(
        '--configs-dir', type=Path, default=_default_path('configs/churn')
    )
    parser.add_argument(
        '--reference-file',
        type=Path,
        default=_default_path('results/reference/churn_official.json'),
        help='saved official Churn numbers (refreshed when the clone is available)',
    )
    parser.add_argument(
        '--device',
        default=None,
        help=(
            'override training.device of the mlp/homogeneous/tabpack configs, e.g. '
            "'cpu' or 'cuda' (the conservative runs reuse the source run's config)"
        ),
    )
    parser.add_argument(
        '--conservative-seeds',
        type=_positive_int,
        default=None,
        metavar='N',
        help='n_seeds of the conservative evaluation (default: from its config)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='rerun runs whose report.json exists (default: skip them)',
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='print the plan and exit'
    )
    parser.add_argument(
        '--skip-download', action='store_true', help='do not check/download the data'
    )
    args = parser.parse_args(argv)
    args.seeds = list(dict.fromkeys(args.seeds))
    args.methods = [m for m in ALL_METHODS if m in args.methods]
    return args


# ----------------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------------


def run_dir(runs_dir: Path, method: str, seed: int) -> Path:
    return Path(runs_dir) / method / f'seed-{seed}'


def config_path(args: argparse.Namespace, method: str) -> Path:
    return Path(args.configs_dir) / f'{method}.toml'


def source_run_dir(args: argparse.Namespace) -> Path:
    return run_dir(args.runs_dir, 'tabpack', SOURCE_SEED)


def build_plan(args: argparse.Namespace) -> list[Step]:
    plan: list[Step] = []
    if not args.skip_download:
        plan.append(Step('download', 'download'))
    for seed in args.seeds:
        for method in RUN_METHODS:
            if method in args.methods:
                output = run_dir(args.runs_dir, method, seed)
                plan.append(Step('run', f'{method}/seed-{seed}', method, seed, output))
    if CONSERVATIVE in args.methods:
        output = Path(args.runs_dir) / CONSERVATIVE
        plan.append(Step('conservative', CONSERVATIVE, CONSERVATIVE, None, output))
    plan.append(Step('summarize', 'summarize', output=Path(args.results_dir)))
    plan.append(Step('plots', 'plots', output=Path(args.results_dir) / 'figures'))
    plan.append(Step('reference', 'reference', output=Path(args.reference_file)))
    return plan


def is_complete(step: Step) -> bool:
    """A run/conservative step is complete when its report.json exists."""
    if step.kind not in ('run', 'conservative') or step.output is None:
        return False
    return (step.output / REPORT).is_file()


def describe_step(step: Step, args: argparse.Namespace) -> str:
    if step.kind == 'download':
        return f'ensure the {DATASET!r} dataset is downloaded'
    if step.kind == 'run':
        device = f', device={args.device}' if args.device else ''
        return (
            f'{config_path(args, step.method)} seed={step.seed}{device} '
            f'-> {step.output}'
        )
    if step.kind == 'conservative':
        n_seeds = args.conservative_seeds or 'from config'
        return (
            f'{config_path(args, CONSERVATIVE)} source_run={source_run_dir(args)} '
            f'n_seeds={n_seeds} -> {step.output}'
        )
    if step.kind == 'summarize':
        return f'{args.runs_dir} -> {step.output}'
    if step.kind == 'plots':
        return f'-> {step.output}/{{{",".join(FIGURES.values())}}}'
    if step.kind == 'reference':
        return f'refresh from the official clone if present, else load {step.output}'
    raise ValueError(f'Unknown step kind: {step.kind!r}')


def print_plan(plan: list[Step], args: argparse.Namespace) -> int:
    """Print the plan (dry run) and check the configs; return the exit code."""
    width = max(len(step.name) for step in plan)
    print(
        f'Plan ({len(plan)} steps; dry run, nothing is executed; '
        f'skip = {REPORT} exists):',
        flush=True,
    )
    for i, step in enumerate(plan, 1):
        if is_complete(step) and not args.force:
            action = 'skip'
        elif is_complete(step):
            action = 'rerun (--force)'
        else:
            action = 'run'
        print(
            f'  [{i:>2}/{len(plan)}] {step.name:<{width}}  {action:<15} '
            f'{describe_step(step, args)}',
            flush=True,
        )
    n_errors = 0
    methods = [m for m in args.methods if any(s.method == m for s in plan)]
    if methods:
        print('Configs:', flush=True)
    for method in methods:
        path = config_path(args, method)
        try:
            config = load_method_config(path)
            if getattr(config, 'method', None) != method:
                raise ValueError(f'method is {config.method!r}, expected {method!r}')
        except Exception as exc:  # noqa: BLE001 (report any config error)
            n_errors += 1
            print(f'  {path}: ERROR {_describe_error(exc)}', flush=True)
        else:
            print(f'  {path}: ok', flush=True)
    if CONSERVATIVE in args.methods and not (source_run_dir(args) / REPORT).is_file():
        planned = 'tabpack' in args.methods and SOURCE_SEED in args.seeds
        note = 'produced by this plan' if planned else 'NOT produced by this plan'
        print(
            f'Note: {CONSERVATIVE} needs {source_run_dir(args) / REPORT} ({note}).',
            flush=True,
        )
    return 1 if n_errors else 0


# ----------------------------------------------------------------------------------
# Step functions. Each takes (step, ctx) and returns a short detail string for the
# summary table; it raises StepSkipped when there is nothing to do.
# ----------------------------------------------------------------------------------


def load_method_config(path: Path) -> Any:
    from tabpack_repro.config import load_config

    return load_config(path)


def get_method_runner(method: str) -> Any:
    """``tabpack_repro.methods.<module>.run`` for a method name."""
    module = importlib.import_module(f'tabpack_repro.methods.{METHOD_MODULES[method]}')
    return module.run


def prepare_run_config(
    config: Any, *, method: str, seed: int, device: str | None
) -> Any:
    """A copy of `config` with the seed (and optionally the device) overridden."""
    if getattr(config, 'method', None) != method:
        raise ValueError(
            f'The config has method {getattr(config, "method", None)!r}, '
            f'expected {method!r}'
        )
    config = dataclasses.replace(config, seed=seed)
    if device is not None:
        training = dataclasses.replace(config.training, device=device)
        config = dataclasses.replace(config, training=training)
    return config


def prepare_conservative_config(
    config: Any, *, source_run: Path, n_seeds: int | None
) -> Any:
    if getattr(config, 'method', None) != CONSERVATIVE:
        raise ValueError(
            f'The config has method {getattr(config, "method", None)!r}, '
            f'expected {CONSERVATIVE!r}'
        )
    changes: dict[str, Any] = {'source_run': str(source_run)}
    if n_seeds is not None:
        changes['n_seeds'] = n_seeds
    return dataclasses.replace(config, **changes)


def step_download(step: Step, ctx: Context) -> str:
    from tabpack_repro.data.download import download_dataset

    return str(download_dataset(DATASET))


def step_run(step: Step, ctx: Context) -> str:
    args = ctx.args
    assert step.method is not None and step.seed is not None and step.output
    config = load_method_config(config_path(args, step.method))
    config = prepare_run_config(
        config, method=step.method, seed=step.seed, device=args.device
    )
    if args.force:
        _remove_reports([step.output / REPORT])
    report = get_method_runner(step.method)(config, step.output)
    return format_scores(report)


def step_conservative(step: Step, ctx: Context) -> str:
    args = ctx.args
    assert step.output is not None
    source = source_run_dir(args)
    if not (source / REPORT).is_file():
        raise FileNotFoundError(
            f'{source / REPORT} does not exist: the conservative evaluation needs a '
            f'finished TabPack seed-{SOURCE_SEED} run'
        )
    config = load_method_config(config_path(args, CONSERVATIVE))
    config = prepare_conservative_config(
        config, source_run=source, n_seeds=args.conservative_seeds
    )
    if args.device is not None:
        logger.info(
            "--device does not apply to %s: it reuses the source run's config",
            CONSERVATIVE,
        )
    if args.force:
        _remove_reports(
            [step.output / REPORT]
            + [step.output / f'seed-{s}' / REPORT for s in range(config.n_seeds)]
        )
    report = get_method_runner(CONSERVATIVE)(config, step.output)
    return format_scores(report)


def step_summarize(step: Step, ctx: Context) -> str:
    from tabpack_repro.reporting.summarize import collect_runs, summarize, write_summary

    runs = collect_runs(ctx.args.runs_dir)
    if not runs:
        raise StepSkipped(f'no {REPORT} under {ctx.args.runs_dir}')
    ctx.summary = summarize(runs)
    assert step.output is not None
    write_summary(ctx.summary, step.output)
    return f'{len(runs)} reports -> {step.output}'


def step_plots(step: Step, ctx: Context) -> str:
    from tabpack_repro.reporting import plots
    from tabpack_repro.utils.io import load_json

    out = step.output
    assert out is not None
    made: list[str] = []
    missing: list[str] = []
    if ctx.summary is not None:
        out.mkdir(parents=True, exist_ok=True)
        plots.plot_method_comparison(ctx.summary, out / FIGURES['comparison'])
        made.append(FIGURES['comparison'])
    else:
        missing.append('summary')
    report_path = source_run_dir(ctx.args) / REPORT
    if report_path.is_file():
        report = load_json(report_path)
        out.mkdir(parents=True, exist_ok=True)
        plots.plot_online_ensemble_history(report, out / FIGURES['history'])
        plots.plot_member_scores(report, out / FIGURES['members'])
        made += [FIGURES['history'], FIGURES['members']]
    else:
        missing.append(str(report_path))
    if not made:
        raise StepSkipped(f'nothing to plot (missing: {", ".join(missing)})')
    if missing:
        logger.warning('plots: missing %s', ', '.join(missing))
    return f'{len(made)} figures -> {out}'


def step_reference(step: Step, ctx: Context) -> str:
    from tabpack_repro.reporting.reference import (
        extract_churn_reference,
        get_reference_dir,
        load_saved_reference,
    )
    from tabpack_repro.utils.io import dump_json

    path = step.output
    assert path is not None
    reference_dir = get_reference_dir()
    if reference_dir is not None:
        try:
            data = extract_churn_reference(reference_dir)
        except Exception as exc:  # noqa: BLE001 (fall back to the saved file)
            logger.warning(
                'Could not read the official clone at %s (%s: %s); using %s',
                reference_dir,
                type(exc).__name__,
                exc,
                path,
            )
        else:
            dump_json(path, data)
            ctx.reference = data
            return f'refreshed {path} from {reference_dir}'
    ctx.reference = load_saved_reference(path)
    return f'loaded {path}'


def run_step(step: Step, ctx: Context) -> str:
    """Dispatch to the step function of `step.kind` (looked up at call time, so
    tests can monkeypatch the module attributes)."""
    if step.kind == 'download':
        return step_download(step, ctx)
    if step.kind == 'run':
        return step_run(step, ctx)
    if step.kind == 'conservative':
        return step_conservative(step, ctx)
    if step.kind == 'summarize':
        return step_summarize(step, ctx)
    if step.kind == 'plots':
        return step_plots(step, ctx)
    if step.kind == 'reference':
        return step_reference(step, ctx)
    raise ValueError(f'Unknown step kind: {step.kind!r}')


# ----------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------


def _describe_error(exc: BaseException) -> str:
    text = str(exc)
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def _remove_reports(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()
            logger.info('--force: removed %s', path)


def _free_memory() -> None:
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get('score')
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def report_scores(report: Any) -> tuple[float | None, float | None]:
    """(val, test) score of a run report, or the mean scores of the conservative
    aggregate report; None where unavailable."""
    if not isinstance(report, dict):
        return None, None
    for key in ('metrics', 'mean'):
        part = report.get(key)
        if isinstance(part, dict):
            val, test = _number(part.get('val')), _number(part.get('test'))
            if val is not None or test is not None:
                return val, test
    return None, None


def format_scores(report: Any) -> str:
    val, test = report_scores(report)
    parts = []
    if val is not None:
        parts.append(f'val {val:.4f}')
    if test is not None:
        parts.append(f'test {test:.4f}')
    return ', '.join(parts)


def _read_scores(path: Path) -> str:
    try:
        from tabpack_repro.utils.io import load_json

        return format_scores(load_json(path))
    except (OSError, ValueError):
        return ''


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f'{minutes}m{secs:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m{secs:02d}s'


def execute_plan(plan: list[Step], ctx: Context) -> list[StepResult]:
    """Run the steps in order; a failing step does not stop the others. Stops (and
    marks the current step 'interrupted') on KeyboardInterrupt."""
    results = ctx.results
    n = len(plan)
    for i, step in enumerate(plan, 1):
        prefix = f'[{i}/{n}] {step.name}'
        if is_complete(step) and not ctx.args.force:
            assert step.output is not None
            detail = _read_scores(step.output / REPORT)
            logger.info('%s: skipped, %s exists', prefix, step.output / REPORT)
            results.append(StepResult(step, 'skipped', 0.0, detail or 'done before'))
            continue
        logger.info('%s: started (%s)', prefix, describe_step(step, ctx.args))
        start = time.perf_counter()
        try:
            detail = run_step(step, ctx) or ''
        except StepSkipped as exc:
            elapsed = time.perf_counter() - start
            logger.warning('%s: skipped: %s', prefix, exc)
            results.append(StepResult(step, 'skipped', elapsed, str(exc)))
        except KeyboardInterrupt:
            elapsed = time.perf_counter() - start
            logger.error('%s: interrupted after %s', prefix, format_duration(elapsed))
            results.append(StepResult(step, 'interrupted', elapsed, 'Ctrl-C'))
            break
        except Exception as exc:  # noqa: BLE001 (log it, continue with the rest)
            elapsed = time.perf_counter() - start
            message = _describe_error(exc)
            logger.error(
                '%s: FAILED after %s: %s\n%s',
                prefix,
                format_duration(elapsed),
                message,
                traceback.format_exc().rstrip(),
            )
            results.append(StepResult(step, 'failed', elapsed, message))
        else:
            elapsed = time.perf_counter() - start
            logger.info(
                '%s: done in %s%s',
                prefix,
                format_duration(elapsed),
                f' ({detail})' if detail else '',
            )
            results.append(StepResult(step, 'ok', elapsed, detail))
        finally:
            _free_memory()
    return results


def format_results_table(results: list[StepResult]) -> str:
    rows = [('step', 'status', 'time', 'detail')]
    rows += [
        (r.step.name, r.status, format_duration(r.seconds), r.detail) for r in results
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for row in rows:
        cells = [row[i].ljust(widths[i]) for i in range(3)]
        lines.append('  '.join([*cells, row[3]]).rstrip())
    lines.insert(1, '  '.join('-' * w for w in [*widths, 6]))
    return '\n'.join(lines)


def _mean_std(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    mean, std = _number(entry.get('mean')), _number(entry.get('std'))
    if mean is None:
        return None
    return f'{100 * mean:.2f}' + (f' ± {100 * std:.2f}' if std is not None else '')


def format_reference(data: Any) -> list[str]:
    """A few lines with the official numbers of results/reference/churn_official.json
    (the TabPack headline and the paper's Table 14 MLP/TabPack rows); tolerant of a
    missing or different structure."""
    if not isinstance(data, dict):
        return []
    lines = []
    headline = data.get('headline')
    if isinstance(headline, dict) and headline.get('text'):
        details = []
        if headline.get('protocol'):
            details.append(f'{headline["protocol"]} protocol')
        if headline.get('n_seeds'):
            details.append(f'{headline["n_seeds"]} seeds')
        suffix = f' ({", ".join(details)})' if details else ''
        lines.append(f'official TabPack: {headline["text"]}{suffix}')
    table = data.get('paper_table14')
    table = table.get(DATASET) if isinstance(table, dict) else None
    if isinstance(table, dict):
        for name in ('MLP', 'TabPack'):
            text = _mean_std(table.get(name))
            if text:
                lines.append(f'paper Table 14, {name}: {text}')
    return lines


def print_summary(ctx: Context, plan: list[Step]) -> None:
    results = ctx.results
    counts = {s: sum(r.status == s for r in results) for s in ('ok', 'skipped')}
    failed = [r for r in results if r.status in ('failed', 'interrupted')]
    not_run = len(plan) - len(results)
    total = sum(r.seconds for r in results)
    print('\n==== run_churn: summary ====', flush=True)
    print(format_results_table(results), flush=True)
    print(
        f'\n{counts["ok"]} ok, {counts["skipped"]} skipped, {len(failed)} failed'
        + (f', {not_run} not run' if not_run else '')
        + f'; total time {format_duration(total)}',
        flush=True,
    )
    if ctx.summary is not None:
        try:
            from tabpack_repro.reporting.summarize import to_markdown

            print('\n' + to_markdown(ctx.summary).rstrip(), flush=True)
        except Exception as exc:  # noqa: BLE001 (the table is a convenience)
            logger.warning('Could not format the summary table: %s', exc)
    reference = format_reference(ctx.reference)
    if reference:
        print('\nOfficial Churn numbers (test accuracy):', flush=True)
        for line in reference:
            print(f'  {line}', flush=True)
    if failed:
        print('\nFailures:', flush=True)
        for r in failed:
            print(f'  {r.step.name}: {r.detail}', flush=True)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr,
    )


def _log_header(args: argparse.Namespace) -> None:
    from tabpack_repro.utils.io import git_commit, git_is_dirty

    commit, dirty = git_commit(), git_is_dirty()
    logger.info(
        'methods=%s seeds=%s device=%s force=%s',
        ','.join(args.methods),
        ','.join(map(str, args.seeds)),
        args.device or 'from config',
        args.force,
    )
    logger.info(
        'runs=%s results=%s configs=%s git=%s%s',
        args.runs_dir,
        args.results_dir,
        args.configs_dir,
        commit,
        ' (dirty)' if dirty else '',
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        return print_plan(plan, args)
    _setup_logging()
    _log_header(args)
    ctx = Context(args)
    execute_plan(plan, ctx)
    print_summary(ctx, plan)
    if any(r.status == 'interrupted' for r in ctx.results):
        return 130
    return 1 if any(r.status == 'failed' for r in ctx.results) else 0


if __name__ == '__main__':
    sys.exit(main())
