"""Figures (a37). Matplotlib, Agg backend, PNG at 150 dpi; never plt.show().

Every figure is built on a bare ``matplotlib.figure.Figure`` rendered through
``FigureCanvasAgg``: no pyplot, no global figure registry, no backend switching, so
the functions are safe in tests, worker processes and headless servers.

Visual conventions shared by all figures:

* Scores are the raw fractions of the run reports and summaries (accuracy on
  Churn); they are shown in percent.
* Methods always appear in the canonical order ``mlp, homogeneous, tabpack,
  tabpack-conservative`` (unknown methods after, sorted), each with a fixed
  Okabe-Ito color, so a method keeps its color across figures.
* Identity is never carried by color alone: methods are named on the axis, series
  have a legend, and selected members differ from the others by fill and size.
* Missing optional fields (no ensemble scores in the history, members without a
  config, a single seed, ...) degrade to an explanatory note instead of an error.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

_DPI = 150

# Canonical method order and Okabe-Ito colors (validated colorblind-safe as a set:
# all-pairs CVD delta E >= 11 in OKLab x100).
_METHOD_ORDER: tuple[str, ...] = (
    'mlp',
    'homogeneous',
    'tabpack',
    'tabpack-conservative',
)
_METHOD_COLORS: dict[str, str] = {
    'mlp': '#E69F00',  # orange
    'homogeneous': '#0072B2',  # blue
    'tabpack': '#009E73',  # bluish green
    'tabpack-conservative': '#D55E00',  # vermillion
}
_EXTRA_COLORS: tuple[str, ...] = ('#CC79A7', '#56B4E9', '#000000')
_FALLBACK_COLOR = '#7F7F7F'
_METHOD_NAMES: dict[str, str] = {
    'mlp': 'MLP',
    'homogeneous': 'MLP ensemble (homogeneous)',
    'tabpack': 'TabPack (single run)',
    'tabpack-conservative': 'TabPack (conservative)',
}

# Series colors of the history figure (distinct from each other within a panel).
_VAL_COLOR = '#0072B2'
_TEST_COLOR = '#D55E00'
_RUNNING_COLOR = '#009E73'
_FINISHED_COLOR = '#E69F00'
_SELECTED_COLOR = _METHOD_COLORS['tabpack']

# Neutral ink and chrome.
_SURFACE = '#FFFFFF'
_INK = '#1A1A1A'
_INK_MUTED = '#5F5F5F'
_GRID = '#E6E6E6'
_SPINE = '#B3B3B3'
_NEUTRAL_MARK = '#7A7A7A'

_LR_PATHS: tuple[tuple[str, ...], ...] = (
    ('optimizer', 'lr'),
    ('lr',),
    ('optimizer.lr',),
)
_N_BLOCKS_PATHS: tuple[tuple[str, ...], ...] = (
    ('model', 'n_blocks'),
    ('n_blocks',),
    ('model.n_blocks',),
)


# ----------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------


def plot_method_comparison(summary: dict[str, Any], path: str | Path) -> None:
    """Test accuracy per method: mean with std error bars + individual seed dots."""
    rows = _summary_rows(summary)
    n_methods = max(len(rows), 1)
    fig = _new_figure((max(5.2, 1.75 * n_methods + 1.8), 4.6))
    ax = fig.add_subplot()
    _style_axes(ax)

    dataset = summary.get('dataset')
    title = f'Test accuracy on {_dataset_name(dataset)}' if dataset else 'Test accuracy'
    ax.set_title(title, loc='left', fontsize=12, color=_INK, fontweight='bold')
    ax.set_ylabel('Test accuracy (%)', fontsize=10, color=_INK)

    if not rows:
        _note(ax, 'No runs to summarize')
        ax.set_xticks([])
        _save(fig, path)
        return

    unknown = sorted(r['method'] for r in rows if r['method'] not in _METHOD_COLORS)
    y_seen: list[float] = []
    tick_labels: list[str] = []
    any_member = False
    for x, row in enumerate(rows):
        method = row['method']
        color = _method_color(method, unknown)
        block = row.get('test') or {}
        values = _pct(_finite(_seed_values(block)))

        # Individual seeds: small translucent dots, left of the mean marker.
        if values.size:
            offsets = np.linspace(-0.07, 0.07, values.size) if values.size > 1 else [0]
            ax.scatter(
                x - 0.14 + np.asarray(offsets),
                values,
                s=30,
                facecolor=color,
                alpha=0.6,
                edgecolor=_SURFACE,
                linewidth=0.8,
                zorder=3,
            )
            y_seen.extend(values.tolist())

        mean = _as_float(block.get('mean'))
        if math.isnan(mean) and values.size:
            mean = float(np.mean(values)) / 100.0
        std = _as_float(block.get('std'))
        n = int(block.get('n', values.size) or 0)
        if not math.isnan(mean):
            mean_pct = mean * 100.0
            std_pct = std * 100.0 if not math.isnan(std) and n > 1 else 0.0
            ax.errorbar(
                [x + 0.1],
                [mean_pct],
                yerr=[[std_pct], [std_pct]] if std_pct > 0 else None,
                fmt='o',
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                capthick=1.8,
                markersize=7.5,
                markeredgecolor=_SURFACE,
                markeredgewidth=1.2,
                zorder=4,
            )
            label = f'{mean_pct:.2f} ± {std_pct:.2f}' if n > 1 else f'{mean_pct:.2f}'
            ax.annotate(
                label,
                xy=(x + 0.1, mean_pct),
                xytext=(9, 0),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=8.5,
                color=_INK_MUTED,
                zorder=5,
            )
            y_seen.extend([mean_pct - std_pct, mean_pct + std_pct])

        # Mean individual-member test score of ensembles: a neutral reference tick.
        member_mean = _as_float((row.get('member_test') or {}).get('mean'))
        if not math.isnan(member_mean):
            any_member = True
            ax.hlines(
                member_mean * 100.0,
                x - 0.3,
                x + 0.3,
                colors=_NEUTRAL_MARK,
                linewidth=1.4,
                zorder=2,
            )
            y_seen.append(member_mean * 100.0)

        name = row.get('display_name') or _METHOD_NAMES.get(method, method)
        wrapped = textwrap.fill(str(name), width=20, break_long_words=False)
        tick_labels.append(f'{wrapped}\n(n = {n})')

    ax.set_xticks(range(len(rows)), tick_labels, fontsize=9, color=_INK)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', length=0, pad=6)
    _set_padded_ylim(ax, y_seen, min_span=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'))

    # Neutral legend proxies (a NaN errorbar is invisible and ignored by autoscaling
    # but draws the real mean ± std glyph in the legend).
    mean_proxy = ax.errorbar(
        [math.nan],
        [math.nan],
        yerr=[1.0],
        fmt='o',
        color=_INK_MUTED,
        elinewidth=1.8,
        capsize=4,
        capthick=1.8,
        markersize=7,
        markeredgecolor=_SURFACE,
        label='mean ± std (ddof = 1)',
    )
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker='o',
            linestyle='none',
            markersize=5.5,
            markerfacecolor=_NEUTRAL_MARK,
            markeredgecolor=_SURFACE,
            alpha=0.6,
            label='individual seed',
        ),
        mean_proxy,
    ]
    if any_member:
        handles.append(
            Line2D(
                [],
                [],
                color=_NEUTRAL_MARK,
                linewidth=1.4,
                label='mean single-member accuracy',
            )
        )
    _legend(fig, handles)
    _save(fig, path)


def plot_online_ensemble_history(report: dict[str, Any], path: str | Path) -> None:
    """For one TabPack run: ensemble val/test score and #running members per epoch."""
    history = [h for h in (report.get('history') or []) if isinstance(h, Mapping)]
    epochs = np.array(
        [
            _as_float(h.get('epoch')) if h.get('epoch') is not None else float(i + 1)
            for i, h in enumerate(history)
        ],
        dtype=float,
    )

    fig = _new_figure((7.4, 5.6))
    ax_score, ax_members = fig.subplots(
        2, 1, sharex=True, gridspec_kw={'height_ratios': [2.0, 1.0]}
    )
    for ax in (ax_score, ax_members):
        _style_axes(ax)

    val, val_key = _history_series(history, ('ensemble_val', 'val_score', 'val'))
    test, test_key = _history_series(history, ('ensemble_test', 'test_score', 'test'))
    is_ensemble = (val_key or test_key or '').startswith('ensemble_')
    what = 'online ensemble during training' if is_ensemble else 'training history'
    fig.suptitle(
        f'{_report_title(report)}: {what}',
        x=0.01,
        ha='left',
        fontsize=12,
        color=_INK,
        fontweight='bold',
    )

    # Top panel: scores. The online ensemble is a held state between updates, so
    # its series are drawn as post-steps; per-epoch model scores (plain MLP) are
    # drawn as ordinary lines.
    drawstyle = 'steps-post' if is_ensemble else 'default'
    plotted_scores: list[float] = []
    score_handles = []
    markevery = max(1, len(history) // 30)
    prefix = 'Ensemble ' if is_ensemble else ''
    for values, color, marker, part in (
        (val, _VAL_COLOR, 'o', f'{prefix}validation'),
        (test, _TEST_COLOR, 's', f'{prefix}test'),
    ):
        if values is None or not np.isfinite(values).any():
            continue
        pct = values * 100.0
        final = pct[np.isfinite(pct)][-1]
        (line,) = ax_score.plot(
            epochs,
            pct,
            drawstyle=drawstyle,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=4.5,
            markevery=markevery,
            markeredgecolor=_SURFACE,
            markeredgewidth=0.8,
            label=f'{part.capitalize()} (final {final:.2f}%)',
            zorder=3,
        )
        score_handles.append(line)
        plotted_scores.extend(pct[np.isfinite(pct)].tolist())
    ylabel = 'Ensemble accuracy (%)' if is_ensemble else 'Accuracy (%)'
    ax_score.set_ylabel(ylabel, fontsize=10, color=_INK)
    if score_handles:
        _set_padded_ylim(ax_score, plotted_scores, min_span=0.5)
    else:
        _note(ax_score, 'No ensemble scores recorded in the history')

    # Bottom panel: member counts.
    count_handles = []
    for key, color, label in (
        ('n_running', _RUNNING_COLOR, 'Running members'),
        ('n_finished', _FINISHED_COLOR, 'Finished members'),
    ):
        counts, _ = _history_series(history, (key,))
        if counts is None or not np.isfinite(counts).any():
            continue
        (line,) = ax_members.plot(
            epochs,
            counts,
            drawstyle='steps-post',
            color=color,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
        count_handles.append(line)
    ax_members.set_ylabel('Members', fontsize=10, color=_INK)
    ax_members.set_xlabel('Epoch', fontsize=10, color=_INK)
    ax_members.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    if count_handles:
        ax_members.set_ylim(bottom=0)
    else:
        _note(ax_members, 'No member counts recorded in the history')
    if not history:
        ax_members.set_xticks([])
    else:
        ax_members.xaxis.set_major_locator(MaxNLocator(integer=True))
    # One legend below both panels: scores in the first column, counts in the second.
    handles = score_handles + count_handles
    if handles:
        _legend(fig, handles, ncols=min(len(handles), 2))
    _save(fig, path)


def plot_member_scores(report: dict[str, Any], path: str | Path) -> None:
    """For one TabPack run: individual member test score vs sampled lr / n_blocks,
    highlighting members selected into the final ensemble."""
    members = [m for m in (report.get('members') or []) if isinstance(m, Mapping)]
    run_config = (
        report.get('config') if isinstance(report.get('config'), Mapping) else {}
    )
    ensemble = (
        report.get('ensemble') if isinstance(report.get('ensemble'), Mapping) else {}
    )
    weights = Counter(_as_int(i) for i in (ensemble.get('ids') or []))

    scores = _pct(
        np.array([_metric(m.get('metrics'), 'test') for m in members], dtype=float)
    )
    selected_count = np.array(
        [weights.get(_as_int(m.get('id')), 0) for m in members], dtype=int
    )
    lrs = np.array([_member_param(m, run_config, _LR_PATHS) for m in members])
    blocks = np.array([_member_param(m, run_config, _N_BLOCKS_PATHS) for m in members])

    fig = _new_figure((10.0, 4.6))
    ax_lr, ax_nb = fig.subplots(1, 2, sharey=True)
    for ax in (ax_lr, ax_nb):
        _style_axes(ax)

    counts = f'{len(members)} finished'
    if weights:
        counts += f', {int((selected_count > 0).sum())} in the final ensemble'
    fig.suptitle(
        f'{_report_title(report)}: individual members ({counts})',
        x=0.01,
        ha='left',
        fontsize=12,
        color=_INK,
        fontweight='bold',
    )
    ax_lr.set_ylabel('Member test accuracy (%)', fontsize=10, color=_INK)
    ax_lr.set_xlabel('Learning rate (log scale)', fontsize=10, color=_INK)
    ax_nb.set_xlabel('Number of MLP blocks', fontsize=10, color=_INK)

    ok = np.isfinite(scores)
    # Left: learning rate.
    ok_lr = ok & np.isfinite(lrs) & (lrs > 0)
    if not ok.any():
        empty = 'No members in the report' if not members else 'No member test scores'
        for ax in (ax_lr, ax_nb):
            _note(ax, empty)
    elif ok_lr.any():
        ax_lr.set_xscale('log')
        ax_lr.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax_lr.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax_lr.xaxis.set_minor_formatter(NullFormatter())
        _member_scatter(ax_lr, lrs[ok_lr], scores[ok_lr], selected_count[ok_lr])
    else:
        ax_lr.set_xticks([])
        _note(ax_lr, 'No per-member learning rate recorded', clear_y=False)
    # Right: number of blocks, with a deterministic horizontal spread per value.
    ok_nb = ok & np.isfinite(blocks)
    if ok_nb.any():
        xs = blocks[ok_nb] + _spread_within_groups(blocks[ok_nb], width=0.36)
        _member_scatter(ax_nb, xs, scores[ok_nb], selected_count[ok_nb])
        uniq = np.unique(blocks[ok_nb])
        ax_nb.set_xticks(uniq, [f'{int(v)}' if v == int(v) else f'{v:g}' for v in uniq])
        ax_nb.set_xlim(uniq.min() - 0.6, uniq.max() + 0.6)
        ax_nb.xaxis.grid(False)
    elif ok.any():
        ax_nb.set_xticks([])
        _note(ax_nb, 'No per-member n_blocks recorded', clear_y=False)

    y_seen = scores[ok].tolist()
    handles: list[Line2D] = []
    final = _metric(report.get('metrics'), 'test') * 100.0
    if not math.isnan(final):
        for ax in (ax_lr, ax_nb):
            ax.axhline(final, color=_INK_MUTED, linewidth=1.2, zorder=1)
        y_seen.append(final)
        handles.append(
            Line2D(
                [],
                [],
                color=_INK_MUTED,
                linewidth=1.2,
                label=f'final {"ensemble" if weights else "prediction"} ({final:.2f}%)',
            )
        )
    if y_seen:
        _set_padded_ylim(ax_lr, y_seen, min_span=0.5)
    ax_lr.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'))

    member_handles = [
        Line2D(
            [],
            [],
            marker='o',
            linestyle='none',
            markersize=6,
            markerfacecolor='none',
            markeredgecolor=_NEUTRAL_MARK,
            markeredgewidth=1.2,
            label='not selected' if weights else 'member',
        )
    ]
    if weights:
        member_handles.insert(
            0,
            Line2D(
                [],
                [],
                marker='o',
                linestyle='none',
                markersize=8,
                markerfacecolor=_SELECTED_COLOR,
                markeredgecolor=_SURFACE,
                label='in final ensemble (area grows with weight)',
            ),
        )
    _legend(fig, member_handles + handles)
    _save(fig, path)


# ----------------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------------


def _new_figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize, dpi=_DPI, layout='constrained', facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix[1:].lower() or 'png'
    fig.savefig(
        path,
        dpi=_DPI,
        format=fmt,
        facecolor=_SURFACE,
        bbox_inches='tight',
        pad_inches=0.12,
    )


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK_MUTED, labelsize=9, width=0.8, length=3)
    ax.grid(True, axis='y', color=_GRID, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)


def _legend_style() -> dict[str, Any]:
    return {
        'frameon': False,
        'fontsize': 9,
        'labelcolor': _INK,
        'handlelength': 1.8,
    }


def _legend(fig: Figure, handles: Sequence[Any], *, ncols: int | None = None) -> None:
    fig.legend(
        handles=list(handles),
        loc='outside lower center',
        ncols=ncols or len(handles),
        **_legend_style(),
    )


def _note(ax: Axes, text: str, *, clear_y: bool = True) -> None:
    """Explain an empty panel; by default drop its (meaningless) y ticks. Pass
    clear_y=False when the y axis is shared with a panel that has data."""
    if clear_y:
        ax.set_yticks([])
        ax.grid(False)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=9.5,
        color=_INK_MUTED,
    )


def _set_padded_ylim(ax: Axes, values: Iterable[float], *, min_span: float) -> None:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if vals.size == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, min_span)
    mid = (lo + hi) / 2.0
    lo, hi = mid - span / 2.0, mid + span / 2.0
    pad = 0.12 * span
    ax.set_ylim(lo - pad, hi + pad)


def _member_scatter(
    ax: Axes, xs: np.ndarray, ys: np.ndarray, selected_count: np.ndarray
) -> None:
    rest = selected_count == 0
    ax.scatter(
        xs[rest],
        ys[rest],
        s=34,
        facecolor='none',
        edgecolor=_NEUTRAL_MARK,
        linewidth=1.2,
        zorder=2,
    )
    sel = ~rest
    ax.scatter(
        xs[sel],
        ys[sel],
        s=60.0 + 40.0 * (selected_count[sel] - 1),
        facecolor=_SELECTED_COLOR,
        edgecolor=_SURFACE,
        linewidth=1.2,
        zorder=3,
    )


def _spread_within_groups(values: np.ndarray, *, width: float) -> np.ndarray:
    """Deterministic horizontal offsets that spread equal values over `width`."""
    offsets = np.zeros(values.shape, dtype=float)
    for v in np.unique(values):
        idx = np.flatnonzero(values == v)
        if idx.size > 1:
            offsets[idx] = np.linspace(-width / 2.0, width / 2.0, idx.size)
    return offsets


# ----------------------------------------------------------------------------------
# Data helpers (all tolerant to missing / None fields)
# ----------------------------------------------------------------------------------


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = summary.get('methods') or []
    if isinstance(raw, Mapping):
        raw = [{'method': k, **v} for k, v in raw.items() if isinstance(v, Mapping)]
    rows = [dict(r) for r in raw if isinstance(r, Mapping) and r.get('method')]
    rank = {m: i for i, m in enumerate(_METHOD_ORDER)}
    return sorted(rows, key=lambda r: (rank.get(r['method'], len(rank)), r['method']))


def _method_color(method: str, unknown: Sequence[str]) -> str:
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    i = list(unknown).index(method) if method in unknown else len(_EXTRA_COLORS)
    return _EXTRA_COLORS[i] if i < len(_EXTRA_COLORS) else _FALLBACK_COLOR


def _dataset_name(dataset: Any) -> str:
    text = str(dataset)
    return text[:1].upper() + text[1:]


def _report_title(report: Mapping[str, Any]) -> str:
    method = report.get('method')
    name = _METHOD_NAMES.get(str(method), str(method)) if method else 'Run'
    seed = report.get('seed')
    return f'{name}, seed {seed}' if seed is not None else name


def _seed_values(block: Mapping[str, Any]) -> list[Any]:
    values = block.get('values')
    return list(values) if isinstance(values, Sequence) else []


def _as_float(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value.get('score')
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.array([_as_float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def _pct(values: np.ndarray) -> np.ndarray:
    return values * 100.0


def _metric(metrics: Any, part: str) -> float:
    """Score of `part` (falls back to accuracy) from a {part: {...}} metrics dict."""
    if not isinstance(metrics, Mapping):
        return math.nan
    part_metrics = metrics.get(part)
    if not isinstance(part_metrics, Mapping):
        return math.nan
    score = _as_float(part_metrics.get('score'))
    return score if not math.isnan(score) else _as_float(part_metrics.get('accuracy'))


def _history_series(
    history: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if any(h.get(key) is not None for h in history):
            return np.array([_as_float(h.get(key)) for h in history], dtype=float), key
    return None, None


def _lookup(config: Any, path: Sequence[str]) -> Any:
    node = config
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _member_param(
    member: Mapping[str, Any],
    run_config: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> float:
    """A hyperparameter of a member: its own config first, then the run config
    (homogeneous members have config None and share the run's hyperparameters)."""
    for config in (member.get('config'), run_config):
        for path in paths:
            value = _as_float(_lookup(config, path))
            if not math.isnan(value):
                return value
    return math.nan
