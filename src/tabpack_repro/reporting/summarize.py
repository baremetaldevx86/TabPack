"""Summaries across seeds and methods (a36)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Load every report.json below runs_dir (recursively), adding "path"."""
    raise NotImplementedError


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method mean/std/min/max/n of test and val scores + mean time.

    Rows (in this order when present): 'mlp', 'homogeneous', 'tabpack' (single-run
    online ensemble over seeds), 'tabpack-conservative' (paper protocol; uses the
    aggregate report's per-seed scores). Also: mean individual-member test score for
    ensembles, mean ensemble size. Uses the sample std (ddof=1), as statistics.stdev.
    """
    raise NotImplementedError


def to_markdown(summary: dict[str, Any]) -> str:
    """A GitHub-flavored markdown table: method | test acc (mean ± std) | val acc |
    n seeds | mean time."""
    raise NotImplementedError


def write_summary(summary: dict[str, Any], out_dir: str | Path) -> None:
    """Write summary.json, summary.md and summary.csv into out_dir."""
    raise NotImplementedError
