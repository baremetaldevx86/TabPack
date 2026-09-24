"""Official Churn numbers for comparison (a38).

Reads the report.json files shipped in the official repository (cloned outside this
repo; path from $TABPACK_REFERENCE_DIR) and extracts Churn results, e.g.
experiments/tabpack/churn/main/report.json (64 models, A100):
online_ensembles.greedy.report.metrics.test.score, best single model, mean member
score, n_models, time. Writes ``results/reference/churn_official.json`` so the
comparison works without the clone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_reference_dir() -> Path | None:
    """$TABPACK_REFERENCE_DIR if it exists, else <repo>/.reference/tabpack if it
    exists, else None."""
    raise NotImplementedError


def extract_churn_reference(reference_dir: str | Path) -> dict[str, Any]:
    """Collect Churn numbers from every experiments/*/churn/**/report.json."""
    raise NotImplementedError


def load_saved_reference(
    path: str | Path = 'results/reference/churn_official.json',
) -> dict[str, Any]:
    raise NotImplementedError
