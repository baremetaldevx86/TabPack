"""Figures (a37). Matplotlib, Agg backend, PNG at 150 dpi; never plt.show()."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_method_comparison(summary: dict[str, Any], path: str | Path) -> None:
    """Test accuracy per method: mean with std error bars + individual seed dots."""
    raise NotImplementedError


def plot_online_ensemble_history(report: dict[str, Any], path: str | Path) -> None:
    """For one TabPack run: ensemble val/test score and #running members per epoch."""
    raise NotImplementedError


def plot_member_scores(report: dict[str, Any], path: str | Path) -> None:
    """For one TabPack run: individual member test score vs sampled lr / n_blocks,
    highlighting members selected into the final ensemble."""
    raise NotImplementedError
