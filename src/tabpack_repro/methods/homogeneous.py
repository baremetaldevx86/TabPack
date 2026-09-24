"""Homogeneous MLP ensemble (a31): K identical-hyperparameter MLPs trained as a pack.

Members differ only by initialization and batch order. AdamWPack with scalar
hyperparameters, make_param_groups(muon=False). Every member early-stops on its own;
the final prediction is the UNIFORM average of all members' best-checkpoint
probabilities (no selection). Also report each member's individual metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabpack_repro.config import HomogeneousEnsembleConfig


def run(config: HomogeneousEnsembleConfig, output_dir: str | Path) -> dict[str, Any]:
    raise NotImplementedError
