"""Ordinary MLP baseline (a30): a plain nn.Sequential MLP trained with torch AdamW.

Deliberately independent of the pack machinery (it doubles as a sanity reference):
Linear -> ReLU -> Dropout blocks (n_blocks x d_block), linear head; weight decay on
weight matrices only (biases: 0.0), same batch size / early stopping / AMP settings /
data pipeline as the other methods; evaluation on val+test after every epoch; the
test metric is taken at the best-val epoch (strict improvement).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabpack_repro.config import MLPMethodConfig


def run(config: MLPMethodConfig, output_dir: str | Path) -> dict[str, Any]:
    raise NotImplementedError
