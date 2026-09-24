"""Reduced heterogeneous TabPack (a32).

1. member configs: config.configs, or sample_configs(config.space, n_models, seed);
2. ModelPack with per-member n_blocks / dropout (d_block fixed);
3. MuonAdamWPack with per-member lr / weight_decay / muon_lr,
   make_param_groups(muon=True);
4. train_pack with an OnlineGreedyEnsemble (val score, max_ensemble_size, patience);
5. final prediction = the online ensemble; report members (with configs), the
   ensemble ids/steps, and the best single member.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabpack_repro.config import TabPackConfig


def run(config: TabPackConfig, output_dir: str | Path) -> dict[str, Any]:
    raise NotImplementedError
