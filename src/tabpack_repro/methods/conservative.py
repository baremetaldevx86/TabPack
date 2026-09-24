"""The paper's conservative evaluation protocol for TabPack (a33).

Read ``<source_run>/report.json`` (a TabPack main run), take the sorted unique ids of
its final online ensemble, look up their configs in report["members"], and for each
seed s in range(n_seeds) run methods.tabpack.run with ``configs=selected``,
``n_models=len(selected)``, ``seed=s`` (all other settings copied from the source
run's config) into ``<output_dir>/seed-<s>``. Aggregate as described in
methods/report.py. Must be resumable: skip seeds whose report.json exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabpack_repro.config import ConservativeEvalConfig


def run(config: ConservativeEvalConfig, output_dir: str | Path) -> dict[str, Any]:
    raise NotImplementedError
