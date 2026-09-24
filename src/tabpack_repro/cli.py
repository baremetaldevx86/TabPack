"""Command-line interface (a34): ``tabpack-repro`` / ``python -m tabpack_repro``.

Subcommands:
  download [--name churn] [--data-dir DIR] [--force]
  run --config configs/churn/<method>.toml [--seed S] --output DIR [--device D]
  conservative --source-run DIR [--n-seeds 5] --output DIR
  summarize --runs-dir runs/churn --output results/churn
  reference --output results/reference/churn_official.json
Uses argparse only. Exit code 0 on success, 2 on usage errors.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError
