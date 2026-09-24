"""The common run-report schema (frozen contract, used by methods/* and reporting/*).

report.json of a single run:
{
  "schema_version": 1,
  "method": "mlp" | "homogeneous" | "tabpack" | "tabpack-conservative-seed",
  "dataset": "churn",
  "seed": int,
  "config": {...},                         # config_to_dict(config)
  "metrics": {"val": {...}, "test": {...}},  # of the method's FINAL prediction
                                           # (single model / averaged ensemble /
                                           # online greedy ensemble); includes "score"
  "members": [                             # every finished member
      {"id": int, "best_step": int, "config": {...} | null,
       "metrics": {"train": {...}, "val": {...}, "test": {...}}}
  ],
  "ensemble": null | {"ids": [...], "steps": [...], "size": int, "n_unique": int},
  "best_member": null | {"id": int, "metrics": {...}},   # by val score
  "n_epochs": int, "n_steps": int, "time_sec": float,
  "env": {"device": str, "gpu": str | null, "torch": str, "git_commit": str | null},
  "history": [...]                          # PackTrainResult.history (or per-epoch
                                           # dicts for the plain MLP)
}
TabPack runs additionally record (a32): "n_models", "n_finished",
"member_configs" (all K sampled configs, list index = member id; needed because the
online ensemble may select a member that never finished, so its config is absent
from "members"; the official code uses experiments.json the same way),
"online_ensemble" (OnlineGreedyEnsemble.report()) and "best_member"."config".
Files next to report.json: ``predictions.npz`` with the final val/test predictions
(keys "val", "test"), and ``config.toml`` (dump_config).

The conservative evaluation writes one report per seed under
``<output_dir>/seed-<s>/`` (method "tabpack-conservative-seed") and an aggregate
``<output_dir>/report.json`` with {"method": "tabpack-conservative",
"source_run": str, "selected_ids": [...], "n_seeds": int, "seeds": [...],
"scores": {"val": [...], "test": [...]}, "mean": {...}, "std": {...}}.
"""

RUN_REPORT_SCHEMA_VERSION = 1
RUN_REPORT_SCHEMA_DOC = __doc__
