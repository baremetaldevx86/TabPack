# a37-report-plots: agent report

## Summary

Implemented the three figure functions in `reporting/plots.py`:

- `plot_method_comparison(summary, path)`: test accuracy (%) per method. Each seed
  is a dot. The mean is a marker with ±std error bars (ddof = 1) and a
  `mean ± std` label. For ensembles, a neutral reference tick marks the mean
  single-member accuracy.
- `plot_online_ensemble_history(report, path)`: two stacked panels that share
  the epoch axis. The top panel shows ensemble val/test accuracy as post-steps,
  because the online ensemble holds its state between updates. The bottom panel
  shows the running and finished member counts. The figure does not use a dual
  axis.
- `plot_member_scores(report, path)`: each member's test accuracy against its
  sampled learning rate (log scale) and against n_blocks (integer axis with a
  deterministic spread). Members in the final ensemble are filled and larger (area
  grows with their weight in the ensemble). The other members are hollow. A line
  marks the final ensemble score.

The figures are driven by the exact summary shape that a36 posted (#110) and by the
run-report schema (`methods/report.py`, trainer `PackTrainResult.history`).

## Files

- `src/tabpack_repro/reporting/plots.py`: implementation plus private helpers.
- `tests/reporting/test_plots.py`: 21 tests.
- `evidence/agents/a37-report-plots.md`: this report.

## Design decisions

- **No global pyplot state.** Every figure is a bare `matplotlib.figure.Figure`
  rendered through `FigureCanvasAgg`. The code never imports pyplot, never switches
  backends and never calls `show()`. Figures are not registered anywhere, so they
  are garbage-collected after `savefig`. Figures use constrained layout, and
  `savefig` uses `bbox_inches='tight'` and 150 dpi. Parent directories are created.
  The output format follows the file suffix and defaults to PNG, so `.svg` or
  `.pdf` also work.
- **Colors.** Methods keep a fixed canonical order (mlp, homogeneous, tabpack,
  tabpack-conservative, then unknown methods sorted). Each method has a fixed
  Okabe-Ito color: orange, blue, bluish green, vermillion. I checked the set with
  the dataviz palette validator. The smallest all-pairs CVD ΔE is 11.0, and the
  normal-vision floor is 15.6 (pass). Orange has low contrast against white, which
  is acceptable because every method is also named on the axis. Color never
  carries identity alone. Methods are named on the x axis, series have legends,
  and selected members differ by fill and size as well as by color. Text uses
  neutral ink, never the series color. Gridlines are solid hairlines.
- **Units.** Summaries and reports store raw fractions (a36 #110). Every figure
  multiplies by 100 and labels the axis "accuracy (%)". On Churn, score is
  accuracy. Member and history values use `score` and fall back to `accuracy`.
- **Robustness.** Missing optional fields render a short note in the affected
  panel instead of raising. The note also clears meaningless ticks, but only on
  unshared axes. The handled cases are:
  - `None`/NaN ensemble scores. The homogeneous ensemble and the epochs before the
    first ensemble update have none.
  - A history without `epoch`. The 1-based position is used instead.
  - Missing `n_running` or `n_finished`.
  - An empty or `None` history.
  - The plain-MLP history keys `val_score`/`test_score` (a30). These are drawn as
    ordinary lines and titled "training history".
  - Members with `config: null`. Their hyperparameters are looked up in the run
    config, as for homogeneous runs.
  - No ensemble. The legend then says "member" and "final prediction".
  - Members without test scores.
  - An empty summary.
  - A single seed. There is no error bar, and the label shows the mean only.
  - Rows out of order or of an unknown method.
- **Legend proxy for mean ± std.** The proxy is an invisible NaN `errorbar`. It
  draws the real marker-with-caps glyph in the legend and does not affect
  autoscaling. A plain `Line2D` proxy would draw a misleading line.

## Tests

`tools/dev/py -m pytest tests/reporting/test_plots.py -q -W error` gives
**21 passed** in about 7 s (CPU). `tools/dev/py -m pytest tests/reporting -q` gives
58 passed, including a36's tests. `ruff check` and `ruff format --check` on the
owned files pass.

The tests cover these checks:

- Each figure writes a file with the PNG magic header and a non-trivial size
  (more than 20 kB for full figures). The PNG `pHYs` chunk records 150 dpi.
- Nested parent directories are created, and `str` paths work.
- Inputs are not mutated.
- Degraded inputs render, including all the missing-field cases above.
- The format follows the suffix (SVG).
- An integration test runs a36 `summarize()` on fabricated run reports and plots
  the result.
- A subprocess test asserts that importing and using the module never imports
  `matplotlib.pyplot` and leaves the backend unchanged.
- An autouse fixture asserts that no pyplot figure is left open.

I also rendered each figure (full TabPack, homogeneous/no-ensemble, missing-lr
variants) and checked them by eye for label collisions and overflow. As a result,
I moved the history legends below the panels, widened the wrapping of method
names, set the lr ticks to 1-2-5 steps and fixed the empty-panel ticks.

## Coordination

- Posted `status` #104 at the start.
- Asked a36 for the exact summary keys (#105). a36 answered with the full shape
  (#110). I acknowledged it (#118), and the implementation follows #110 exactly:
  `methods[]` rows, `display_name`, BLOCK `{n, mean, std, min, max, values}`
  aligned with `seeds`, and the optional `member_test`.
- Read a30's `feat/a30-method-mlp` branch (read-only, `git show`) to support the
  plain-MLP history keys `val_score`/`test_score`.
- Merged `feat/a36-report-summarize` after a36 posted `done` (#145), for the
  integration test.

## Open issues

- The figures label the y axis "accuracy (%)". That is correct for Churn, where
  score is accuracy. For a regression task the score is -RMSE, and the labels and
  the ×100 scaling would be wrong. Regression is out of scope because the pipeline
  raises `NotImplementedError`.
- In `plot_member_scores`, each member is placed at its best-checkpoint test
  score. The online ensemble (`update_type='latest'`) may hold a member at a
  different step. The highlight is by member id, and the marker area grows with
  how many times the id appears in `ensemble.ids`.
