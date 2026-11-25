Processing multiple bootstrap samples together could change the estimate for the unchanged original sample. The production `make monte_carlo` workflow combines samples into one enrollment worksheet, but `process_workbook` counted elapsed time by birth year and dose without `mc_id`. Observations from one calendar week therefore received different time indices, affecting skipped weeks and fitting.

This PR gives each Monte Carlo sample its own time counter. It also fixes the related diagnostic reporting defect: Monte Carlo fit records have four-part keys, while several summaries expected three-part keys. Failed fits are now counted and retained with their sample identifiers.

Scope is three files: the production fix, a synthetic regression test, and `test/monte_carlo/README.md`. The README explains the trigger, execution path, cause, reproduction, and interpretation boundaries for human and LLM reviewers. Ordinary single-dataset grouping, the mathematical model, bootstrap sampling algorithm, interval formulas, and manuscript are unchanged.

Validation:

- On the unmodified upstream implementation, three tests run and two fail as expected.
- On the patched implementation, all three pass, including sample-count independence, shuffled/relabelled samples, failed-fit reporting, and ordinary processing on the fixture.
- Replaying retained ten-sample synthetic input restores all 100 original-sample KCOR curve points exactly to the original-alone result. At week 80, the old combined result was 1.053172263346009; both original-alone and corrected combined results are 1.004049494273245. Replay artifacts remain local; the committed regression is standalone.
- `git diff --check` passes.

Run: `python test/monte_carlo/test_sample_isolation.py`.

This establishes the software repair, not confidence-interval coverage. Intervals generated through the affected workflow need recomputation and fresh calibration. Whether historical manuscript intervals are affected requires identifying their generating revision and workflow.
