# Standalone bug package: Monte Carlo sample isolation

This software-fix package can be reviewed and adopted independently of the objections to KCOR's statistical method. It was discovered during objection 3 work, but its validity does not depend on that objection.

[PR #4](https://github.com/skirsch/KCOR/pull/4), **Fix Monte Carlo sample clocks and fit diagnostics**, was checked on 2026-11-13: open and unmerged. The pinned PR head is `ab5baf07fad1b0b340e494d22bb39c13295a2441`, based on `b639be2572e04c3fbc8d15e102c56b87295959c9`.

## What is included

- [REVIEWER_NOTE.md](REVIEWER_NOTE.md): exact copy of the explanation already included in the PR as `test/monte_carlo/README.md`, written for human and LLM reviewers.
- [fix.patch](fix.patch): exact diff from the PR base to its head; production change, reviewer note, and standalone regression.
- [test_sample_isolation.py](test_sample_isolation.py): copy of the committed regression for inspection. Apply the patch and run it in its intended `test/monte_carlo/` location.
- [verification.json](verification.json): retained verification against the exact PR checkout, including the original-versus-fixed regressions and historical synthetic workbook replay.
- [provenance.json](provenance.json): source revisions and package hashes.

## Why these are bugs under KCOR's own definition

KCOR measures elapsed weeks since enrollment and applies the same fitting procedure to each bootstrap replicate. Adding other alternative datasets to a workbook cannot make more calendar time elapse for an unchanged dataset.

The affected time counter grouped by birth year and dose but omitted the sample identifier. For ten samples on a complete weekly grid, the original cohort could receive times 0, 10, 20, 30 instead of 0, 1, 2, 3. The skipped-week rule and fitted mortality model use those numbers. This is not a harmless label change or a consistent conversion of units.

The original input was identical when analyzed alone and alongside nine resamples. At week 80, its output changed from 1.004049494273245 to 1.053172263346009. The repair restores all 100 original curve points exactly. These are synthetic reproduction values.

The accompanying reporting defect came from expecting ordinary three-part fit-record keys while Monte Carlo records have a fourth part for the sample identifier. The summary could therefore omit failed fits. The regression explicitly supplies a failed fit and verifies that it is counted and attributed correctly.

## Resampling scope

The clock defect is downstream of resample generation. It would affect person resamples or weekly-count resamples passed through this affected combined-workbook path. The separate simulation-grid Poisson procedure was not shown to have this shared-clock defect.

The production person bootstrap, the separate weekly Poisson experiment, and the paper's aggregate-block description are different procedures. Choosing one is a statistical question; maintaining the correct timeline for a selected sample is a software correctness requirement.

This patch implements sample isolation under the existing complete-weekly-grid convention. It does not redesign handling of missing calendar weeks. Ordinary single-dataset runs do not trigger this cross-sample defect. Attributing it to historical manuscript outputs requires their generating source revision and workflow.

## Reproduce or adopt independently

Against the matching PR base in a clean checkout with existing KCOR dependencies:

```powershell
git apply --check "path/to/fix.patch"
git apply "path/to/fix.patch"
python test/monte_carlo/test_sample_isolation.py
```

The test creates synthetic workbooks and requires no original person records. Two of three tests fail on the affected source; all three pass with the fix. The test and reviewer note are included in the patch.

The older [objection3_production_fix](../objection3_production_fix/) folder remains the immutable preparation and replay archive. Its preparation README predates submission and should not be used for current PR status. [Objection 3](<../../objections/1.3/README.md>) separately evaluates corrected interval calibration. This package does not propose the later joint-fit candidate.

