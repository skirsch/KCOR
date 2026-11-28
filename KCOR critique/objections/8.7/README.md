# 8.7 - Frailty-misspecification experiments overclaimed

**Severity:** Moderate to high for validation reporting

**Origin:** New specific unsupported completed-study claim

**Related issues / root causes:** 1.4, 1.5, 4.8: evidence and simulation target

**Finding:** Both blind reviewers could not find the claimed five-distribution by four-metric results for gamma, lognormal, two-point, bimodal and correlated frailty. Existing studies use different estimators and targets.

**Proposed scope (for discussion):** Trace original evidence and scope wording to what was actually run; add missing evidence only if needed and authorized.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m142; m141 preserved. Independent closure not yet evaluated.


## Original evidence inventory and approved study design

The five-distribution language traces to manuscript-only commits b7b79ce8 and c0858a5a. No accompanying matched simulation code or result matrix was added in those commits. The frozen candidate includes the historical gamma/lognormal simulation grid (test/sim_grid/code/generate_sim_grid.py, out/sim_grid_diagnostics.csv), the previous five-scenario coverage archive (out/bootstrap_coverage.csv, using a single fitted reference rather than a known population target), and the continuous tail-sampling study (code/generate_tail_sampling_sim.py, out/tail_sampling/tail_sampling_diagnostics.csv). The tail-null median was 0.9501 with drift -0.0598 under the earlier single-window fitter. Searches found no matching two-point or correlated-frailty replicated coverage studies. The historical evidence is preserved.

The new matched experiment is approved to complete the original intended comparison. Use the original simulation-grid scale (100,000 per cohort, 120 weeks, weekly baseline .002, theta 1 and .3, quiet weeks 20-80, skip 2), updated to the current fixed-slope Gompertz model and all-window fitting objective (one designated quiet window and early anchors). All five laws share generating mean-one and variance within each arm. The target anchored endpoint is 1 under the fixed-frailty null. Use 100 independent datasets per law, 199 full-refit resamples per dataset. Independent cases resample individuals; correlated gamma case uses clusters of 20 sharing frailty, with a paired person-resampling diagnostic. These are explicit validation settings, not new default KCOR settings. Report numerical-fit status flags separately from exploratory residual screen and actual failures; coverage denominator includes all attempted independent datasets. No automatic diagnostic threshold is introduced.

[Study generator](misspecification_study.py) | [Ordinary workbook verification](workbook-verification.json).

## Completed results

All 500 independent datasets and 119,400 refits (including the paired correlated bootstrap) completed with no numerical failures. All mean-KCOR Monte Carlo confidence intervals contain 1; mean bias ranges from -0.00635 to +0.00503. Coverage is 92--99%, with all Wilson intervals containing 95%. The correlated design produces similar cluster- and person-bootstrap coverage and widths; this does not establish that person resampling is valid for arbitrary dependence. The residual screen flagged 4--13 of 100 datasets per scenario, not necessarily a detected distributional mismatch. The non-gamma fitted frailty variance need not equal the true variance.

| Frailty / bootstrap | Mean KCOR [95% MC CI] | Sampling SD | Coverage [95% Wilson CI] | Residual flags |
|---|---|---:|---|---:|
| gamma / person | 0.995 [0.988, 1.003] | 0.0378 | 98% [93.0, 99.4] | 4 |
| lognormal / person | 0.994 [0.986, 1.001] | 0.0375 | 99% [94.6, 99.8] | 6 |
| two_point / person | 0.994 [0.986, 1.002] | 0.0406 | 94% [87.5, 97.2] | 12 |
| bimodal / person | 1.002 [0.993, 1.011] | 0.0453 | 92% [85.0, 95.9] | 9 |
| correlated_gamma / cluster | 1.005 [0.997, 1.013] | 0.0394 | 96% [90.2, 98.4] | 13 |
| correlated_gamma / person | 1.005 [0.997, 1.013] | 0.0394 | 96% [90.2, 98.4] | 13 |

[Full results](study/summary.json), [dataset-level intervals](study/replicates.csv), [design](study/design.json), [expected counts](study/expected-counts.json). Main Section 3.3.1 and SI S4.5 now distinguish historical tests from current matched validation; SI S9 reports all outcomes. No estimator changes.

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).
