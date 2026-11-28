# 8.8 - Sensitivity figure provenance, scales and robustness claim

**Severity:** Moderate to high for displayed evidence

**Origin:** New specific reporting defect, linked to 8.3

**Related issues / root causes:** 2.5, 8.3: sensitivity population and actual plotting evidence

**Finding:** Both blind reviewers observed different color bars despite common scale claim. B found 12/34 archived grids straddle 1 and the displayed figure does not exactly match the archived grid. The figure shows only selected 2021-24 comparisons.

**Proposed scope (for discussion):** Trace original figure and data; regenerate with corrected sweep if necessary. Distinguish window fitting and normalization-reference effects, state actual ranges/selection and scale.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented jointly with 8.3 in m138, preserving m137. Independent review not yet repeated.


The source of the old figure could not be matched exactly to its archived v6 grid. The archive itself has 12/34 comparisons straddling 1; the current 27-comparison sweep uses the ordinary corrected pipeline and current one-year horizon, so the two sweeps are not a controlled like-for-like historical comparison. The original three 2021-24 displayed comparisons are retained in the new figure with a single symmetric color scale around 1. The supplement states the actual settings, selected comparisons, 15/27 crossings and full-grid range, and distinguishes quiet-start from reference sensitivity. The point-only ranges are:

| Comparison | Original archived workbook range | Current corrected range |
|---|---|---|
| 1/0 | 1.029-1.136 | 0.960-1.052 |
| 2/0 | 1.105-1.281 | 1.036-1.188 |
| 2/1 | 1.066-1.128 | 1.068-1.141 |

Within these three current grids the maximum quiet-start shift is 0.011 at fixed reference, and maximum reference change is 0.147 at fixed quiet start. These are point sensitivities, not confidence intervals. The current figure is generated directly from the verified current workbook. [Old figure](figure-before.png) | [New figure](figure-after.png) | [Full sweep](../8.3/KCOR_SA.xlsx) | [Historical ranges](../8.3/historical-ranges.json).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).
