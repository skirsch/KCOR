# 8.3 - Sensitivity sweep double-counts overlapping populations

**Severity:** High for sensitivity analysis

**Origin:** New specific defect

**Related issues / root causes:** 6.1, 7.4, 5.7: dedicated path bypasses ordinary population and normalization logic

**Finding:** Blind B verified that the sensitivity branch sums disjoint birth bands with all-ages and younger totals. In Czech 2021-26, dose 0 genuine 883/5,859,062 deaths/risk becomes 1926/15,492,550; dose 2 genuine 630/1,837,004 becomes 1290/4,302,387. The path also uses row-number time and proportional inversion despite a possible forced nonlinear fit.

**Proposed scope (for discussion):** Use the established ordinary population/calendar/normalization path with only sweep parameters varied. Verify one matched-settings cell and redundant-total invariance, then rerun affected sensitivity evidence.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m138, preserving m137. Independent review not yet repeated.


The SA branch now selects the same all-ages source as the ordinary estimator, invokes its normal calendar checks, fitting and normalization for each shifted quiet window, and reanchors its unanchored series to the specified reference date. No separate proportional inversion remains, so forced-alpha fits and curves use the same survivor correction. Only the sweep path changed. A regression tested no redundant aggregate, all-ages total, and both all-ages and younger total, at 2, 4 and 8 reference weeks in both module-off and forced-alpha modes. Matched ordinary points and redundant-total invariance passed to 1e-10.

The current grid preserves the original 7 x 7 parameter design (offsets -12..+12, end 2023-52, reference lengths 2..8), uses the current 27 configured comparisons, and current one-year reporting dates. The archived v6 workbook contains 34 comparisons, some with different reporting dates and older fitting settings; it remains unchanged. All 1,323 current cells are available, with contributing numerical fit status OK. Numerical acceptance does not establish diagnostic identification.

At identical current settings (2021-24, quiet start offset 0, four reference weeks, report 2022-06-13), the population-only comparison gives:

| Comparison | Double-counted population | Correct population |
|---|---:|---:|
| 1/0 | 1.017931 | 1.007088 |
| 2/0 | 1.102381 | 1.099128 |
| 2/1 | 1.082962 | 1.091392 |

These are point estimates in a sensitivity comparison, not confidence intervals. Across all current comparisons, 15/27 grids straddle one and the range is 0.698755-1.602944. All 1,323 workbook cells match the saved CSV within 6.67e-16. The new figure and scope are addressed in [8.8](../8.8/README.md). No change to ordinary KCOR or Table 8.

[Grid workbook](KCOR_SA.xlsx) | [Cells](KCOR_SA_cells.csv) | [Ranges](ranges.csv) | [Population-only comparison](population-only-comparison.json) | [Regression](test_sensitivity_ordinary.py) | [Provenance](provenance.json).

**Round 9 independent disposition:** partial; see [9.2](../9.2/README.md). [Resolution](../../iterative%20review/evaluations/9/resolution.md).
