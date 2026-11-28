# 6.8 ? Age-shift control captions claim missing interval bands and guaranteed flatness

**Severity:** Moderate

Both published historical plots appear to display no 95% bands although captions claim them. Age-shift controls lack a known exact numerical target. Verify actual series, reference and bootstrap provenance; correct captions or provide the appropriate intervals.

**Related issues:** 1.4, 2.10.

**Origin:** Sharpened existing control limitation plus new caption issue.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md).

**Status:** addressed in m125, preserving m124.


The archived analysis workbook identifies the historical charts as v4.3 (2025-09-07), enrollment 2021_24, pseudo birth-year 1960 (all vaccination categories), comparisons 1/0 and 2/0. Neither chart has error bars. Its reference is 2021-07-12 (index 4). Historical final values are 0.986862 and 0.964170; historical curve intervals are not available.

The revised replay retains the same archived Table 8 input, birth-band mapping, enrollment and end date (2024-10-07), uses the current all-window joint fitter with the optional NPH module inactive, and anchors at 2021-07-19 (index 5, end of fourth accumulating week after two skipped weeks). Each of the 999 matched draws samples pooled disjoint history counts with the Table 8 seed and reconstructs the full-population strata. The three age groups are refitted in each draw. The original estimate and draws 1 and 42 match the ordinary workbook's direct curves at every plotted week to relative tolerance 1e-8. All 999 draws numerically succeed and have no insufficient-signal flags. The oldest group triggers the exploratory residual screen in the original fit; contributing fits trigger that screen in 987/999 and 950/999 draws for the 10- and 20-year comparisons. All retained, disclosed in both captions.

| Age shift | Historical endpoint (CI unavailable) | Revised endpoint [95% CI] | Plotted pointwise intervals containing 1 |
|---|---:|---:|---:|
| 10 years | 0.986862 | 1.003561 [0.947360, 1.053157] | 168/169 |
| 20 years | 0.964170 | 1.027059 [0.964180, 1.091413] | 159/169 |

The paper now states the conditional proportionality requirement for flatness. Both figure captions identify the estimator, ages, enrollment, reference, end date and pointwise bootstrap method. The previously corrected five-year birth-band labels remain unchanged. No new estimator or modification to Table 8.

Reproduce from candidate root: `python analysis/age_shift/replay.py --draws 999` then `python analysis/age_shift/plot.py`. Copies of the scripts, all draws, curves, diagnostic flags, summary, and old/new figures are here. [Paper patch](paper.patch) | [Supplement patch](supplement.patch) | [Summary](summary.json).
