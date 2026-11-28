# 7.7 - Supplementary Figure S10 uses an unmatched trajectory and time range

**Severity:** Moderate for empirical figure provenance

**Origin:** New reporting/reproducibility defect

**Related issues / root causes:** 6.4, 6.10.

**Finding:** Figure S10 shows calendar dates from July 2022 to autumn 2024, despite a 2023-only and weeks-since-enrollment caption. Independent current fit puts dose 3/0 below unity at end-2023 and final date, whereas the plotted curve is above unity. Companion age-band historical figures have not been replayed.

**Proposed scope:** Trace the existing figure's exact inputs and settings or regenerate from the documented current specification with matched uncertainty if appropriate; correct dates/axis units and verify companion figures. No effect on the core equations.

**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m133; all four figures regenerated with current estimator and 999 person-bootstrap draws. No core method changes.

Original intent verified in commit fa6fc24a: illustrative end-to-end worked example, same 2022-26 fixed enrollment, dose 2/0 and 3/0, three narrow birth bands and all ages. The original fit log records KCOR6 fits and a cached longer-window workbook, not the current all-window fit. Original figures retained under before/. New figures use the supplied product-restricted 2022_26 CMR sheet, 2022-26 through 2024-41, two skipped weeks, four effective weeks to reference (2022-08-01), configured quiet windows 2023-19..40 and 2024-11..14, early anchors, NPH off. Fixed-person counts were checked for consistency; disjoint birth strata sampled once per draw and combined to preserve matching of all-ages draws. Every all-ages count matched supplied aggregate. All 12 point fits accepted; residual flags for 1940-1944 dose 3 and all-ages doses 0 and 3. All 8 endpoints include 1 in their pointwise 95% interval. This is descriptive only.

| Birth band | Comparison | End 2023 [95% CI] | Oct 7, 2024 [95% CI] |
|---|---|---|---|
| 1930-1934 | 2/0 | 0.915 [0.808, 1.061] | 0.895 [0.787, 1.035] |
| 1930-1934 | 3/0 | 0.921 [0.828, 1.029] | 0.907 [0.818, 1.014] |
| 1940-1944 | 2/0 | 1.052 [0.940, 1.175] | 1.043 [0.928, 1.167] |
| 1940-1944 | 3/0 | 1.007 [0.927, 1.114] | 0.979 [0.902, 1.088] |
| 1950-1954 | 2/0 | 1.051 [0.920, 1.199] | 1.048 [0.936, 1.190] |
| 1950-1954 | 3/0 | 0.989 [0.886, 1.109] | 0.998 [0.898, 1.117] |
| All ages | 2/0 | 0.983 [0.945, 1.026] | 1.001 [0.966, 1.040] |
| All ages | 3/0 | 0.973 [0.944, 1.003] | 0.976 [0.948, 1.004] |

[Reproduction script](evidence/replay.py) | [Summary](evidence/summary.json) | [Supplement patch](supplement.patch). Raw trajectories and draws are in evidence/. The all-ages endpoints reproduce the independent reviewer results to 5 decimals.

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).
