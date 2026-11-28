# Independent scientific and statistical review of m167

I reviewed the candidate manuscript, supplement, production fitter and reporting code. I checked two discrepancies against the ordinary workbook entry point or the archived generating counts. The proportional gamma inversion and the fixed-Gompertz-slope identification equation in §2.4 are algebraically consistent under the stated survivor model. Neither finding below requires replacing that model.

## 1. An unavailable reference can erase earlier valid unanchored estimates

**Moderate severity.** The unanchored ratio in §2.1.1 uses mortality accumulated through the date being reported. It has no dependence on the presentation reference. Sections 2.6.1 and 2.8 promise to preserve supported earlier estimates and to distinguish the two estimands. In `code/KCOR.py`, `_flag_unavailable_normalization` (lines 4305–4344) applies a single mask to `KCOR_unanchored` and the anchored columns. When the hazard first becomes infinite on or before the reference, the `numerical_range_unavailable` branch extends that mask to earlier dates. The unanchored bootstrap also requires a finite original point, so it cannot populate intervals for those dates.

I ran `process_workbook` with a complete, death-only 85-week history starting 2022-06-27, birth years 1940 and 1950, and doses 0 and 2. Each age-dose cohort started with 10,000 people. Each week 15 died in dose 0 and 16 in dose 2, except that everyone remaining in the 1940 dose-2 cohort died in week index 60. I used the module-off Czech fit, a two-week skip, and the prespecified reference at week index 60. Both 1940 fits had status `OK`. On 2023-08-14, one week before exhaustion, the program reported adjusted cumulative hazards of 0.097721 and 0.091452, a ratio of about 1.069. The same row reported `KCOR_unanchored = NaN` and `normalization_status = numerical_range_unavailable`. The defect also reproduced with the default reference and exhaustion in week index 5.

**Minimal fix.** Apply date-specific support masks to the direct unanchored point and its interval. Keep the reference-validity mask on the anchored point, its interval and any anchored pool. A regression case should verify availability of the unanchored prefix and unavailability from the first unsupported week, with the anchored series unavailable throughout. This example tests exhaustion; it does not establish a change to any published Czech endpoint.

## 2. Table 9 reports an anchored endpoint under the unanchored estimand's name

**Moderate reporting error.** Section 2.1.1 defines `KCOR(t)` as the unanchored cumulative ratio and says anchoring will be explicit (`paper.md`, line 169). Table 9 labels its result “Observed KCOR(t) at week 80” and §3.2 does not identify the reference. The replay divides each cohort's fitted curve by its own week-index-5 value before forming the dose comparison (`analysis/positive_control/replay.py`, lines 101–106). Week index 5 is the end of the fourth accumulating week after a two-week skip.

| Scenario | Week-80 unanchored ratio | Reference ratio | Week-80 anchored ratio shown in Table 9 |
|---|---:|---:|---:|
| Benefit | 0.831143 | 0.997061 | 0.833593 |
| Harm | 1.157454 | 0.996344 | 1.161701 |

I calculated both ratios from the archived workbook's dose-specific weekly deaths and opening risks and the fitted variances recorded in `analysis/positive_control/results.json`. Its bootstrap code applies the same anchoring within each draw. Both point estimates move in the intended direction under either definition. The reported intervals quantify the anchored statistic.

**Minimal fix.** Label the table column and §3.2 result `KCOR(80; t₀)` and identify `t₀` as the end of the fourth accumulating week. This preserves the existing numerical results and avoids assigning their intervals to the primary unanchored estimand.
