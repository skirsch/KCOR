# 6.4 ? Historical SI S6.1.2 numbers lack a reproducible matching configuration

**Severity:** High for empirical example

Two supplied workbooks give cumulative hazards 0.0202043 and 0.0740106, ratio 0.272992, versus printed 0.023160 and 0.073323, ratio 0.3159, for the stated inclusive range. The reported historical variance also differs from the current fitter. Locate the precise historical configuration before changing results or label the illustration accurately.

**Related issues:** 1.4, 5.12.

**Origin:** New reproducibility discrepancy; source configuration remains unverified.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** corrected in m120 with a reproducible replay of the three supplementary tables. m119 is preserved.

Independent current-tree check (m119): summing exact weekly integrated hazards from the supplied -2 rows over ISO weeks 2021-24 through 2024-16 inclusive gives 149 weeks per dose. Both `data/Czech/KCOR_CMR.xlsx` and `data/Czech/reference_without_COVID_correction-1-24-26/KCOR_CMR.xlsx` give dose 0 = 0.02020432296966776, dose 2 = 0.07401056258700445, and ratio = 0.27299242518142197. These differ from S6.1.2's 0.023160, 0.073323, and 0.3159. The exact historical input/configuration is still to be established. The separate S6 all-Yes diagnostic-gate claim will be checked with this item.

## Historical provenance

The three tables first appear in commit 6b727556 (2026-01-11). The archived CMR counts in that commit reproduce the current raw full-population values exactly. Combining the five-year birth bands into age groups as of calendar year 2024 and omitting two stabilization weeks reproduces the six published age-stratified raw rows closely. The known-birth-year-only total after the same skip yields 0.02315802 and 0.07330423, with a ratio rounding to 0.3159. This nearly reproduces the published total. The exact earlier computation was not located; the contemporaneous fitted workbook uses a different 2023-only quiet-window configuration and does not reproduce the printed frailty table. [Historical comparison](historical-provenance.json).

## Current replay

The replay in `analysis/s6/replay.py` uses the supplied current CMR sheet 2021_24 through week 2024-16, age in calendar year 2024, two skipped weeks, and the current all-window fitter with the optional NPH module off. The supplied full aggregate includes unknown ages. It uses the ordinary workbook processing entry point and records the actual fitted parameters, status, count residual flag and shape diagnostics. All fourteen fits were numerically accepted and six triggered the exploratory residual screen. The separate all-Yes claim is replaced with these measured results; window-omission stability is explicitly untested here. The six age-specific raw ratios change only slightly; the full-population ratio is 0.2710 after the skip. The updated frailty estimates are descriptive fitted values, not interval estimates. No new bootstrap was performed.

[Replay](replay.py) · [Results](results.json) · [Supplement patch](supplement.patch). To reproduce from the candidate root run `python analysis/s6/replay.py`.

## Frailty precision follow-up

[40-49 targeted investigation](frailty-resolution.md) and [all-age matched precision comparison](all-age-precision.md) record 999 matched draws per group for both fitting scopes and the local information limits. The youngest dose-2 raw variance spans [0,100]; the all-window fitter reduces its model-conditional local SE by about 31-fold relative to first-window fitting, but absolute resolution remains poor. The reference period has only 13 dose-2 deaths, and [the anchor diagnostic](anchor-precision.json) shows why the associated KCOR interval is broad even when fitted frailty is held fixed. Related to 2.7 and the remaining 5.11 fallback discussion.

**Approved reporting update (m121):** Table S7 includes raw frailty percentile intervals and anchored KCOR percentile intervals for all seven groups. The 40-49 dose-2 variance is displayed as poorly resolved [0,100], with its point estimate retained in the supporting results. The 50-59 and 60-69 dose-2 boundary estimates are explicitly qualified. The note distinguishes the raw constrained parameter bootstrap from inference about the underlying frailty and discloses the existing fallback for the youngest cohort. [Interval patch](intervals-supplement.patch) · [Previous point-only PDFs](checkpoints/m120/). Existing estimator and bootstrap fallback policy unchanged.
