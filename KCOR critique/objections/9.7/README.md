# 9.7 - Protective proportional gap is clipped to zero excess

**Severity:** High

**Origin:** New counterexample to known restriction

**Related issues / root causes:** 3.1, 4.4, 2.3.

**Finding:** Exact gamma theta 4, 60-week gap multiplier 0.3: fit theta 2.25257; true anchored endpoint 0.661664 vs fitted 0.588477, status OK. Multipliers 1 and 1.7 recover truth.

**Proposed scope (for discussion):** Verify original nonnegative-gap intent; distinguish signed cumulative excess from nonnegative total exposure; discuss flag/applicability or signed-gap change with author before altering estimator.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m148 as a diagnostic-only correction; m147 preserved. Independent closure not yet evaluated.

## Original-intent verification

The original version-7.5 specification (commit c70375db, March 27, 2026, section 2b) defines delta as excess exposure from elevated epidemic-wave mortality. It explicitly says for negative provisional delta: "Do not clamp and continue" because the post-wave cumulative exposure and fitted theta would be systematically wrong. Original fallback was later replaced by an insufficient-signal/zero-theta rule (4233f1ce and the insufficient_signal_switch plan). The round-3 counterexamples showed that even a truly zero offset can activate that safeguard due to numerical error; the approved replacement continues the pooled fit and separately checks whether a signed first-pass offset's 95% percentile interval lies entirely below zero. That diagnostic, tested on 45 historical Table 8 intervals, flagged none. The prior 4.4 correction removed cumulative double counting while intentionally retaining the nonnegative wave-offset constraint.

The round-9 counterexample is different: the generating proportional gap genuinely has mortality below baseline, not an elevated-mortality wave and not sampling error. Independent reproduction on m147 gives gap multiplier 0.3, fitted k=0.00200068235, fitted theta=2.25256719 (truth 4), signed raw offset -0.0924684311, target 0.6616641832, fitted KCOR 0.5884770772, numerical status ok and clamp flag true. Multipliers 1 and 1.7 recover endpoints exactly. These are deterministic expected-hazard examples, not sampling intervals.

Recommendation for discussion: preserve the original nonnegative-excess wave model and the user's earlier decision not to redesign the fitter, but explicitly identify material negative excess as model/applicability failure rather than a successful fit. The existing separately measured uncertainty diagnostic can distinguish sampling variation from a material deficit; production reporting and manuscript need to distinguish numerical convergence from adequacy of the nonnegative-gap assumption. A signed proportional-gap trial would be a separate methodological extension requiring lead-author approval, and is not applied. No candidate change yet.

## Implemented and verified

The current proportional fitter captures signed differences between consecutive raw cumulative offsets at the actual first pass. Ordinary bootstrap draws retain these values; complete pointwise percentile intervals with upper endpoint below zero mark gap_assumption_status=violation and flag affected contrasts while preserving their numerical estimates and intervals for audit. Incomplete intervals are not a pass. Numerical convergence is separate. Exact protective, null and elevated-gap smoke tests and an actual 19-draw written workbook passed. The written workbook protective gap interval was [-0.054542,-0.054104], flagged, with audit endpoint 0.767199 [0.760532,0.772767]. These smoke tests are not universal calibration results. The fresh 999 matched-person-draw primary Table 8 replay produced all 54 complete offset intervals and zero violations; all six point estimates and percentile bounds were identical to the earlier ceiling-500 archive. One prior bootstrap draw (2022_06 draw 806, max difference 0.00019315) changed at m143 due to 9.11 optimizer safeguard, independently verified in m142/m143 and m147/m148. Fresh arrays and earlier archives are retained separately, and printed values/intervals unchanged. The forced-alpha column is not covered by the proportional-gap diagnostic and its older interval archive is retained; no fresh full forced-alpha refit. Paper Section 2.5 and empirical validation paragraph updated. No signed-gap estimator or reset policy change. [Matched replay](table8/summary.json) | [Offset intervals](table8/gap-diagnostics.csv) | [Regression](diagnostic-regression.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).
