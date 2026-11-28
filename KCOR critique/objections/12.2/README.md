# 12.2 - Identity wave branches miscount marked weeks and strict deficits

**Severity:** Moderate for optional wave diagnostic

**Origin:** Direct return of 11.14; nonlinear branch resolved

**Related issues / root causes:** 11.14, 11.5, 4.3, 5.10.

**Finding:** Alpha one returns zero marked weeks before counting; at zero frailty, cumulative attribution can mask a later weekly deficit. Three-week example has two strict deficits, not one zero-excess evaluation. Cumulative identity remains correct.

**Proposed scope (for discussion):** Preserve cumulative identities; count eligible marked weeks and apply exact weekly attribution to identity cases or explicitly mark unavailable diagnostics.

**Review evidence:** A, R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m163; m162 preserved. Independent closure not yet evaluated.


Weekly proportional identity attribution now compares each observed integrated hazard with the planned conditional baseline at that week opening. The total reference exposure curve is returned unchanged. Three marked weeks in the exact control recover two strict deficits, two zero-excess weeks, and wave exposure 0.01995905 for theta=0/alpha=1.18, theta=.5/alpha=1, and theta=0/alpha=1. Equality is not a strict deficit. Missing baseline gives known marked-week denominator with unavailable counts. Nonlinear tail continuity regression passes. No new full forced-alpha refit; primary inactive analysis unaffected.

Written ordinary workbook with forced alpha=1 and three marked weeks: all six dose/cohort rows export mode identity, wave_points=3, deficit_weeks=2 and zero_excess_weeks=2.


**Round 13 independent disposition:** partial; original strict-deficit and denominator examples pass, exact equality rounding remains (13.1). [Targeted review](../../iterative%20review/evaluations/13/resolution.md).
