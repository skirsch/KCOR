# 9.1 - Missing sex suffix in overlapping control

**Severity:** High

**Origin:** Direct return; direct predecessor 8.1

**Related issues / root causes:** 7.2.

**Finding:** Deleting a male suffix with 552,591 opening risk is accepted by ordinary validation; inclusive 2/0 1.027059 to 1.019856, weighted 1.050521 to 1.046511, OK. Bootstrap rejects.

**Proposed scope (for discussion):** Validate identifiable sex/DCCI histories before collapsing, preserve genuine exhaustion.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m144; m143 preserved. Independent closure not yet evaluated.

The overlapping layout now combines only indistinguishable rows, retaining sex and DCCI while checking every identifiable weekly history, including the death-only accounting rule already required by the overlapping bootstrap. The same validation precedes the dedicated Table 8 replay. The disjoint-layout exit policy is unchanged (separate 9.5). The deliberately missing nine-week male suffix is rejected by all three paths; affected inclusive, weighted and direct aggregate results are unavailable (missing_week_history); the unaffected ages remain 1.250497647 and 0.932343504. A valid constructed exhausted component was accepted by both ordinary and bootstrap validators. All four earlier incomplete-history cases and genuine exhaustion passed. Both original Czech sheets passed source validation and all 12 Table 8 points and calendar records matched the archived replay within 2.23e-16. The 999-draw archived intervals are retained (no full refit). Paper and PDF unchanged: Section 2.9.1 already specifies preaggregation component validation and truncated positive-risk handling. [Workbook verification](workbook-verification.json).

**Round 10 independent disposition:** partial; see [10.1](../10.1/README.md). [Resolution](../../iterative%20review/evaluations/10/resolution.md).
