# 9.5 - Unexplained nondeath risk-set losses accepted

**Severity:** High

**Origin:** New specific defect

**Related issues / root causes:** 8.1, 7.2, 4.6.

**Finding:** Ordinary 56-week fixture: removing 12,000 people from each of two sex risk sets without recorded deaths changes true-null endpoint 1 to 1.45793, OK.

**Proposed scope (for discussion):** Check original distinction between death-only and declared exit-capable inputs; explicit reporting and identical point/bootstrap validation.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m146; m145 preserved. Independent closure not yet evaluated.

The ordinary fixed-cohort workbook and its person bootstrap default to death-only accounting. An unexpected loss fails source validation before aggregation, causing the affected age and dependent pools to be unavailable with invalid_risk_history; the independent age is unchanged. Explicit `KCOR_RISK_HISTORY_POLICY=allow_exits` preserves the already documented disjoint exit-capable sensitivity, labels output and logs exit counts. The overlapping Table 8 layout remains strictly death-only. The rounded Table 9 runner and Makefile explicitly select allow_exits. The altered fixture is rejected in default ordinary and bootstrap paths, whereas explicit opt-in reproduces 1.457933581 (deterministic point-only fixture, not an inferential estimate). All seven current Czech enrollment sheets satisfy accounting across the audited analysis age range, with zero invalid groups. The Table 9 benefit 0.834 [0.774, 0.897] and harm 1.162 [1.087, 1.248] estimates reproduce to <3.3e-10; the archived 999-draw intervals are retained, not newly refitted. No estimator change. Section 2.9.1 and Table 9 caption now identify the policy. The question of zero opening risk is separate 9.6. [Source audit](czech-history-audit.json) | [Table 9 verification](positive-control-verification.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).
