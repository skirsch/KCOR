# 7.2 - ISO enrollment and missing-week calendar handling

**Severity:** High

**Origin:** New implementation defect

**Related issues / root causes:** 4.6, 6.9.

**Finding:** Both blind reviews independently reproduced year-dependent one-week enrollment displacement and a missing row silently shifting the reference date. A complete 2025_24 workbook drops June 9 and anchors July 21 rather than July 14; a complete 2020_24 workbook drops June 8 and anchors July 20 rather than July 13. A missing internal/reference week in 2021_24 produces a later reference labeled OK. Complete Czech 2021 and 2022 inputs avoid the demonstrated ISO-year defect.

**Proposed scope:** Construct ISO dates correctly, derive elapsed time from dates and explicitly check missing histories/reference weeks. Preserve the intended two-week skip and four accumulating-week reference.

**Review evidence:** [Blind A](../../iterative%20review/evaluations/7/blind-a.md) | [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m129, preserving m128 (approved batch of straightforward corrections).


ISO enrollment uses datetime.fromisocalendar, and event time uses elapsed weeks rather than per-group row counts. Incomplete fixed-cohort weekly histories are flagged and unavailable, including synthesized aggregates depending on a missing component. Other cohorts remain valid. The reference itself remains the fourth accumulating week after the two skipped weeks. Regression tests preserve all 20 weeks for 2020_24, 2025_24 and 2021_24, with references July 13, July 14 and July 19 respectively. Removing the actual reference row flags the affected cohort, pooled and derived aggregate, without shifting the reference.

[Combined batch code patch](../../iterative%20review/evaluations/7/post-review-batch-1/workbook.patch) | [Paper patch](../../iterative%20review/evaluations/7/post-review-batch-1/paper.patch) | [Tests](../../iterative%20review/evaluations/7/post-review-batch-1/tests.patch)

**Round 8 independent disposition:** partial; remaining concrete issue tracked in [8.1](../8.1/README.md).
