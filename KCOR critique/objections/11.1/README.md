# 11.1 - All-ages population depends on redundant input total

**Severity:** High for affected ordinary workbooks

**Origin:** New specific defect

**Related issues / root causes:** 9.1, 9.2, 10.1.

**Finding:** Default display-year filtering removes unknown years before reconstruction. Supplying a correct redundant -2 total changed anchored KCOR 1.177097 to 1.165209 in an equivalent generated input.

**Proposed scope (for discussion):** Construct the eligible aggregate before filtering age-specific outputs; verify representation invariance and matched bootstrap population.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Aggregate membership determined before display-year filter. Both input representations give exactly the same point and three matched ordinary bootstrap intervals: 0.999967 [0.956208, 1.013158] (smoke only). [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** partial; original unknown-year fixture resolved; post-2020 cutoff still breaks representation invariance (12.1). [Targeted review](../../iterative%20review/evaluations/12/resolution.md).
