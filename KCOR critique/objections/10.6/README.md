# 10.6 - Spawned bootstrap workers lose effective age grouping

**Severity:** High for affected workbook users

**Origin:** New specific defect

**Related issues / root causes:** 4.6, 9.8.

**Finding:** Windows five-year AGE_RANGE=5 point for 1940 receives finite draws from ten-year merged 1940+1945 group; 1945 gets missing draws. Dedicated S6 archive unaffected.

**Proposed scope (for discussion):** Transfer effective age grouping to workers, test distinct five-year groups and ensure corresponding keys represent the same people.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m153; m152 preserved. Independent closure not yet evaluated.

Workers inherit effective AGE_RANGE; five-year 1940 and 1945 groups each retain their own three matched draws. Independently sampled and refitted manual draws reproduce both written intervals. No default grouping change. [Batch patches and regression](../../iterative%20review/evaluations/10/post-review-batch-1/).


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).
