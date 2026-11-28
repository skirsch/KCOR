# 13.4 - Table 9 benefit control meets the gap-assumption audit-only rule

**Severity:** High for validation claim

**Origin:** New specific validation/diagnostic discrepancy

**Related issues / root causes:** 12.4, 9.7, 1.4, 2.6.

**Finding:** Benefit inserted before first quiet window yields signed gap -0.0227414 with matched 999-draw interval [-0.0422275,-0.0068647]; 997 negative draws. Existing anchored endpoint 0.834 [0.774,0.897] reproducible but existing rule qualifies interpretation. Harm and shared control gap intervals include zero.

**Proposed scope (for discussion):** Record gap diagnostic in paper-specific replay and qualify interpretation under existing rule; other validation designs require separate discussion.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in persistent m169; independent closure not yet evaluated.


Full matched 999-draw production replay: treated benefit offset -0.0227414 [-0.0422275,-0.0068647], harm +0.0210405 [-0.0008843,+0.0377747], common control -0.0005130 [-0.0192159,+0.0149087]. Anchored endpoints and intervals unchanged at displayed precision; benefit now marked audit-only stress test, no estimator or experiment changes.


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).
