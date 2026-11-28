# 16.4 - Partial hidden population can have more cause-specific deaths than all deaths

**Severity:** Low input-integrity defect; no demonstrated all-cause numerical error

**Origin:** Newly identified pre-existing validation gap related to 15.3

**Related issues / root causes:** 15.3, 14.4.

**Finding:** A partial supplied-minus-displayed remainder has one COVID death among zero all-cause deaths while all supplied histories individually pass; full-minus-young complement also passes. Joint intervals already withheld.

**Proposed scope (for discussion):** Reuse the cause-specific count feasibility check for the partial remainder, not only the nested complement.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/16/blind-a.md) | [Blind B](../../iterative%20review/evaluations/16/blind-b.md) | [Resolution](../../iterative%20review/evaluations/16/resolution.md)

**Status:** Implemented in m178; targeted verification complete.


**Approved correction:** Cause-specific feasibility enforced in partial undisplayed remainder; review contradiction rejected and valid comparison retained. [Verification](../../iterative%20review/evaluations/16/straightforward-batch/README.md).

**Round 17 independent disposition:** Resolved in original scope. Hidden cause-specific feasibility checks verified. [Verification](../../iterative%20review/evaluations/17/resolution.md).
