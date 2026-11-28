# 9.11 - Valid small frailty ceiling breaks supplementary fit start

**Severity:** Low

**Origin:** New specific defect

**Related issues / root causes:** 8.4.

**Finding:** Configured cap 0.25, truth theta 0.1: hardcoded second start 0.5 causes proportional error and marked-wave ValueError. Default 500 unaffected.

**Proposed scope (for discussion):** Clip/deduplicate starts to bounds, retain successful first fit, regress both modes.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m143; m142 preserved. Independent closure not yet evaluated.

Both supplementary starts are constrained to the configured bound; duplicate starts skipped and unsuccessful supplementary fits cannot erase successful primary fits. Exact theta 0.1 recovered under cap 0.25 in both paths; existing cap500 cases and 24 low-rate grid cases passed. [Batch patches and verification](../../iterative%20review/evaluations/9/post-review-batch-1/).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).
