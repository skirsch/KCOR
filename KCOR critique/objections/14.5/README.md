# 14.5 - Optional negative-control report substitutes earlier available endpoint

**Severity:** Moderate for affected ancillary report

**Origin:** New specific reporting defect

**Related issues / root causes:** 13.6, 11.2, 12.8.

**Finding:** Requested January 24 after exhaustion silently replaced with January 10; output omits date. Same writer also rejects valid tiny reference.

**Proposed scope (for discussion):** Use exact requested date and shared finite-ratio helper; report unsupported endpoint unavailable.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Optional negative-control workbook uses the exact requested reporting date, retains unsupported dates, exports ReportDate and endpoint_status, and uses the shared finite quotient. Integrated exhaustion test reports the requested date, unavailable, and blank KCOR rather than substituting an earlier value.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).
