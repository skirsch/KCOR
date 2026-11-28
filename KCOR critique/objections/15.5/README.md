# 15.5 - Illustrative S6/S5.6 figure replays do not assess signed gap assumptions

**Severity:** Minor to moderate diagnostic reporting

**Origin:** New related transparency issue

**Related issues / root causes:** 9.7, 13.5.

**Finding:** Figure replay reports numeric fit and residual flags but does not return the signed gap screen. The curves use unmarked gaps; no evidence of numerical change.

**Proposed scope (for discussion):** Explicitly state the signed-gap condition is unassessed for these illustrative figures; avoid implying an assessed pass.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/15/blind-a.md) | [Blind B](../../iterative%20review/evaluations/15/blind-b.md) | [Resolution](../../iterative%20review/evaluations/15/resolution.md)

**Status:** Implemented in m177; caption-only diagnostic disclosure.


**Approved correction:** Add the exact sentence "The signed-gap diagnostic was not assessed for these illustrative curves." to the three S5.6 birth-band captions and the S6.1.3 all-ages illustrative caption. No diagnostic result is invented, and no existing figure or interval is changed. [Patch and PDF verification](../../iterative%20review/evaluations/15/final-s6-wording/README.md).


**Round 16 independent disposition:** Resolved in the original scope by independent targeted round-16 review. [Resolution](../../iterative%20review/evaluations/16/resolution.md).
