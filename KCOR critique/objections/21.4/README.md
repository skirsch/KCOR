# 21.4 - Legacy method description and missing interval method fallback

**Severity:** Minor code documentation/reporting

**Origin:** Newly verified specific finding in round 21; not a demonstrated regression from round 20.

**Related issues / root causes:** 18.5, 20.3, 1.3.

**Finding:** build_kcor_rows docstring still describes slope removal and analytic count variance. Standalone summary defaults absent CI_method to analytic_count_based, inconsistent with the current refitted person-bootstrap policy.

**Proposed scope (for discussion):** Update docstring to actual procedure and mark absent interval method unavailable, without changing existing numerical output or interval method.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/21/blind-b.md)

**Status:** Open; correction not adopted.
