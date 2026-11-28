# 20.3 - Baseline rate k is incorrectly called a gamma frailty parameter

**Severity:** Minor nomenclature in section 2.6

**Origin:** Newly verified specific finding in round 20.

**Related issues / root causes:** 18.5, 1.1.

**Finding:** The sentence calls both fitted theta and k gamma frailty parameters. k is the baseline hazard scale; the inversion removes curvature attributed to depletion and retains baseline level. The main equations and other passages already distinguish these.

**Proposed scope (for discussion):** Narrowly correct the description to curvature attributed to frailty depletion; no change to model, formulas, or meaning of baseline rate.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/20/blind-b.md)

**Status:** Implemented and verified in persistent m185; independent review pending.

**Approved correction:** Implemented and verified in m185. [Evidence](../../iterative%20review/evaluations/20/post-review-revisions.md).

**Round 21 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/21/resolution.md).
