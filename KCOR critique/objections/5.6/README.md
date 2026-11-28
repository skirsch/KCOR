# 5.6 - Round 5 finding

**Severity:** Potentially high for source cohorts

Converter's non-mRNA product exclusion examines later doses before constructing earlier enrollment cohorts. Verify original population intent and actual affected records.

**Related issues:** Fixed-cohort construction, new code-path issue

**Review evidence:** B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** addressed by clarification in m108; m107 preserved. Application-specific study design, not a defect of the general KCOR estimator.

The lead-author review should distinguish the selection of a study population from the construction of fixed cohorts within that population. The full-course product eligibility restriction belongs to the particular application, and post-enrollment information can create selection bias. The approved sentence in section 2.2 makes that distinction explicit; no restriction or converter behavior has changed. The particular application's population definition and its implications may still deserve evaluation, but do not constitute a general KCOR algorithm error.

[Paper patch](paper.patch). No numerical results changed; existing draft PR unchanged.
