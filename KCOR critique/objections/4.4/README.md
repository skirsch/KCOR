# 4.4 — Raw gap offsets and recovery from deficits

**Moderate, potentially high — arithmetic verified**

The profiled proportional objective clamps changes between successive raw offsets. Raw offsets [0,-0.001,0] imply applied offsets [0,0,0.001], accumulating positive exposure even when the raw signal returns to zero. Full-pipeline magnitude not measured.

**Root cause / related objections:** Related 2.3 and 3.1; distinct from the invalid frailty-reset safeguard. Preserve original intent and assess the cumulative offset arithmetic.

**Source review:** [repeat-1/original-handoff.md](<../../iterative review/evaluations/4/repeat-1/original-handoff.md>) · [repeat-1/evaluated-same.md](<../../iterative review/evaluations/4/repeat-1/evaluated-same.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m97.** All three proportional gap calculations now subtract the previously applied cumulative offset (iteration, joint residual and final reconstruction), retaining the nonnegative increment constraint and the signed diagnostic. The actual profiled code returns [0,0,0] for raw offsets [0,-0.001,0]. This fixes double accumulation, not all bias from nonnegative constraints.

[Regression checks](../4.4/check.py) · [Verification results](../4.4/verification.json). The candidate manuscript section 2.5 was also updated. Full Table 8 bootstrap rerun remains pending final pipeline validation; point/first-draw checks showed only small changes. This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
