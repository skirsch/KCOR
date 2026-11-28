# 4.2 — Fitted discontinuity at alpha=1

**High — new verified optional-module defect**

The survival identity is exact at alpha=1, but the integrated fitter switches masks and proportional-offset handling. In a noiseless case with true theta=2, the alpha=1 route gives theta=17.25177; at alpha=1.0000001 theta=2.00000. The first route uses 54 rows, the second 50.

**Root cause / related objections:** Related 1.2 and 3.2. Keep marked-wave fit eligibility and exposure conventions consistent across the exponent sensitivity curve; the no-wave module-off comparison is a separate choice.

**Source review:** [repeat-1/original-same.md](<../../iterative review/evaluations/4/repeat-1/original-same.md>) · [repeat-1/evaluated-handoff.md](<../../iterative review/evaluations/4/repeat-1/evaluated-handoff.md>) · [repeat-3/chain-current.md](<../../iterative review/evaluations/4/repeat-3/chain-current.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m97.** A specified proportional wave now uses the same marked-wave fit mask and exposure reconstruction as a nonlinear wave. At alpha=1 the survival law uses exact gamma inversion; the module-off route remains unchanged. In the noiseless regression, alpha=1 and 1.0000001 both recover theta=2 with 50 eligible weeks. 4.1 eligibility regression also passed at both exponents.

[Regression checks](../4.4/check.py) · [Verification results](../4.4/verification.json). The candidate manuscript section 2.5 was also updated. Full Table 8 bootstrap rerun remains pending final pipeline validation; point/first-draw checks showed only small changes. This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
