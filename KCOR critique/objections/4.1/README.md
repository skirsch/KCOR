# 4.1 — Nonlinear fit eligibility

**High — new verified implementation defect**

A nonlinear fit with a marked wave covering all 12 originally quiet weeks has zero eligible rows but returns the preliminary proportional fit and status ok, n_fit=12, theta=2. This is a false-success reporting path, not an identification limitation.

**Root cause / related objections:** Related 1.2, 1.3, 3.2. Check eligibility after all exclusions; do not silently report the preliminary fit as a successful nonlinear fit.

**Source review:** [repeat-1/original-handoff.md](<../../iterative review/evaluations/4/repeat-1/original-handoff.md>) · [repeat-3/chain-current.md](<../../iterative review/evaluations/4/repeat-3/chain-current.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m95 (approved by user).** After nonlinear wave exclusion, fewer than three eligible weeks now return insufficient_data, success=False, the true eligible count and mask, and no fitted parameters. Tests for 0, 1 and 2 weeks reject the fit; 3 weeks fit successfully. All six Table 8 points and the first matched bootstrap draw reproduce m94 exactly. Full 999-draw rerun not performed; eligibility is unchanged for these cohorts. The paper already states the eligibility requirement, so no manuscript wording change was necessary.

[Patch](eligibility.patch) · [Verification](verification.json) · [Regression test](check.py) · [Corrected module](kcor_survivor.py). This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
