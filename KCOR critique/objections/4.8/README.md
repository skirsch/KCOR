# 4.8 — Positive-control fitted frailty under misspecification

**Moderate — new quantified validation limitation**

The positive-control generator holds individual baseline constant; the fitted baseline imposes Gompertz growth. True control theta=0.5, fitted 1.39250 [1.06539,1.70183], no flag. The original intended signal-direction control is still valid as such. It does not validate parameter recovery under misspecification.

**Root cause / related objections:** Related 2.6, 3.3, 1.1. Distinguish validation scope from redesign of the intended test.

**Source review:** [repeat-3/original-same.md](<../../iterative review/evaluations/4/repeat-3/original-same.md>) · [repeat-2/evaluated-handoff.md](<../../iterative review/evaluations/4/repeat-2/evaluated-handoff.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

Accepted in m101: one sentence in ?3.2 clarifies the baseline misspecification and the directional validation scope. No generator, fitter, or numerical result changed. The original intent is documented in the original paper ?3.2 and the generator docstring. Table 9 retains benefit 0.834 [0.774, 0.897] and harm 1.162 [1.087, 1.248]. See [paper patch](paper.patch). This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
