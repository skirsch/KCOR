# 4.9 — Remaining paper and configuration alignment

**Minor — specific text/figure issues**

The frozen candidate retains a fixed-k workflow graphic although the objective is joint; sum -log(1-d/N) is not exactly Nelson–Aalen; Table 8 caption should cite the actual updated result file; the fixed-slope identity requires proportional survivor history. The extreme-H numerical cap should be disclosed if active.

**Root cause / related objections:** Some are candidate-integration omissions rather than original KCOR defects. Correct the descriptions and source links, not the core estimator.

**Source review:** [repeat-3/blind.md](<../../iterative review/evaluations/4/repeat-3/blind.md>) · [repeat-1/blind-targeted.md](<../../iterative review/evaluations/4/repeat-1/blind-targeted.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

Corrected in m102. The paper and regenerated workflow now show joint fitting and the consistent nonlinear survivor law; the zero-frailty limit names the implemented discrete log-survival sum; the fixed-slope proposition requires proportional survivor history; Table 8 names the exact verified v4.3 commit and retains the current JSON sources. The extreme-exponent numerical cap is disclosed as an approximation, not represented as an exact inverse. No estimator or diagnostics changed; automatic cap-activation flagging is not implemented by this documentation correction. See [paper patch](paper.patch), [workflow patch](workflow.patch), and [figure](fig_kcor_workflow.png). This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
