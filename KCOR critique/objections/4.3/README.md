# 4.3 — Wave deficit attribution and sampling bias

**High for optional nonlinear wave — verified statistical consequence**

At theta=1, known B=0.001, true W=0, N=5000, exact binomial enumeration of a marked wave week gives mean B+W 0.00098573 at alpha=1.18 and 0.00091248 at alpha=2. At 500k and alpha=1.18 the mean is 0.00099853. No evidence yet of the size of the cumulative cohort-ratio bias.

**Root cause / related objections:** Related 1.2. Exact survival reconciliation alone cannot establish correct baseline/wave attribution. Discuss whether the existing deficit convention is acceptable before any design change.

**Source review:** [repeat-1/chain-current.md](<../../iterative review/evaluations/4/repeat-1/chain-current.md>) · [repeat-2/chain-current.md](<../../iterative review/evaluations/4/repeat-2/chain-current.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

No correction adopted yet. This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.

## Signed-adjustment trial (not adopted)

The existing m95 solver was compared with an isolated alternative that solves for total nonnegative wave exposure each week, allowing a decrease from the preceding estimate. When even zero wave exposure exceeds observed survival loss, it resets W to zero and solves for baseline exposure. Both fitting and inversion used the corresponding attribution rule. The ordinary m95 source was not changed; the separately agreed 4.2 eligibility change was not yet integrated into this reference. The study uses the existing main design (53 weeks, 8 marked wave weeks, two strata, theta 4 and 0.5, alpha 1.18) with new matched random draws, fixed cohort sizes and fixed stratum weights 2:1. True KCOR is 1 for both zero-wave and actual-wave scenarios. These are not the original full coverage draws.

| Scenario | People/cohort | Existing mean KCOR [95% MC CI] | Signed mean [95% MC CI] |
|---|---:|---|---|
| Zero wave, known parameters | 12,500 | 1.02469 [0.99948,1.04990] | 1.02376 [0.99857,1.04894] |
| Zero wave, known parameters | 250,000 | 0.99959 [0.99438,1.00480] | 0.99938 [0.99417,1.00459] |
| Real wave, known parameters | 12,500 | 1.00538 [0.98103,1.02973] | same |
| Real wave, known parameters | 250,000 | 1.00365 [0.99824,1.00906] | same |
| Zero wave, fitted | 250,000 | 0.99318 [0.98183,1.00454] | 0.99301 [0.98166,1.00436] |
| Real wave, fitted | 250,000 | 1.00469 [0.99415,1.01523] | same |

100 independent data sets per known-parameter setting, 20 per fitted setting. In the zero-wave known-parameter 12,500-person setting, mean W across the four cohorts fell from 0.001107 to 0.000627 (43% reduction); true W=0. For the first fitted zero-wave dataset, KCOR 0.936175 [0.891361,0.976401] versus 0.936168 [0.891081,0.976001] (99 person-bootstrap draws). For the first fitted real-wave dataset, both 1.001740 [0.958609,1.043100]. No full coverage study. Fitted zero-wave mean theta (four cohorts) changed from 3.9413, 0.4903, 4.2299, 0.5959 to 3.9423, 0.4904, 4.2313, 0.5962. Real-wave fits were identical. No decreasing corrected cumulative hazard or baseline path was observed in these sampled curves; signed W reductions occurred frequently in zero-wave curves. That does not prove physically consistent survivor histories in general. A single marked week has no prior W to revise and therefore retains the original bias.

**Conclusion:** the trial reduces fictitious estimated wave exposure but changes endpoint bias only slightly and does not solve the one-week attribution objection. Do not adopt as a fix yet. [Experiment](signed_trial.py) · [Results, including all parameter intervals](signed-results.json) · [Paired draws](signed-draws.npz).

## Approved disposition

Treat the zero-wave test primarily as wave-window misspecification. Retain the existing algorithm and disclose the limitation in section 2.7.3, including the possibility of sampling-related attribution bias when excess mortality is weak. [Manuscript patch](wave-window-disclosure.patch). The signed adjustment is not adopted. This addresses the paper disclosure, not elimination of finite-sample bias. Candidate m96 preserves m95 and frozen m94.
