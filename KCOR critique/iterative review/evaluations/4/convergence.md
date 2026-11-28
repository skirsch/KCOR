# Series 4 — convergence and objection dispositions

**Scope:** hypothetical post-approval integration of the draft PR #6 proposals (frozen m94). The real PR and official sources have not changed. The [verification](verification.md) shows the included model and data checks. The raw LLM assessments, follow-ups, [input manifest](inputs.json) and [exact overlay](edition-overlay.zip) remain separately available. This assessment distinguishes verified software/formula defects from known model assumptions, deliberate design choices, and validation limits.

## What converged

The reviewers agree that the all-window joint fit, proportional pre-entry correction and specified nonlinear survivor-law reconstruction are substantial improvements. They no longer repeat the previous objections that the pooled refit necessarily fixes k or that the nonlinear wave correction must continue assuming a mean-one gamma distribution. Direct known-truth testing, matched point estimates and the stored bootstrap draws support that conclusion. Adequacy of the final estimator under sparse data, incorrect window labels or an imperfect baseline model remains conditional.

All 18 current open assessments request major revision (previous series: 17/18 explicitly requested major revision). All 48 turns and 18 independent conversation identities passed the audit. [Specific findings by path](path-counts.md). Reviewers still request major revision. This does **not** mean the earlier mathematical corrections failed. Several criticisms recur because they concern interpretation of a model-dependent estimand, uncertainty calibration or a design choice rather than a fault in the gamma identity.

## Disposition of earlier objections

| ID | Current assessment | Reason |
|---|---|---|
| 1.1 | Partly resolved / interpretation limitation | Bias direction and undetectable contamination are now qualified; the observational-equivalence limitation is inherent, not a new arithmetic error. Some broad claims could still be narrowed. |
| 1.2 | Original algebra corrected; further issue | Exact conditional survivor inversion resolves the incorrect in-wave/exit moments at specified inputs. New branch continuity and finite-sample attribution issues are distinct (4.2, 4.3). |
| 1.3 | Person resampling and numerical instability substantially addressed; coverage/reporting open | Joint fit reduces pathological intervals, matched validation shows 90/100 nonlinear coverage; ordinary workbook's analytic intervals still differ from refitted bootstrap (4.6); initialization guard is a separate issue (4.5). |
| 1.4 | Control results reproduced; reporting/alpha open | Table 8 all intervals include one but empirical target unknown. Alpha still unidentified; new module-off comparison and configuration discrepancy (4.7). |
| 1.5, 2.1 | Original Cox disclosure corrected | Supplied true frailties and approximate comparator disclosed, intervals shown; benchmark is not fitted-method validation. Requests for a new comparison are validation suggestions, not a failure of the existing intended test. |
| 1.6, 2.2 | Substantive math errors corrected | Correct scaling, integrated weekly fit and inversion, fixed-slope proportional identification. Explicit nonlinear-history scope and extreme numerical cap reporting are narrower follow-ups. |
| 2.3, 2.4 | Corrected | Last gap week and first modeled wave week accounted for. New raw-offset clamp behavior does not undo the timing correction. |
| 2.5 | Corrected | All 78 attempts counted; new scan uses the consistent fitter and all pass. Parameter variation across placements is a scientific limitation, not another denominator fault. |
| 2.6, 3.3 | Corrected as direction-of-effect control | Generator description and fitted workbook rerun are correct; fitted variance under a misspecified baseline is a newly quantified limitation, not evidence of incorrect generator intent (4.8). |
| 2.7 | Claim corrected; statistical limitation remains | Small-cohort null bias is explained and quantified, larger-cohort companion recovers the expected null within uncertainty. Not a universal remedy for low information. |
| 2.8 | Corrected | Reference convention and historical sensitivity accurately disclosed. Anchoring cancels sustained proportional differences by definition; this is an intended change-of-relative-ratio estimand, not evidence of a mathematical error. |
| 2.9 | Fitting and pooling disclosure corrected | Incomplete/flagged-fit reporting remains a distinct issue. |
| 2.10 | Example selection disclosed | Existing caption explains outcome/curvature selection; configuration comment about the 2023 window is a separate provenance question (4.7). |
| 3.1 | Original reset corrected | Zero additional gap no longer automatically erases frailty. Raw incremental gap clamp is a separate issue (4.4). |
| 3.2 | Proportional entry corrected; nonlinear unlabeled scope remains | Known-truth and sampled experiments recover the fitted target with labeled exposure; not possible to infer every mixed wave from counts. |

## New specific findings and continuing issues

### High priority: verified new defects in the hypothetical integration

| ID | Issue | Severity | Verification |
|---|---|---|---|
| 4.1 | Nonlinear fit can return `ok` with zero eligible fitting rows | High | At alpha=1.18, 12 quiet rows all inside a marked wave yield zero eligible rows, yet the fallback reports n_fit=12, success and theta=2. Direct reproduction. |
| 4.2 | Fitted estimator discontinuity near alpha=1 | High for the optional module | At alpha=1 and 1.0000001 the same noiseless history returns theta 17.25177 and 2.00000, respectively; 54 versus 50 fit rows. The exact proportional identity is correct; the switch changes fitting data and exposure treatment. |
| 4.3 | Weekly wave/deficit allocation creates spurious wave exposure under ordinary sampling | High for the optional module | Exact binomial enumeration with known k, theta=1, true B=0.001 and W=0 yields mean B+W=0.00098573 at alpha=1.18, N=5k; alpha=2 gives 0.00091248. At N=500k and alpha=1.18 the mean is 0.00099853. This is statistical bias in attribution, not solver inaccuracy. No claim of the size of a final cohort-ratio bias in Czech data. |
| 4.4 | Positive increments from successive raw gap offsets can accumulate fictitious excess | Moderate / potentially high for many gaps | Raw cumulative offsets [0,-0.001,0] produce applied offsets [0,0,0.001]. Exact arithmetic matches the coded clamp. Magnitude on actual endpoint and full repeated-sample tests remain to be determined. Distinct from the old zero-reset correction. |
| 4.5 | All-window fit can still reject an informative later window before the pooled optimization | Moderate | With 33 noiseless quiet rows (two in the first segment, 31 later), the integrated fit returns `first_quiet_window_too_small` before attempting the pooled solution. |

[Reproduction code](verify_new.py) and [specific numerical counterexamples](counterexamples.json). These objections require discussion one by one. No fixes adopted during the review.

### Continuing objections with new concrete evidence

| ID | Issue | Severity and disposition |
|---|---|---|
| 4.6 | Reporting and uncertainty differ between ordinary workbook and validation pipeline | High for inference. An ordinary Table 9 workbook benefit interval is approximately [0.8144,0.8533], while the matched refitting bootstrap is [0.77412,0.89705] at the same point 0.83359. The ordinary workflow uses count-based intervals and can emit ratios despite insufficient-signal fallback. Related to 1.3/1.4, not a new general statistical objection. |
| 4.7 | Czech configuration and missing core comparison | High for interpretation, small configuration correction. YAML stores both `identified_alpha: 1.18` and `forced_alpha: 1.18` despite non-identification. The 2023-window rationale comment says it was chosen for least vaccine harm; chronology is unknown, so this warrants disclosure/verification rather than inferring actual bias. The [new module-off Table 8](verification.md#core-module-off-comparison-requested-by-reviewers) uses 999 person-bootstrap draws. All six intervals contain one without flags. |
| 4.8 | Positive-control misspecification and validation scope | Moderate for interpretation. Constant baseline generator with true theta=0.5 yields fitted theta=1.39250 [1.06539,1.70183] under positive Gompertz slope, despite successful diagnostics. This demonstrates expected conditional model dependence, not that the intended direction-of-effect control was invalid. Present the mismatch and retain correct original intent. |
| 4.9 | Remaining manuscript/configuration alignment | Minor, except if unsupported scientific claims are made. The workflow graphic still says k is fixed (a candidate-packaging omission); use of Nelson–Aalen for sum of -log(1-d/N) is imprecise; Table 8 caption should cite actual new result source; clarify the proportional-history scope of the identification identity after nonlinear selection. In extreme cases the existing inversion guard caps theta without prominent reporting; flagging/cap impact are separate checks. |

## Limitations, not newly demonstrated bugs

- **Frailty identification is model conditional.** Mortality alone cannot distinguish arbitrary time-varying baseline hazards from latent frailty. This was already acknowledged and is not solved by the all-window optimizer or theoretical information bound. The latter conditions on the working model and slope. The final targeted follow-up also gives an exact illustration: an unmarked additive shock shared by cohorts with different frailty violates the proportional model and can create a contrast even with perfect earlier quiet fits. This is a model-misspecification example, not a new arithmetic error. [Follow-up](repeat-3/chain-follow.md).
- **Anchoring intentionally removes the reference level.** All three blind reviewers note its sensitivity; the 0.782 versus 2.116 historical contamination example is already disclosed. Distinguish the anchored and unanchored estimands; do not redesign the experiment by shifting its reference.
- **Coverage is not universal.** 90/100 (82.4%,95.1%) in the nonlinear main test raises concern but does not by itself prove 95% undercoverage. Stress mean bias and sparse-event cases remain. This is distinct from numerical fit failures. The observed low-frailty information limit stays open as a statistical limitation.
- **Alpha is not identified.** The improved calculation does not establish a true exponent. Its formal cross-cohort external-scale procedure differs from the illustrative objective. Pairwise and collapse agreement is not independent evidence if their weights and samples make the losses algebraically equivalent; the exact historical weighting/data differences warrant attention if alpha work resumes. The user's decision is to defer alpha identification; don't add a new estimator here.
- **Control interpretation.** The empirical age-shift control has no guaranteed exact target, supplied-parameter Cox and synthetic checks test particular properties, and the positive control tests retrospective signal direction. Those are limitations of conclusions, not reasons to replace the intended tests.

## Root causes and priority for discussion

1. Fit eligibility and inconsistent branching (4.1, 4.2, 4.5).
2. One-sided reconstruction of time-varying exposure under noisy counts (4.3, 4.4); different from the previously confirmed low-frailty boundary bias.
3. Output/reporting versus actual pipeline (4.6, 4.7, 4.9).
4. Genuine limits of data and working-model adequacy (1.1, 1.3, 2.7, 4.8). No optimizer alone resolves these.

The revised candidate and all review inputs remain unchanged. The new checks are external adjudication, not subsequent revisions.
