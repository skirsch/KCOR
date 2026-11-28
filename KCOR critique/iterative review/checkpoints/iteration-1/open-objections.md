# Open objections to the corrected KCOR edition

The starting point for the next revision is the [evaluated paper and method](evaluations/2026-11-21/README.md). The archive pins the files, including the working-tree corrections through PR #5 item 6.2. The identifiers below let us discuss one discrepancy at a time. An entry records an objection, not approval of a proposed fix.

## Current disposition of PR #5

The revised paper and supplement now address the remaining wording in item 1 and items 6.1, 6.3 and 6.4. Section 2.4.4 identifies frailty from the level and derivative of the exact quiet-window hazard at fixed Gompertz slope. Section 2.6.2 uses the fitted Gompertz cumulative shape as the diagnostic reference. Historical Czech diagnostic results describe the checks originally performed.

**Item 2 remains open for the method.** Section 2.7.3 and SI §S2.1 now identify the mean-one gamma proxy and cumulative division as approximations during a nonlinear wave. The conditional post-wave equation still takes the resulting exit exposures as inputs. The precision and reporting decisions listed below remain deferred. The manuscript correction for **N1** is recorded below. The next individual evaluation finding for discussion is **N2**, the weekly quantity used in fitting.

### Findings in the evaluated edition before these edits

| Item | What remains in the evaluated edition | Where to review it |
|---|---|---|
| 1. Diagnostic claims | The revised main text allows contamination to go undetected and distortion in either direction. Some supplement and discussion wording still implies visible failure or conservative attenuation. Prognosis-dependent crossover also has no guaranteed bias direction. | §5, SI §§S2 and S6. [Evaluation findings](evaluations/2026-11-21/convergence.md) |
| 2. In-wave adjustment when α differs from 1 | The tail equation uses the baseline and wave exposures estimated at wave exit. During the wave, the implementation evaluates a moment of a new mean-one gamma distribution and divides accumulated excess by the current week's factor. The first calculation can give the wrong survivor moment, and the second lacks the required cumulative identity. The original review already raised both the surviving-population issue and the cumulative-versus-increment issue. The new evaluation supplied a concrete implementation counterexample, 1.5 versus 0.5. | §§2.7.1–3, SI §S2.1. [Original objection](../original-review/KCOR_Independent_Review_EN.md) · [Equation and code checks](evaluations/2026-11-21/checks.md) |
| 6.1. Scaling | The corrected sentence distinguishes scaling observed hazards from scaling individual baseline hazards. The abstract, discussion and SI assumption A3 still use broader invariance wording. | §2.4.4, abstract, §4.1, SI §S2. [Revised review](evaluations/2026-11-21/repeat-3/B.md) |
| 6.3. Identification from curvature | §2.4.4 equates positive frailty with nonzero curvature of observed cumulative hazard. A rising Gompertz baseline already gives curvature at zero frailty. A positive frailty variance can exactly offset that rise, giving a straight cumulative curve. | §2.4.4 proposition. [PR #5 shape checks](<../06 - Objection 6/objection6_shape_claims/README.md>) |
| 6.4. Shape after correction | The diagnostic asks whether corrected cumulative hazard is straight in calendar time. Under the stated model its reference shape is the Gompertz cumulative curve, which is straight in Gompertz-transformed time. | §2.6.2 and SI diagnostics. [PR #5 proposed wording](<../06 - Objection 6/PROPOSED_MANUSCRIPT_WORDING.md>) |

## Specific discrepancies to take up from the evaluation

“New” here means a distinct matter for the next revision discussion. Most were already present in the earlier edition. Several give a specific code or evidence problem behind a broader PR #5 concern.

| Item | What the paper, code or evidence actually does | Location |
|---|---|---|
| N1. Information given to KCOR in the Cox experiment | Corrected in §2.11.1. The generator supplied the true frailty variances. The manuscript now describes the experiment as an inversion with known variances. A fresh evaluation of this wording is pending. | [Evaluation finding](evaluations/2026-11-21/repeat-3/A.md) · [generator](../../test/sim_grid/code/generate_cox_bias_sim.py#L234) |
| N2. Weekly quantity used in fitting | The fitting objective compares integrated weekly observations to instantaneous model hazards. The approved exact weekly reconstruction applies after a value of frailty has been supplied. | §2.5 Steps 1 and 4; [revised fitter](revised/code/KCOR.py) |
| N3. Last week of a gap | The offset uses cumulative hazard at the start of the final gap week. A shock wholly in that week is omitted from the offset applied to the next quiet window. | §2.5 Step 3; [revised fitter](revised/code/KCOR.py#L1454) |
| N4. First week of a wave | The no-wave reference starts from the cumulative value at the end of the first wave week. This assigns that week's excess to baseline. The issue occurs at wave entry, upstream of the corrected post-wave equation. | §2.7.3; [wave calculation and check](evaluations/2026-11-21/repeat-3/A.md) |
| N5. Quiet-window robustness scan | SI §S5.4 says the scan used the revised estimator. Its script fits a constant-baseline model. Separately, the summary computes the pass rate after filtering to passing windows. One group passed 3 of 13 attempted windows and is summarized as 100%. | [Scan fitter and summary](../../test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py) |
| N6. Positive-control injection | The supplement describes an intervention on individual baseline hazard. The generator multiplies the already mixed cohort hazard. The two interventions imply different subsequent survivor composition. | SI §§S4.3–4; [positive-control generator](../../test/positive_control/code/generate_positive_control.py) |
| N7. Contamination threshold | The study has a valid equal-baseline null. What its output fails to establish is a threshold at which the estimate first reverses direction. In the comorbidity sweep, 15 of 20 estimates already exceeded one at zero added contamination. | §3.3.2; [PR #5 target audit](<../05 - Objection 5/INITIAL_TARGET_AUDIT.md>) · [checked counts](evaluations/2026-11-21/repeat-2/B-follow.md) |
| N8. Anchor definition | One passage uses a single reference week and another a mean over a reference window. The pipeline and Table 8 use one week. The choice determines which contrast a reader reproduces. | §§2.1.1 and 2.8; [follow-up](evaluations/2026-11-21/repeat-1/B-follow.md) |
| N9. Unit of fitting and pooling | §2.9.1 says strata are used only for aggregation and uncertainty. Table 8 fits each subgroup and pseudo-dose, then combines anchored ratios using weights based on early mortality. | §2.9.1, Table 8 and SI §S6; [follow-up](evaluations/2026-11-21/repeat-1/C-follow.md) |
| N10. Selection of empirical examples | The illustration selects on the final KCOR and on improvement in curvature. A Czech configuration comment cites observed harm in motivating window choice. The filter is verifiable; the comment alone cannot establish when the windows were chosen. | [Illustration selector](../../code/generate_kcor_empirical_intuition_figure.py) · [Czech configuration](../../data/Czech/Czech.yaml) |

### Supporting tests, figures and study provenance

- **Post-wave regression test.** One existing test fails on the corrected edition because it still requires the original uncorrected post-wave increments. The separate check for the new equation passes. This is a test maintenance item. [Verification](evaluations/2026-11-21/checks.md)
- **Other simulation evidence.** The Cox figure caption describes bootstrap bands while the plotting function draws a line and points. The recovery experiment uses noiseless counts and the initial generating rate. The historical grid and one empirical illustration use earlier procedures. The shared-frailty Cox curve is already disclosed as a calibrated proxy. Each result supports a different claim about the estimator. [Validation review](evaluations/2026-11-21/repeat-3/A.md) · [Comparator check](evaluations/2026-11-21/repeat-3/C-follow.md)
- **Older coverage pilot.** The target comes from a finite reference draw. Several generators produce deterministic expected counts across nominally repeated datasets. That pilot cannot calibrate the revised person bootstrap. Separate later studies produced 189/200 and 194/200; both counts have identified sources. [Study provenance](evaluations/2026-11-21/checks.md)

## Previously deferred decisions

- **Initial fitter and interval coverage (PR #5 item 3).** The joint first-window fit is the intended default, pending lead approval. Ordinary production still fixes the early baseline rate, and the final fit can include anchor weeks outside the paper's quiet-only mask. Coverage for the complete pooled, multiwindow analysis remains open. The person resampling unit is settled for Table 8. [Decision log](decisions.md) · [Paired fitter comparison](rounds/round-03-objection-3-person-bootstrap/paired-seed-fit/assessment.md)
- **Table 8 reporting (item 4).** All six revised points reproduced. The open question is how to report contrasts and intervals that include an insufficient-signal fallback under the paper's stated reporting rule. The age-shift exercise has no guaranteed exact null; intervals containing one do not establish equivalence. Interpretation of these results and the explicitly forced Czech exponent belongs with the project lead. [Checks](evaluations/2026-11-21/checks.md)

## Corrections retained in the next edition

Retain the proportional-wave identity at α = 1, the conditional post-wave continuation, the exact weekly reconstruction in 6.2, the person bootstrap, the Table 8 recalculation, the corrected frailty-attribution sentence, and the conditional treatment-effect null in the Cox comparison. The approved scaling sentence also stands. The [decision log](decisions.md) records the individual changes and verification.
