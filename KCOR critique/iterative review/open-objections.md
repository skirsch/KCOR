# Open objections to the corrected KCOR edition

**Latest: [Round 7 convergence](evaluations/7/convergence.md).** Six new and five returning issues, indexed [7.1](../objections/7.1/README.md)-[7.11](../objections/7.11/README.md). Earlier sections below retain historical dispositions. The round-6 candidate m128 remains frozen; no fixes or PR updates were made during the evaluation.

The starting point for the next revision is the [evaluated paper and method](evaluations/2/README.md). The archive pins the files, including the working-tree corrections through PR #5 item 6.2. The identifiers below let us discuss one discrepancy at a time. An entry records an objection, not approval of a proposed fix.

## Multi-edition evaluation of iteration 2

[Series 3 dispositions and comparisons](evaluations/3/convergence.md) distinguish the resolved N1?N6 and N8?N10 discrepancies from new or continuing limitations. N7 remained open at that evaluation. [3.1 ? No-shock gap fallback](../objections/3.1/README.md), [3.2 ? First quiet-window rate](../objections/3.2/README.md), and [3.3 ? Positive-control pipeline](../objections/3.3/README.md) index the specific follow-ups. The broader Table 8 coverage and reporting decisions remain deferred. No method changes were made for this evaluation.

Iteration 3 addressed the [3.1 gap reset](../objections/3.1/verification.md) and [3.3 Table 9 fitted provenance](../objections/3.3/README.md). [3.2](../objections/3.2/README.md) is deferred. Series 3 has no further numbered objection. [2.7](../objections/2.7/README.md) was addressed for the comorbidity threshold in the manuscript. Low-variance recovery remains deferred.

## Current disposition of PR #5

The revised paper and supplement now address the remaining wording in item 1 and items 6.1, 6.3 and 6.4. Section 2.4.4 identifies frailty from the level and derivative of the exact quiet-window hazard at fixed Gompertz slope. Section 2.6.2 uses the fitted Gompertz cumulative shape as the diagnostic reference. Historical Czech diagnostic results describe the checks originally performed.

**Item 2 remains open for exposure during a nonlinear wave.** The first modeled week now uses the opening survivor moment under baseline-only selection and accounts for the scale of the preceding gamma inversion. The opening factor approximates a full week. Later weeks still use the mean-one proxy and cumulative rescaling. The post-wave equation conditions on the resulting exit exposures. [Wave-entry correction and paired replay](../objections/1.2/entry-moment/README.md). The precision and reporting decisions listed below remain deferred. Iteration 1 is preserved as a [checkpoint](checkpoints/iteration-1/revised/). Iteration 2 has applied **N2** through **N6** and **N8–N9**. Iteration 3 corrected the contamination threshold claim in **2.7**. Both wording changes in **2.10** have been applied. Three fresh reviewers confirmed the latest supporting-evidence disclosures on targeted follow-up. The next deferred question is why low frailty is difficult to recover, using the [simulation diagnostic](../objections/2.7/README.md#why-the-zero-contamination-mean-exceeded-one) and the [Table 8 replay](../objections/1.4/table-8-replay.md#iteration-2-check-against-the-low-frailty-simulation).

### Findings in the evaluated edition before these edits

| Item | What remains in the evaluated edition | Where to review it |
|---|---|---|
| 1. Diagnostic claims | The revised main text allows contamination to go undetected and distortion in either direction. Some supplement and discussion wording still implies visible failure or conservative attenuation. Prognosis-dependent crossover also has no guaranteed bias direction. | §5, SI §§S2 and S6. [Evaluation findings](evaluations/2/convergence.md) |
| 2. In-wave adjustment when α differs from 1 | The tail equation uses the baseline and wave exposures estimated at wave exit. During the wave, the implementation evaluates a moment of a new mean-one gamma distribution and divides accumulated excess by the current week's factor. The first calculation can give the wrong survivor moment, and the second lacks the required cumulative identity. The original review already raised both the surviving-population issue and the cumulative-versus-increment issue. The new evaluation supplied a concrete implementation counterexample, 1.5 versus 0.5. | §§2.7.1–3, SI §S2.1. [Original objection](../original-review/KCOR_Independent_Review_EN.md) · [Equation and code checks](evaluations/2/checks.md) |
| 6.1. Scaling | The corrected sentence distinguishes scaling observed hazards from scaling individual baseline hazards. The abstract, discussion and SI assumption A3 still use broader invariance wording. | §2.4.4, abstract, §4.1, SI §S2. [Revised review](evaluations/2/repeat-3/B.md) |
| 6.3. Identification from curvature | §2.4.4 equates positive frailty with nonzero curvature of observed cumulative hazard. A rising Gompertz baseline already gives curvature at zero frailty. A positive frailty variance can exactly offset that rise, giving a straight cumulative curve. | §2.4.4 proposition. [PR #5 shape checks](<../objections/1.6/objection6_shape_claims/README.md>) |
| 6.4. Shape after correction | The diagnostic asks whether corrected cumulative hazard is straight in calendar time. Under the stated model its reference shape is the Gompertz cumulative curve, which is straight in Gompertz-transformed time. | §2.6.2 and SI diagnostics. [PR #5 proposed wording](<../objections/1.6/PROPOSED_MANUSCRIPT_WORDING.md>) |

## Specific discrepancies to take up from the evaluation

“New” here means a distinct matter for the next revision discussion. Most were already present in the earlier edition. Several give a specific code or evidence problem behind a broader PR #5 concern.

| Item | What the paper, code or evidence actually does | Location |
|---|---|---|
| [2.1 · N1. Information given to KCOR in the Cox experiment](../objections/2.1/README.md) | Corrected in §2.11.1. The generator supplied the true frailty variances. Series 3 reviewers confirmed the known-variance disclosure. The revised figure build reproduced all seven KCOR points and their interval endpoints in iteration 3. | [Evaluation finding](evaluations/2/repeat-3/A.md) · [generator](../../test/sim_grid/code/generate_cox_bias_sim.py#L234) · [figure verification](../objections/2.1/README.md#kcor-error-bars-in-the-cox-experiment) |
| [2.2 · N2. Weekly quantity used in fitting](../objections/2.2/README.md) | Series 3 confirmed the integrated weekly calculation in the core fit and diagnostics. Iteration 3 extended it to the optional wave reference in §2.7.3 and SI §S2.1. | [Core fit verification](../objections/2.2/verification.md) · [Wave reference and Table 8](../objections/2.2/wave-reference/README.md) |
| [2.3 · N3. Last week of a gap](../objections/2.3/README.md) | Corrected in iteration 2. Step 3 reads the reconstructed and expected cumulative hazards at the start of the next quiet week. The boundary check and Table 8 replay compare against the N2 edition. The series 3 targeted reviews confirmed this specific correction. | §2.5 Step 3 · [Boundary checks and replay](../objections/2.3/verification.md) |
| [2.4 · N4. First week of a wave](../objections/2.4/README.md) | Corrected in iteration 2. The no-wave reference starts immediately before the first modeled wave week and accumulates the fitted increment for that week. Replayed against the N3 edition. The series 3 targeted reviews confirmed this specific correction. | §2.7.3 and SI §S2.1 · [Boundary checks and Table 8](../objections/2.4/verification.md) |
| [2.5 · N5. Quiet-window robustness scan](../objections/2.5/README.md) | Corrected in iteration 2. All 78 attempted fits are retained; each of the six groups passed in 13/13 placements with the revised joint fitter. Fitted variances varied across placements. The author deferred interpretation of the figure until after the objection review. The series 3 targeted reviews confirmed this specific correction. | SI §S5.4 · [Results and replay](../objections/2.5/verification.md) · [Figure and discussion note](decisions.md#deferred-discussion-after-the-objection-review) |
| [2.6 · N6. Positive-control injection](../objections/2.6/README.md) | Corrected in iteration 2. The equation and prose identify the treated cohort's own unperturbed weekly hazard among survivors as the reference. The series 3 targeted reviews confirmed this specific correction. | SI §§S3.1 and S4.3 · [positive-control generator](../../test/positive_control/code/generate_positive_control.py) |
| [2.7 · N7. Contamination threshold](../objections/2.7/README.md) | Addressed in iteration 3. The revised paper compares the 200-run means with zero added hazard and gives pointwise intervals for the differences. The original figure and table identify their 20 replicates per cell. Low-variance recovery remains deferred. | §3.3.2; [PR #5 target audit](<../objections/1.5/INITIAL_TARGET_AUDIT.md>) · [checked counts](evaluations/2/repeat-2/B-follow.md) |
| [2.8 · N8. Anchor definition](../objections/2.8/README.md) | Series 3 confirmed the single-week formula corrected in iteration 2. All three primary contamination sweeps use the fifth-week anchor specified in their plan. Section 3.3.2 also reports the same-curve fourth-week workbook sensitivity. | [Paired comparison and revised experiment](../objections/2.8/README.md) |
| [2.9 · N9. Unit of fitting and pooling](../objections/2.9/README.md) | Corrected in iteration 2. Section 2.9.1 distinguishes the primary fitted comparison groups from strata retained for resampling or standardization, and describes the subgroup-by-pseudo-dose fits and pooling used for Table 8. The series 3 targeted reviews confirmed this specific correction. | §2.9.1, Table 8 and SI §S6; [follow-up](evaluations/2/repeat-1/C-follow.md) |
| [2.10 · N10. Selection of empirical examples](../objections/2.10/README.md) | Figure 1 discloses both outcome-based selection criteria and no longer calls the example representative. The worked example in §2.12 describes the fit windows as configured. The general method calls for prespecification. Series 3 reviewers confirmed the selection disclosure. | [Illustration selector](../../code/generate_kcor_empirical_intuition_figure.py) · [Czech configuration](../../data/Czech/Czech.yaml) |

### Supporting tests, figures and study provenance

- **Post-wave regression test.** Resolved for the revised edition. The corrected curve stays at its wave-exit level for a week with zero deaths and recovers a known baseline increase the following week. The original edition passes its own continuity test. [Verification](../objections/1.2/verification.md#edition-specific-regression-tests)
- **Cox figure.** The revised plot shows pointwise person-bootstrap intervals for all seven endpoints. [Figure and interval calculation](../objections/2.1/README.md#kcor-error-bars-in-the-cox-experiment)
- **Multiwave parameter recovery.** Sections 3.1.1 and S4.2.1 now specify that weekly deaths equal their expectations and the baseline rate is fixed at the exact first hazard. [Recovery evaluation](evaluations/2/repeat-3/B-follow.md)
- **Procedure used for historical results.** Figure 1 identifies the monthly quiet-period slope adjustment. The simulation comparison table identifies the single-window cumulative-hazard fitter. [Figure review](evaluations/2/repeat-3/A.md) and [grid review](evaluations/2/repeat-3/B-follow.md)
- **Older coverage pilot.** Section 2.9.1 explains the weekly Poisson resampling, fitted reference value and four deterministic scenarios. The empty table and the interpretation based on it have been removed. The separate person-bootstrap studies retain their reported results. [Pilot review](evaluations/2/repeat-3/C-follow.md)

## Previously deferred decisions

- **Weekly in-wave exposure recovery (PR #5 item 2).** The isolated survival-equation trial recovered known-input curves and produced very similar Table 8 endpoints at the configured exponent of 1.18, despite a detectable change in the frailty propagation. The author deferred adoption to a lead-author decision. [Trial and all interval comparisons](../objections/1.2/weekly-survival/README.md) · [Decision evidence](decisions.md#evidence-for-the-lead-author-decision-on-12)

- **Initial fitter and interval coverage (PR #5 item 3).** The joint first-window fit is the intended default, pending lead approval. Ordinary production still fixes the early baseline rate, and the final fit can include anchor weeks outside the paper's quiet-only mask. The [pooled multiwindow coverage follow-up](../objections/1.3/multiwindow-coverage/README.md) found coverage of 94% and 92% with the joint fitter in two well-informed scenarios and unusably wide joint-fit intervals in a separate short-window stress. Wider validation and the low-information fitting problem remain open. The person resampling unit is settled for Table 8. [Decision log](decisions.md) · [Paired fitter comparison](../objections/1.3/paired-seed-fit/assessment.md)
- **Table 8 reporting (item 4).** All six revised points reproduced. The open question is how to report contrasts and intervals that include an insufficient-signal fallback under the paper's stated reporting rule. The age-shift exercise has no guaranteed exact null; intervals containing one do not establish equivalence. Interpretation of these results and the explicitly forced Czech exponent belongs with the project lead. [Checks](evaluations/2/checks.md)

## Corrections retained in the next edition

Retain the proportional-wave identity at α = 1, the conditional post-wave continuation, the exact weekly reconstruction in 6.2, the person bootstrap, the Table 8 recalculation, the corrected frailty-attribution sentence, and the conditional treatment-effect null in the Cox comparison. The approved scaling sentence also stands. The [decision log](decisions.md) records the individual changes and verification.

## Status after the low-frailty investigations (2026-11-24)

The numbered straightforward objections have been addressed individually. This does not constitute a new independent review of the current edition.

- 1.2: exact weekly nonlinear-wave survival inversion tested; adoption and the deficit attribution convention await lead-author decision.
- 1.3 / 2.7: the all-window joint fitting trial substantially improved the stress and Table 8 results, with the core model retained. The theoretical information analysis separates irreducible low-frailty uncertainty from fitter instability. The complete coverage validation and the zero-added-comorbidity null bias remain open. Both trials are separate from the primary edition. See [all-window fitter trial](../objections/2.7/low-frailty/all-window-joint/README.md) and [theoretical limits](../objections/2.7/low-frailty/theoretical-limits/README.md).
- 3.2: the preceding-shock/mixed-week experiment remains deferred. The all-window fit does not change its first-window entry convention; therefore this objection is not resolved by that trial.
- 1.4: Table 8 reporting rules and forced-alpha sensitivity interpretation require a lead-author decision; any final change in fitter must also be reflected in the full validation pipeline and intervals.
- 2.5: quiet-window variance scan is a deferred discussion of findings, not an outstanding correction to the denominator or fitter of the existing scan.
- Author proposals: the analytical Cox/gamma derivation and the theoretical frailty/KCOR limits are separate potential contributions, not requirements imposed by the critiques.

After decisions on the experimental methods, freeze a new checkpoint, rebuild both manuscript PDFs and their evidence links as appropriate, and rerun the multi-edition independent review procedure before declaring convergence.

### Follow-up on 2.7

The [all-window contamination replay](../objections/2.7/all-window-contamination/README.md) is complete. Low frailty estimation and sampled anchor explain the persistent 50k finite-sample bias; the 5M null is 1.0013 [0.9993,1.0033] and all contamination levels are reported. The original low-information experiment remains informative as a stress case. Decision on adding the high-information experiment and adopting the fitter remains pending; broader coverage validation is next.

### Follow-up on 1.3

The [matched all-window candidate coverage validation](../objections/1.3/all-window-coverage/README.md) is complete for both archived main scenarios, previous stress scenario and two single-window null populations. Candidate coverage is 97/100, 90/100, 48/50, 187/200 and 192/200, respectively; exact coverage intervals include 95% in all five settings, marginally so for the nonlinear case. Full production workbook equivalence was checked in three cases. This supports the candidate conditional on model assumptions, but does not establish universal coverage or resolve the approximate nonlinear-wave calculation (1.2). Lead-author decision and final pipeline revalidation remain.

### Follow-up on 3.2

[Entry-aware all-window study](../objections/3.2/all-window-entry/README.md) complete. Proportional shocks before the first quiet window: noiseless 1.20743 to 1.00000; sampled 1.20386 [1.19425,1.21347] to 0.99774 [0.99321,1.00228]. Table 8 changes negligibly. Nonlinear short-window error is linked to nonquiet early fitting anchors; quiet-only diagnostic strongly improves contaminated-anchor simulations but loses information when anchors are clean. The first-entry correction is a candidate for approval; no general mask change adopted.

### Follow-up on 1.2

[Combined exact-wave and all-window validation](../objections/1.2/all-window-weekly-survival/README.md) complete. Nonlinear coverage unchanged (90/100 main, 48/50 stress); main mean 0.98982 [0.98493,0.99471], versus approximate 0.99565 [0.99072,1.00057]. True-frailty substitutions recover exact noiseless null, showing remaining fit-model mismatch after nonlinear selection. Table 8 largely unchanged. Candidate and deficit attribution approval remain pending; mixed-period/consistent survivor fitting discussion linked to 3.2.

### Nonlinear survivor-model follow-up to 3.2 / 1.2

[Exact survivor-law fitter](../objections/3.2/nonlinear-survivor-fit/README.md) trial complete. Fully labeled known-truth nonlinear waves recover k, theta, and KCOR; main sampled bias corrected (mean 1.00093 [0.99596,1.00591]); coverage unchanged 90/100 main and 48/50 stress. Mixed-window controls substantially improve when early rows contain a pulse; clean-anchor precision can worsen; partial unlabeled nonlinear wave is still approximate. All six Table 8 intervals contain one; 2022 widths increase 29�35%. Candidate approval, final diagnostics and full pipeline integration remain pending. Next substantive reporting discussion is 1.4.

### Review schedule

After the remaining 1.4 reporting discussion, stop the iterative LLM evaluation process for lead-author review. No new review iteration or PR is authorized at this stage.


**Final 1.4 alpha replay:** [historical pooled Czech objective](../objections/1.4/alpha-pooled-replay/README.md) reproduced exactly; new all-window and exact-model optima reach the lower boundary and fail identification. Forced-alpha and final Table 8 reporting remain lead-author decisions. No further LLM review iteration or PR.


## Series 4 hypothetical post-approval evaluation

See [convergence and new objection dispositions](evaluations/4/convergence.md). The actual PR remains a draft. The new high-priority issues concern false success after nonlinear row exclusion, the fitted discontinuity at alpha=1, and the bias of one-sided wave attribution. Raw gap clamping, short seed eligibility, ordinary interval reporting, configuration and minor manuscript alignment are separately recorded.
