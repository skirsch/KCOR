# Review decisions

Iteration 3 begins with [3.1, the first-pass gap reset](../objections/3.1/README.md). Apply zero additional gap hazard when its estimate is nonpositive and continue the existing pooled frailty fit. Table 8 uses the same person draws for the two editions. Report empirical estimates with their intervals and report both widths when comparing uncertainty.

The [original intent audit](original-intent-audit.md) records the documented conventions behind earlier revisions. In 3.1 the tested offset was truly zero. The applied rule also covers materially negative offsets, which the original method deliberately flagged. For 2.4, the author confirmed that the entire modeled wave is eligible for correction. The revised reference preserves the original continuity goal by starting immediately before the first wave week. [Wave-entry decision](../objections/2.4/README.md)

For 3.1, flag a signed first-pass gap offset whose pointwise 95% bootstrap interval lies entirely below zero. Apply this as a [separate diagnostic on the Table 8 replay](../objections/3.1/README.md#gap-deficit-flag). KCOR uses the existing fit and calculation.

For 1.2, the author approved an experiment with [weekly survival inversion inside the wave](../objections/1.2/weekly-survival/README.md). The lead author must approve adoption of the revised attribution. The existing revised paper and method remain the primary edition.

### Evidence for the lead-author decision on 1.2

- The trial extends the post-wave survival equation already in §2.7.3. It reconciled each modeled wave boundary with the observed survival and recovered the generating cumulative hazard to numerical precision in 28 known-input cases. The starting fitted rate and frailty variance are identical in both calculations.
- In the 2021-24 unvaccinated subgroup, pseudo-dose 0, the 24th-week mean-one gamma proxy was 0.887 [0.692, 1.074], while the trial's normalized survivor variance was 1.044 [0.782, 1.323]. Cumulative wave hazard was 0.04333 [0.03731, 0.04860] versus 0.04433 [0.03897, 0.04950]. The relative changes were about +18% and +2.3%, respectively.
- In the same subgroup for 2022-06, wave week five gave 0.803 [0.364, 3.147] versus 0.819 [0.367, 3.428] for normalized frailty variance and 0.00958 [0.00000, 0.01183] versus 0.00961 [0.00000, 0.01186] for wave exposure.
- The six joint-fit Table 8 endpoints and their intervals were very similar. The largest point change was +0.00054 in the 2022 pseudo-dose 2 vs 0 comparison. The estimate moved from 1.0724 [0.3413, 1.1927] to 1.0729 [0.3413, 1.1951]. The trial also evaluates the distinct attribution rule for mortality below the fitted baseline, which preserves observed deaths and the accumulated wave exposure and allows a smaller baseline increment. This occurred in 1.3% and 30.5% of evaluated bootstrap subgroup-weeks in the two enrollments.
- The comparison uses the configured sensitivity exponent 1.18. All twelve endpoint comparisons, including both fitters, and all interval endpoints and widths are in the [trial record](../objections/1.2/weekly-survival/README.md). The [261 weekly comparisons](../objections/1.2/weekly-survival/weekly-comparison.csv) include pointwise intervals for all subgroup-weeks. Approval covers both the weekly survival equation and its trial handling of deficits.

| Objection | Decision |
|---|---|
| 1. Diagnostic and conservativeness claims | Approved. Replace the sentence in section 2.6.2, the opening sentence on original page 38, and the heading and paragraph on overlap in section 5.2. |
| 2. Proportional wave (α = 1) | Approved. Section 2.7.3 states the identity for every week and restricts division by the wave factor to α ≠ 1. Supplement S2.1 uses the same condition. The implementation already returns the ordinary inverted curve at α = 1. |
| 2. Nonproportional wave (α ≠ 1) | Applied the post-wave increment correction in section 2.7.3 and supplement S2.1. Conditional survival carries forward the existing terminal in-wave estimate. The empirical in-wave attribution has not been validated by this change. [Verification](../objections/1.2/verification.md). |
| 3. Bootstrap resampling unit | Adopt the existing pooled person bootstrap for the real-data Monte Carlo procedure. Section 2.9.1 and the supplement describe sampling eligible people with replacement, retaining death weeks and rebuilding risk sets. Historical intervals have not been recomputed. General coverage calibration and provenance of the empirical negative-control table remain open. [Paired resampling evidence](<../objections/1.3/objection3_paper_method/paired_resampling/ASSESSMENT.md>). |
| 3. Production Monte Carlo | Applied the documented sample-clock and fit-diagnostic repairs in the revised code. Corrected the zero-resample interval summary. Confirmed isolation and percentile endpoints on the Czech audit workbook. Documented the remaining 502/600 production coverage result in section 2.9.1. [Verification](../objections/1.3/verification.md). |
| 3. Interval coverage and baseline fitting | Use the §2.4 joint fit of baseline rate and frailty in the first quiet window as the intended default for subsequent analyses and drafts of revised KCOR. Proceed on the working assumption that the project lead will approve. Formal approval still needs to be requested. The ordinary code path currently uses the early-rate fit. In the paired simulation, intervals covered the true ratio in 160/200 datasets with the early-rate fit and 194/200 with the joint fit. [Paired comparison](../objections/1.3/paired-seed-fit/assessment.md). |
| 3. Pooled multiwindow coverage follow-up | The current production calculation was verified against the actual MC workbook. With one million people across two age strata and two quiet windows, fixed/joint coverage was 78% [68.6%, 85.7%] / 94% [87.4%, 97.8%] without a wave and 88% [80.0%, 93.6%] / 92% [84.8%, 96.5%] with the supplied nonlinear-wave exponent. The full interval-width and point-estimate comparisons are recorded. A separate 50,000-person short-first-window stress achieved 96% / 100% coverage, with median interval width 0.319 / 213.056 and multiple diagnostic fallbacks. High nominal coverage alone is insufficient in the low-information setting. The stress result belongs with the deferred low-frailty investigation and the lead-author fitter decision. [Study](../objections/1.3/multiwindow-coverage/README.md). |
| 4. Origin of fitted frailty | Replaced the final sentence of the frailty exogeneity paragraph with one sentence stating the evidence needed to distinguish a treatment-induced change in frailty from pre-existing selection. |
| 4. Empirical age-shift control and Table 8 | Reproduced the historical estimates and intervals. Added the six joint-fit estimates and their 999-draw person-bootstrap intervals to Table 8 and described the 2022 results in §3.1.2. The new column and its intervals already use the intended fitter. The early-rate fit is reported as a comparison. [Verification](../objections/1.4/table-8-replay.md). |
| 4. Table 8 discussion | Retain both sets of estimates and intervals in the manuscript. Defer the explanation of the insufficient-signal flags and the lower-tail example for discussion with the project lead. The evidence is preserved in the [review notes](../objections/1.4/table-8-replay.md). |
| 5. Cox comparison | Qualified the null treatment hazard ratio as conditional on fixed frailty in the simulation setup and table caption. Named the treatment-effect and cumulative-hazard benchmarks in the results. |
| 6.1. Constant hazard scaling | Replaced one sentence in section 2.4.4 to distinguish the parameter change under scaling an observed cohort hazard from scaling each individual baseline hazard. Pending the next LLM evaluation. |
| 6.2. Weekly reconstruction | Applied the exact weekly gamma inversion in both reconstructions in the revised fitter. Sections 2.4.2 and 2.5 give the corresponding paper correction. Recomputed the Table 8 joint-fit column and its intervals. [Verification](../objections/1.6/verification.md). |
| 1 and 6.1. Diagnostic and scaling wording after evaluation | Qualified remaining claims about detectable failure, treatment crossover and attenuation. Distinguished a multiplier on observed hazards from a multiplier on individual baseline hazards in the abstract, main text and assumptions table. |
| 6.3 and 6.4. Identification and shape | Replaced the curvature equivalence in §2.4.4 with the exact hazard identity at fixed Gompertz slope and its two counterexamples. Section 2.6.2 and the supplement now distinguish the expected Gompertz shape from approximate straightness in historical Czech plots. [Algebraic checks](<../objections/1.6/objection6_shape_claims/README.md>). |
| 2. In-wave interpretation after evaluation | Identified the mean-one survivor-moment proxy and the cumulative rescaling in §2.7.3 and SI §S2.1 as approximations. Estimating exposure during nonlinear waves remains an open method objection. This manuscript edit preserves the evaluated calculations. [Current disposition](open-objections.md). |

## Evaluation findings

| Objection | Decision |
|---|---|
| N1. Information given to KCOR in the Cox experiment | Revised §2.11.1 to say the two methods used the same datasets and KCOR used the generating frailty variances for inversion. The reported experiment and its numbers are unchanged. The blue N1 links in the redline identify both edits. [Evaluation finding](evaluations/2/repeat-3/A.md) · [Generator](../../test/sim_grid/code/generate_cox_bias_sim.py#L234) |
| N2. Weekly fitting objective | Iteration 2 uses the difference of the gamma-frailty cumulative hazard at consecutive Gompertz week boundaries for the first-window fit, pooled refit and residuals. The Table 8 comparison uses the same people and draw seeds as the iteration 1 checkpoint. [Checks and comparison](../objections/2.2/verification.md). |
| PR 5 item 2. Opening survivor moment | Retain the enrollment gamma shape, update its scale using baseline-only survival through the preceding week, and use the opening moment divided by the survivor mean on the inverted scale. Later wave weeks retain the existing approximation. [Formula and tests](../objections/1.2/entry-moment/README.md). |
| N2 follow-up. Weekly wave reference | Iteration 3 integrates the fitted baseline rate over each modeled wave week, starting at wave entry. Section 2.7.3 and SI §S2.1 describe this convention. [Matched Table 8 comparison](../objections/2.2/wave-reference/README.md). |
| N3. Final week of a gap | Step 3 reads both cumulative hazards at re-entry into a quiet window and applies the offset there. The iterative fit and final residual calculation use the same boundary. [Boundary tests and paired Table 8 replay](../objections/2.3/verification.md). |
| N4. First wave week | Initialize the optional correction from the preceding end-of-week total and include the first modeled week in the existing baseline recurrence. At the origin of modeled follow-up the opening total is zero. Updated the paper, supplement and Table 8. [Verification](../objections/2.4/verification.md). |
| N5. Quiet-window robustness scan | Repeated all 78 historical placements and groups with the iteration 2 joint fitter and evaluated pass rates over all attempts. Updated the main-text claim and SI §S5.4 with the new figure, pass counts and variability of fitted frailty. [Verification and results](../objections/2.5/verification.md). |
| N6. Positive-control injection | Corrected the descriptions in SI §§S3.1 and S4.3 and the equation in §S3.1 to identify the treated cohort's own unperturbed weekly hazard among survivors as the reference. [Objection 2.6](../objections/2.6/README.md). |
| N8. Anchor definition | Section 2.8 uses the cumulative KCOR value at one prespecified week. Section 3.3.2, its figures and table use the planned fifth accumulating week. The fourth-week workbook reference is a sensitivity analysis on the same fitted curves. [Objection 2.8](../objections/2.8/README.md). |
| N9. Fitting and pooling | Clarified the bootstrap refitting unit in §2.9.1 and described the Table 8 subgroup-by-pseudo-dose fits and subsequent pooling. [Objection 2.9](../objections/2.9/README.md). |
| N10. Figure 1 selection | Disclosed the final-KCOR range and preference for reduced curvature in the lead-in; removed “representative” from the caption. In §2.12 the worked example now says “configured quiet windows.” The method still calls for prespecifying them. [Objection 2.10](../objections/2.10/README.md). |
| Cox experiment, KCOR error bars | Added pointwise 95% person-bootstrap intervals at all seven frailty settings and updated the caption. [Figure comparison](../objections/2.1/README.md#kcor-error-bars-in-the-cox-experiment). |

The author approved three supporting-evidence edits in iteration 2. Sections 3.1.1 and S4.2.1 specify the conditions of the multiwave recovery example. The Figure 1 and simulation comparison captions identify the procedures used. Section 2.9.1 describes the historical coverage pilot, replacing its empty table and the interpretation in section 5. All three series 3 targeted reviewers confirmed that the three disclosure objections were addressed. [Open versus targeted results](evaluations/3/convergence.md)

## Deferred discussion after the objection review

Defer [3.2, mortality before the first quiet window](../objections/3.2/README.md). Revisit the [matched Table 8 intervals and known-variance checks](../objections/3.2/experiment.md) alongside low-frailty recovery. The entry-survival calculation remains an experiment.

Return to low frailty recovery after the remaining straightforward objections. The [2.7 simulations](../objections/2.7/README.md#why-the-zero-contamination-mean-exceeded-one) distinguish frailty estimation from random anchoring. The [Table 8 replay](../objections/1.4/table-8-replay.md#iteration-2-check-against-the-low-frailty-simulation) shows how an unstable fit affects a shared denominator. The empirical groups’ true frailties are unknown. Iteration 3 replaced the §3.3.2 threshold claim with measured differences from zero added hazard using the original fitter and 200 simulations per level.

Revisit the [quiet-window frailty figure from SI §S5.4](revised/documentation/preprint/figures/fig_si_quiet_window_theta_scan_czech_2021_24.png). The author flagged the pattern across window midpoints in the six birth-decade and dose groups for discussion after all objections have been reviewed. The [78 window-level results](../objections/2.5/windows.csv) are saved alongside the scan.

## Author proposals

- The author proposed showing analytically why ordinary Cox regression departs from 1 in the gamma-frailty experiment, alongside or in place of the simulation. Inclusion in §2.11.1 is open. [Proposed derivation](author-proposals/objection-5-cox-gamma-derivation.md).

## Forthcoming PR briefing (2026-11-25)

Record the [all-window contamination follow-up](../objections/2.7/all-window-contamination/README.md) and its finite-sample boundary-bias explanation. The author wants a top-level [PR presentation](../PR-PRESENTATION.md) of the deep findings, evidence and pending decisions. Maintain it as the review proceeds. No approval to alter the official method, paper, or open a PR is implied.

### Candidate coverage evidence

[1.3 coverage validation](../objections/1.3/all-window-coverage/README.md) reproduces both main scenarios on matched draws, the three actual MC workbook checks, and the original biased single-window nulls. Results inform the pending all-window fitter decision. No adoption or change to production/paper.

### Lead-author briefing versus internal working notes (2026-11-25)

Maintain [PR-PRESENTATION.md](../PR-PRESENTATION.md) as the scientific findings and changes document, ordered from major results to minor corrections. Keep review iterations, work to do, and deferred author discussion in [WORKING-NOTES.md](../WORKING-NOTES.md) and the existing internal archives. Do not include informal Cox proposal or seasonal-window interpretation in the lead-author briefing unless they later become substantive adopted contributions.

### 3.2 follow-up evidence

[Entry-aware all-window replay](../objections/3.2/all-window-entry/README.md) supports first-entry reconstruction under proportional shocks. The separate nonquiet fitting-anchor exclusion is diagnostic only and trades accuracy against precision. Both remain outside the approved pipeline. The [lead-author presentation](../PR-PRESENTATION.md) and [internal working notes](../WORKING-NOTES.md) record the scientific result and pending decision separately.

### Combined 1.2 trial with all-window fitter

[Combined validation](../objections/1.2/all-window-weekly-survival/README.md) complete. Exact weekly inversion is valid with known inputs; in the actual fitted pipeline coverage does not improve and main mean becomes slightly biased below one. Fitted frailty under nonlinear survivor selection is the main identified deterministic discrepancy. Table 8 changes are small; approve any adoption and deficit attribution separately. No general method or paper change.

### Exact nonlinear survivor-law fit trial

[3.2/1.2 coherent fitting and wave inversion study](../objections/3.2/nonlinear-survivor-fit/README.md) complete with full matched draws. Exact labeled-model recovery and improved main sampled mean support lead-author consideration. Coverage unchanged; clean-anchor precision and unlabeled wave limits remain. No adoption. Present scientific findings and preserve internal decisions separately.

### Lead-author review gate (2026-11-25)

The author instructs that after the remaining objection discussion, do not start another LLM review iteration. Present the findings and candidate changes to the lead author first. Any subsequent evaluation is deferred until that review and a new decision. No PR opened.


### 1.4 historical alpha replay

The [matched historical real-data analysis](../objections/1.4/alpha-pooled-replay/README.md) reproduces the published illustrative minima exactly; all-window and exact survivor fitting fail identification on the original grid, and exact-model minima track the extended lower boundary. No external scale was supplied. Keep 1.18 sensitivity-only and the new machinery experimental pending lead-author review. No new LLM evaluation, PR, or production/paper change.
