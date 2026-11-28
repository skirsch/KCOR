# Independent review of KCOR

**Question:** Please examine KCOR closely. What is wrong, if anything, and how would you fix it?

**Assessment.** The proportional gamma inversion is mathematically sound when the variance and time origin refer to the stated population. The fixed-Gompertz-slope proposition is a valid working-model result. The manuscript appropriately distinguishes the Czech descriptive comparisons from intervention effects and records the lack of empirical identification of the optional exponent. I found two consequential data-processing errors, overstated validation evidence, and an important qualification to the null benchmark. These warrant correction before relying on the framework's reported robustness.

## 1. The all-ages sensitivity sweep counts the same people two or three times

The sensitivity target in `test/sensitivity/Makefile:21–43` uses the standard Czech converter workbook. In `code/KCOR.py:6414–6426`, the sensitivity branch groups **all** input rows by dose and date. Each birth-band constituent appears once on its own, again in the supplied all-ages total (birth code −2), and, for people born in 1960 or later, a third time in the supplied −60 total. These totals are produced in `code/KCOR_CMR.py:1620–1648`. The younger and older populations thus receive different weights. Duplication also changes the counts used by the minimum-deaths rule.

I independently summed the rows in `data/Czech/KCOR_CMR.xlsx`, sheet `2021_24`, in the first accumulating week (2021-26). The disjoint constituent counts exactly match the supplied −2 counts.

| Dose | Actual deaths / risk set | Sensitivity-branch deaths / risk set | Actual weekly integrated hazard | Sensitivity-branch hazard |
|---|---:|---:|---:|---:|
| 0 | 883 / 5,859,062 | 1,926 / 15,492,550 | 0.0001507181 | 0.0001243255 |
| 2 | 630 / 1,837,004 | 1,290 / 4,302,387 | 0.0003430085 | 0.0002998785 |

This affects the dedicated sensitivity path. The ordinary fit has an explicit all-age source selector (`code/KCOR.py:4252`). The sweep also builds time with row number at line 6426 and, if forced nonlinear correction is configured, its final trajectory uses only proportional gamma inversion at lines 6557–6563 even though its fit receives the wave calendar at lines 6489–6498.

**Fix.** Construct the population, calendar and normalized trajectory through the same functions used by the ordinary estimator, varying only the selected sensitivity parameters. Verify that adding either or both redundant totals leaves a sweep cell invariant and that a single cell with matched primary settings reproduces the ordinary direct all-ages result. Regenerate the affected analyses and identify the version of every archived grid.

## 2. A missing constituent row can become an apparently complete fixed-cohort history

The ordinary pipeline buckets birth years and pools sex/comorbidity rows before checking calendar continuity (`code/KCOR.py:6729–6785`). Another component's row can consequently fill a missing calendar position. A whole subgroup of survivors can disappear and even reappear while the aggregate passes the calendar check. The bootstrap separately checks each component's start and adjacent dates (`code/bootstrap_reporting.py:71–92`) but accepts a component ending with positive survivors weeks before the declared contrast horizon.

**Reproduction.** I ran the ordinary `process_workbook(..., return_rows=True)` entry point on a small workbook with one sheet `2021_24`, one birth year (1950), doses 0 and 2, and 56 Mondays from 2021-06-14. At each dose, male and female risk sets each start at 30,000. Males have 20 deaths per week and females one, with the next opening count reduced by exactly those deaths. I used the Czech configuration and dose pair 2/0. With the identical histories the final anchored KCOR equals 1.

| Change to dose 2 only | Final anchored KCOR | Normalization status | Bootstrap history validation |
|---|---:|---|---|
| None | 1.000000 | ok | Accepted |
| Omit the male row on 2021-12-27 only | 0.982656 | ok; both numerical fit statuses OK | Rejected |
| Omit male rows from 2021-12-27 to the end | 0.511294 | ok; numerator has an insufficient-signal fit flag | Accepted |

The first omission creates a rebound in the reported risk set even though enrollment is closed. The second was deliberate removal of records for living people; a fitted-parameter flag cannot diagnose their loss from the data. Results contradict the ordinary-workbook claim concerning incomplete histories in §2.9.1 (`documentation/preprint/paper.md:568`). I tested the point path and bootstrap history constructor directly, without running a bootstrap.

**Fix.** Check coverage, duplicate keys and risk-set transitions for each original disjoint input component before pooling. Establish a common observation endpoint or explicit component censoring status. A zero-risk component can justifiably cease contributing once exhausted. For live components with missing rows, mark affected comparisons unavailable at the affected horizons and align bootstrap eligibility with the point estimator. Check for unexplained increases in aggregate risk sets as a second line of defense.

## 3. The supplement's sensitivity figure does not support the across-comparison robustness claim

SI §S4.4 (`documentation/preprint/supplement.md:215–223`) says the panels use a unified scale centered at one and there are no qualitative changes across comparisons. The supplied PNG actually displays three panels from enrollment 2021-24 with three **different** color bars. The archive `test/sensitivity/out/KCOR_SA.xlsx` contains 34 comparison sheets. The minimum and maximum across the numeric 7 × 7 grid straddle 1 in **12**. For example, 2022-26 dose 1/0 ranges from 0.907 to 1.182, dose 2/1 from 0.813 to 1.110, and 2022-47 dose 3/1 from 0.841 to 1.073. In 2022-06 dose 3/1 the range is 1.271–1.906. The archived log identifies a v6.0 run. The matched 2021-24 dose 2/0 panel also differs numerically from the archived sheet (figure range about 1.099–1.272, workbook 1.105–1.281). I could not establish that this precise figure is reproducible from the archived grid.

**Fix.** Explicitly label the three displayed comparisons, provenance and fitter. Use the exact source used to produce the figure, verify its cells, use the claimed common scale or label the actual scales, and report across-enrollment sign changes and ranges for the corrected sweep. Distinguish sensitivity of the fitted geometry from the mathematical effect of changing the anchoring denominator.

## 4. The stated misspecification evidence exceeds what is reported

Main §3.3.1 (`documentation/preprint/paper.md:770–782`) says bias, variance, coverage and non-identifiability rate **are reported for each** of gamma, lognormal, two-point, bimodal, and within-subgroup correlated frailty. I could not find the five-by-four set of results in the frozen manuscript, supplement or accompanying analyses. The main simulation generator has a lognormal non-gamma scenario (`test/sim_grid/code/generate_sim_grid.py:498`). The tail-mixture study fits a single-window cumulative-hazard model (`test/sim_grid/code/generate_tail_sampling_sim.py:160–214,434`). Its diagnostic CSV gives, for example, median KCOR 0.9501 and drift −0.0598 for the tail-sampling null, without repeated-sample coverage or non-identification rates. The bootstrap pilot's non-gamma entry is compared against a fitted single-dataset reference, a limitation the manuscript itself recognizes in §2.9.1. These are different experiments and do not validate coverage or failure detection by the revised all-window estimator for the list of distributions in §3.3.1. Trajectory stability within one realization also differs from sampling variance across independent realizations.

**Fix.** Report the actual scenario, target, fitter, replications and measured outcomes for each archived result. Remove unsupported completed-study claims, or supply the missing independently replicated studies using known population targets, current code, and explicit diagnostic rules. For correlated outcomes, specify the dependence unit and match the resampling unit to it.

## 5. After the stabilization skip, a treatment null does not generally imply unanchored KCOR of one, even under the exact model

Box 2, §§2.1.1 and 2.7 define the primary contrast at unit frailty in two **separately mean-one** cohorts at the post-skip origin. The general validation bullet at `documentation/preprint/paper.md:716` says a true null effect should give a flat KCOR at 1. Section 2.11.1 does describe a narrower, legitimately equal-baseline simulation. A skip can itself break the equal-reference-baseline condition when a null is specified at actual enrollment.

Take two mean-one cohorts at actual enrollment, with identical individual baseline and gamma variances 10 and 0. Let the common cumulative baseline through the skip be 0.02 (possible under the assumed Gompertz law). In the first cohort, the surviving frailties have mean 1/(1 + 10 × 0.02) = 5/6 relative to the original unit. In the second cohort the mean remains one. Renormalizing each survivor law to mean one at the start of accumulation therefore gives post-skip reference baseline hazards in ratio 5/6. Indeed, for subsequent common baseline increment U, gamma inversion of the post-skip observed increment gives U/(1 + 10 × 0.02) in cohort one and U in cohort two. With perfect information the primary KCOR is **0.8333 at every post-skip horizon**, and the separately defined anchored KCOR is 1. No treatment, misspecification, sampling error or numerical failure is involved. A similar distinction applies if pre-enrollment selection already creates a cohort-specific mean risk level.

**Fix.** State the additional benchmark assumption explicitly. Unanchored KCOR equals one under equality of the cohort-specific mean-one *post-skip* reference cumulative hazards. Proportional reference hazards suffice for anchored KCOR to remain one. Explain why reference frailty one need not denote the same absolute susceptibility in two cohorts. Section 2.1 (`paper.md:141`) should also qualify its statement that cumulative outcome accumulation is invariant to depletion. Raw cumulative mortality changes under differential frailty; any invariance of the corrected target comes from the correctly specified inversion.

## Smaller interpretation errors

- SI §S6.1.2 (`supplement.md:413`) associates near-zero frailty variance with effectively linear cumulative hazards. At zero variance, observed cumulative hazard follows the increasing Gompertz cumulative curve. Exact linearity at positive slope is possible when kθ = γ and the marginal hazard is constant, as the main paper correctly explains in §2.4.4. Report local precision or a profile alongside a boundary estimate when claiming it is poorly resolved.
- SI §S2 (`supplement.md:59–68`) heads the final column “Consequence if violated.” Row I1 currently says a violation of “All required diagnostics pass” implies “KCOR interpretable.” Correct the reversed consequence and align it with I7.

I examined the frozen source, selected archived workbooks and figures, and performed small deterministic and ordinary-workbook checks. I did not rerun the national study or its bootstrap. The observed input defects establish a problem with those operational paths. The size of their effect on each previously published endpoint requires a matched replay with corrected inputs.
