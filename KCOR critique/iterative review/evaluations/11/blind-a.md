# Independent scientific and statistical review of KCOR

I reviewed candidate `m156` against the question “Please examine KCOR closely. What is wrong, if anything, and how would you fix it?” I examined the paper, supplement, fitting and reporting code, and selected numerical archives, and ran small deterministic examples and an ordinary workbook export. I found no algebraic error in the core gamma inversion or the fixed-slope instantaneous-hazard identification proposition. The empirical contrasts remain conditional on the working model. Section 3.1.2 already discloses the residual flags across all 999 primary Table 8 draws, the absence of an exact empirical null, and the uncertain chronology of quiet-window selection.

## 1. Enabling a wave with exponent one can change the estimate

**Moderate, method description.** Main §2.7.1 and SI §S2.1 say the module is effectively inactive when alpha equals one. Only the inversion is then an identity relative to proportional gamma inversion at the *same fitted variance*. An enabled module still excludes marked weeks from the fit and reconstructs their exposures. The module-off fit can include those weeks as quiet weeks or early anchors. Section 2.5 correctly specifies this distinction.

I tested the production fitter on 60 exact weekly hazards with starting rate 0.002, annual slope 0.085, variance 2, and a proportional excess increment of 0.005 in weeks 10–19. All weeks were designated quiet, so the marked wave overlapped fitting weeks. With the module off, 60 fitting weeks gave fitted variance 5.61408 and normalized endpoint 0.237333. With the wave enabled at alpha one, 50 fitting weeks gave variance 2.00000 and endpoint 0.176082. Both fits returned numerical status OK. This isolates the effect of the marked rows without sampling noise.

**Fix.** Revise the two “effectively inactive” statements to specify that the nonlinear inversion reduces to proportional inversion, while wave marking can change the fitted parameters and KCOR. Preserve the same marked-week exclusions across supplied exponents when comparing them. See `code/KCOR.py:10028` and `code/kcor_survivor.py:301`.

## 2. The ordinary workbook misstates the analysis and its configuration

**Moderate, reproducibility.** I wrote and reread an ordinary v7.6 workbook for enrollment 2021-24. Its About sheet describes a v4.1 slope adjustment based on centered-smoothed mortality and a binomial variance approximation. The ordinary analysis uses the joint Gompertz/gamma fit and, when requested, person-bootstrap percentile intervals. The sheet also pairs the wrong values with configuration names. My workbook reported `YEAR_RANGE = 4`, `ENROLLMENT_DATES = 0`, `OVERRIDE_DOSE_PAIRS = (1920, 2009)`, and put the actual pair `[(1, 0)]` beside “Methodology Information.” Both the reference-length and slope-anchor configuration rows were blank.

**Fix.** Build the About sheet from explicit field/value pairs and the active estimator configuration. Include the actual reference week, fitting windows, NPH mode, risk-history policy and bootstrap settings. Verify the values in a written workbook. See `code/KCOR.py:8804`, `:8827`, and `:8977`. The error affects interpretation and reproduction from the workbook, not the already computed ratio.

## 3. The workbook reports weekly exposure in its cumulative person-time column

**Minor, incorrect numerical output.** In the exported `by_dose` sheet, for one cohort the first three opening risk sets were 20,000, 19,980 and 19,959. The column `Cumu_Person_Time` contained those same three values. Under the workbook's opening-risk person-week convention, cumulative values are 20,000, 39,980 and 59,939. The implementation computes `cumPT` but writes `row["PT"]` instead. This affects the exported audit column independently of KCOR.

**Fix.** Export the already computed per-cohort cumulative exposure under the current header. Verify that differencing its successive values reproduces the weekly opening counts. See `code/KCOR.py:7949`, `:7970` and `:8004`.

## 4. The specified wave-deficit frequency is discarded in ordinary reporting

**Minor, missing diagnostic.** Main §2.7.3 requires reporting the frequency of weeks with observed mortality below the fitted conditional baseline. The inversion explicitly counts these as `deficit_weeks`, but ordinary workbook processing logs only the count with *zero added wave exposure* and discards the deficit count. Exact agreement with baseline also yields zero added wave exposure. In a three-week deterministic test with one week at its planned baseline, one at half its planned baseline and one with positive wave exposure, the routine returned `wave_points=3`, `skipped_nonpositive_excess=2` and `deficit_weeks=1`. The log reports two skipped weeks, although the specified deficit frequency is one third.

**Fix.** Retain and export the actual deficit numerator and wave-week denominator for each cohort using the weekly correction. Distinguish identity or unavailable cases explicitly. See `code/kcor_survivor.py:195` and `code/KCOR.py:7833`.

## 5. The simulation figures label a finite endpoint as an asymptote and a single realization as bias

**Minor, scientific reporting.** Section 2.11.1, the Cox comparison table, and its figures use “asymptote” and `KCOR∞`. The generator assigns the last simulated value directly, `kcor_asymptote = float(kcor_t[-1])` (`test/sim_grid/code/generate_cox_bias_sim.py:241`), and the interval-generating script evaluates week 52. SI §S4.2.1 labels the endpoint minus one “bias” while explicitly stating that each point is one realization. A single estimate's deviation from its target contains sampling error. The simulation does not estimate a long-time limit or an expectation across independent datasets at each variance.

**Fix.** Label the result “week-52 KCOR” and its subtraction from one “observed deviation from the null.” Use those definitions in the captions, axes and column names. The existing endpoints and pointwise intervals suffice for this demonstration.

## 6. Table 8 cites the older primary archive and describes the sensitivity as current

**Minor, numerical provenance.** The Table 8 caption cites `analysis/ceiling500/table8/identified_only-table8.json` for the current primary column and calls the forced-alpha column current. Section 3.1.2 and SI S10 explain that the primary analysis was refreshed and the forced-alpha intervals were archived from the preceding seed implementation. The current primary archive is `analysis/seed_ceiling/table8-primary/identified_only-table8.json`. For example, its first 2021-24 point is 1.0161931143408671 versus 1.0161931145964265 in the cited older archive. Both round to the published number. The matched draws also changed in some cases, as recorded in the new archive's `summary.json`.

**Fix.** Cite the current primary archive in the caption and label the forced-alpha intervals as archived under the preceding implementation, consistently with the results text. This correction needs no large rerun.

## 7. Cox intervals use model-based standard errors under a deliberately nonproportional hazard

**Moderate, comparator inference.** Section 2.11.1 correctly explains why the constant Cox coefficient can depart from the fixed-frailty null. Its table and figure nevertheless label the conventional Wald intervals as 95% CIs without specifying their inferential status. The code calls `PHReg.fit(disp=False)`, uses `res.bse`, and takes its p-value directly (`test/sim_grid/code/generate_cox_bias_sim.py:190–201`). In the generating model with variance 20, the marginal hazard ratio varies from 1 at enrollment to about 0.325 at week 52. For inference on the limiting constant-coefficient *summary* under misspecification, equality of the model information and the variance of the estimating score is not guaranteed. In one small repeat of the supplied design (5,000 people per group, variance 20), the same coefficient had ordinary SE 0.074393 and subject-level sandwich SE 0.074551. This check establishes the difference in calculation, not the magnitude of any coverage error in the published grid.

**Fix.** Either label the original intervals and p-values as the nominal output of the deliberately misspecified proportional model, or recompute sampling intervals for the specified follow-up-dependent Cox summary using within-cohort person resampling or suitable sandwich inference. Keep the fixed-frailty null and the marginal Cox summary distinct in describing which parameter a test addresses.
