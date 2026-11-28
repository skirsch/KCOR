# KCOR revision — findings for the lead author

**25 November 2026 | Start here**

For a short, plain-language introduction, see the [simplified briefing](PR-BRIEFING.md).

The independent review and subsequent investigations exposed several apparently separate problems in KCOR: unstable frailty fits, very wide uncertainty intervals, a biased synthetic null, and approximation errors in wave correction. Much of the work traced these symptoms back to a smaller number of common causes. The findings below distinguish mathematical or software mistakes from inherent statistical limitations, and distinguish implemented corrections from proposed changes to the method.

The objective remains the same: compare cumulative mortality hazards after accounting for frailty selection. The investigations preserve the documented experiment intent and do not use agreement with a desired result as a selection rule.

## 1. Overview — problems, causes, solutions and evidence

| Observed problem | Root cause established by the investigations | Proposed or implemented correction | What it solved; what remains |
|---|---|---|---|
| Extreme fitted frailties and very wide KCOR intervals (1.3, 1.4, 2.7) | A weak first-window fit can be locked into the later fit; reconstructed gap offsets reinforce the erroneous seed. | Jointly fit the common baseline rate and frailty variance across all quiet windows, recompute offsets for every candidate pair, and check a second optimizer starting point. | Eliminated extreme errors in the matched stress study; 2022 Table 8 intervals became 73–90% narrower. **Candidate.** Genuine weak statistical information remains. |
| KCOR above 1 under zero added hazard (2.7, 2.8); unreliable small frailty estimates | Weak mortality curvature, the nonnegative frailty boundary and sampling of the early normalization ratio. Not the gap reset or a mere anchor-index mistake. | Derive theoretical precision limits; test population-size dependence and retain the intended contamination experiment. Correct the unsupported threshold claim. | Explained persistent null bias in the original small cohort; 5M-per-cohort null is consistent with 1. **Claim corrected; additional scientific results proposed.** Small-sample bias is not eliminated. |
| Nonlinear-wave correction can use incorrect fitted frailty and survivor moments (1.2, 3.2) | For alpha ≠ 1, selection generally changes the survivor distribution away from gamma. The fit and inversion use inconsistent survival laws. | Solve the survival equation week by week and fit quiet mortality using the same nonlinear survivor law. | Recover generating parameters and KCOR exactly in labeled noiseless cases; improve sampled mean in the main scenario. **Candidate.** Coverage and weak-information limits remain. |
| Rate and frailty distorted after a shock before the first quiet window (3.2) | Pre-window selection was not carried into the fit; early fitting anchors can include wave deaths. | Carry proportional entry exposure into the first quiet window; investigate marked-wave fitting with the consistent survivor law. | Exact recovery in the proportional-shock control; improved sampled marked-wave cases. **Candidate.** Unmarked mixed exposures are not identified. |
| Correct positive frailty reset to zero (3.1); gap and wave timing errors (2.3, 2.4) | A nonpositive gap increment was mistaken for absence of frailty; some weekly boundaries were omitted or misaligned. | Keep fitted frailty, flag material deficits; correct week integration and indexing. | Direct counterexamples corrected. **Implemented.** Flagged low-information fits may still occur. |
| Alpha estimate unstable despite a very large registry (1.4) | The historical loss uses few positive-excess observations and lacks an independent effect scale; temporal shape alone has weak information. | Replay original objectives with improved fits and exact survivor moments; calculate model-based information. | Original 1.190/1.185 minima reproduced; new optima reach search boundaries. **Identification not established.** Keep 1.18 as sensitivity-only. |
| Uncertainty and supporting evidence not consistently aligned with the actual method (1.1, 1.3–1.6, 2.1, 2.5–2.10, 3.3) | Resampling, workbook routing, reporting denominators, experiment descriptions and claims differed from the intended statistical quantities. | Person resampling and aligned pipeline checks; correct experiment provenance and wording. | Reproducible comparisons and more precise claims. **Implemented in revised edition.** Final numerical tables depend on the approved fitter. |

These causes are not interchangeable. In particular, an optimizer defect can be fixed, whereas limited information cannot be created by a better optimizer. The distinction is central to the revision.

## 2. Main finding: all-window fitting fixes the numerical instability, not the information limit

The chain observed in the difficult multi-window case is:

**Small death counts in the first quiet interval → poorly determined rate and frailty → erroneous reconstructed gap exposures → unstable final KCOR and bootstrap intervals.**

The earlier joint fitter estimated k and theta together in the first window but then held k fixed when pooling subsequent quiet windows. The proposed fitter uses one common k and theta across all quiet windows, recomputing the gap exposures within the objective and evaluating a second optimizer start. It does not introduce a separate baseline scale for each window.

| Matched 50,000-person stress case (true KCOR = 1) | Previous joint fitter | All-window candidate |
|---|---:|---:|
| Pathological dataset KCOR [95% bootstrap interval] | 812.675 [0.848, 744.878] | 1.016 [0.875, 1.131] |
| Offending subgroup frailty, true 0.5 [95% bootstrap interval] | 91.841 [0, 91.639] | 0.462 [0, 2.227] |
| Median 95% interval width | 213.056 | 0.315 |
| Coverage | 50/50 | 48/50 |

The old apparent high coverage arose with unusably wide intervals. The new candidate provides meaningful precision in this stress design. In the matched 2022 Table 8 replay, all three 95% intervals narrow substantially; for example, pair 1/0 moves from 1.030 [0.322, 1.236] to 1.012 [0.968, 1.055]. The 2021 results are broadly unchanged, with modestly wider intervals. Coverage in sampled studies is not universally improved.

**Proposed change:** adopt the all-window common-parameter fitter, subject to approval and integration. [Full stress, Table 8 and coverage comparisons](PR-SCIENTIFIC-EVIDENCE.md#1-all-window-joint-fitting-and-uncertainty-coverage).

## 3. New theoretical finding: poor frailty precision does not necessarily imply poor KCOR precision

How much frailty can be resolved depends on cohort size, mortality, the magnitude of frailty, quiet-window duration and placement, and information from survival through gaps. Small frailty contributes little curvature, making the rate and variance difficult to distinguish. In finite samples, the constraint theta ≥ 0 also makes errors asymmetric.

A Fisher-information calculation for the actual multi-window model quantifies the joint uncertainty of the fitted parameters and of KCOR, allowing for uncertain gap exposures. In the matched stress design, the theoretical local pooled KCOR interval at true ratio 1 is **[0.877, 1.140]** (width 0.263); the new fitter's observed median bootstrap width is 0.315. Even though frailty itself is poorly determined, endpoint precision approaches the model-based benchmark in this design.

The original zero-addition contamination experiment is a separate single-window example of the information limit. For 200 matched runs per level:

| People per cohort | Null mean KCOR [95% Monte Carlo interval] |
|---|---|
| 50,000 | 1.0494 [1.0305, 1.0684] |
| 500,000 | 1.0090 [1.0027, 1.0153] |
| 5,000,000 | 1.0013 [0.9993, 1.0033] |

The low-variance boundary mechanism predicts the observed frequencies of estimates near zero: 44.3%, 32.6% and 7.7% versus 43.0%, 35.0% and 7.5% observed. More information reduces both spread and bias, not merely interval width. The sampled anchor contributes separately. The high-information result does not justify excluding the original stress case or a universal contamination threshold.

**Proposed contributions:** the information-limit derivation and a larger-cohort companion experiment that preserves the original contamination design. [Precision theory](PR-SCIENTIFIC-EVIDENCE.md#2-theoretical-limits-of-frailty-and-kcor-precision) · [Complete contamination comparison](PR-SCIENTIFIC-EVIDENCE.md#3-why-the-original-contamination-experiment-had-kcor-above-one).

## 4. Nonlinear wave correction: the fitter and inversion must describe the same survivors

For alpha = 1, gamma inversion remains exact. For nonlinear wave exposure, the survivor distribution generally ceases to be gamma. The proposed consistent model uses the same survival function for fitting quiet mortality and inverting the observed wave:

$$
S(B,W)=E[\exp(-ZB-Z^\alpha W)].
$$

Exact inversion alone cannot repair an upstream frailty estimate from the wrong working model. In the noiseless main experiment the exact inversion with fitted gamma-working inputs gives KCOR 0.988967; supplying generating frailty restores 0.999943. Using the consistent survivor-law fit and exact inversion instead recovers the generating parameters and KCOR = 1 to numerical precision.

For the sampled main experiment, mean KCOR improves from 0.98982 [0.98493, 0.99471] to 1.00093 [0.99596, 1.00591]. Coverage stays 90/100 [82.4%, 95.1%]. The short-window stress still has positive bias. For Table 8 the exact survivor-model intervals all contain one; compared with the all-window gamma-working fit with exact inversion, the 2022 widths increase 29–35%. True empirical frailty is unknown.

A related but separately verified issue is mortality before the first quiet window. The proposed proportional entry calculation corrects the known-truth sampled mean from 1.20386 [1.19425, 1.21347] to 0.99774 [0.99321, 1.00228]. The exact nonlinear candidate improves marked-wave cases, but cannot uniquely identify unlabeled wave exposures. Dropping clean fitting-anchor rows can worsen precision.

**Proposed change:** consider consistent nonlinear fitting and exact weekly inversion together, alongside the proportional entry correction. Approve explicitly how marked weeks and weekly mortality deficits are treated. [Detailed wave and entry results](PR-SCIENTIFIC-EVIDENCE.md#4-nonlinear-wave-inversion-and-mortality-before-the-initial-quiet-period).

## 5. Alpha: more accurate modeling does not establish identification

The original pooled Czech analysis was reproduced exactly: pairwise optimum 1.190 and collapse 1.185. The historic selection has 10.36 million overlapping enrollment memberships from a 12.60-million-record registry. Yet the positive-excess filter retains only 62 cohort-week rows from eight cohorts. The illustrative historical objective did not supply the independent effect scale required by the formal method.

| Fit | Pairwise optimum | Collapse optimum |
|---|---:|---:|
| Historical | 1.190 | 1.185 |
| All-window joint | 1.000 (boundary) | 1.000 (boundary) |
| Exact survivor | 1.000 (boundary) | 1.000 (boundary) |

All historical-grid bootstrap stability ranges span [1.00, 1.30] (not confidence intervals; enrollments overlap). On the wider diagnostic grid the exact-model optima again follow the lower boundary, including after flagged cohorts are excluded. The separate temporal-shape information calculation also indicates weak identification.

**Conclusion:** retain the intended alpha ≥ 1 domain and treat 1.18 as a sensitivity value, not a point estimate. No adoption of a different exponent is supported here. [Historical replay and objective curves](objections/1.4/alpha-pooled-replay/README.md).

## 6. Corrections already incorporated — from algorithmic to editorial

The following corrections do not depend on adoption of the new fitter or exact nonlinear candidate.

| Scientific or technical issue | Implemented correction | Objections |
|---|---|---|
| Mathematical consistency and weekly indexing | Exact integrated weekly hazard and gamma inversion, last gap week and first wave week, conditional wave-exit continuation and opening survivor moment; preserve the alpha=1 identity. | [1.2](objections/1.2/README.md), [1.6](objections/1.6/README.md), [2.2](objections/2.2/README.md), [2.3](objections/2.3/README.md), [2.4](objections/2.4/README.md) |
| Nonpositive gap offsets | Do not erase positive frailty, retain a diagnostic flag for material deficits. No redesign of the intended pipeline. | [3.1](objections/3.1/README.md) |
| Uncertainty and workbook processing | Resample people, preserve their death times and risk sets, refit each draw; correct sample-clock and reporting behavior. Reproduce the empirical intervals. | [1.3](objections/1.3/README.md), [1.4](objections/1.4/README.md), [2.9](objections/2.9/README.md) |
| Matched frailty-law validation | Completed the missing five-structure comparison: mean anchored KCOR 0.994--1.005, coverage 92--99%, no numerical failures in the tested design. | [8.7](objections/8.7/README.md) |
| Fixed-cohort and inference safeguards | Validate source histories and calendar before aggregation; enforce death-only accounting by default, withhold unsupported estimates and mixed-composition intervals, retire legacy analytical uncertainty, and separate convergence from residual and gap-assumption diagnostics. | [9.1](objections/9.1/README.md), [9.5](objections/9.5/README.md), [9.6](objections/9.6/README.md), [9.7](objections/9.7/README.md), [9.8](objections/9.8/README.md), [9.9](objections/9.9/README.md), [9.10](objections/9.10/README.md) |
| Validation controls | Correct scan fitter and denominator (all 78 attempts counted); match positive-control description to the generator; run the fitted workbook pipeline. Preserve the intended contamination and reference timing, compare additions against the measured null. | [2.5](objections/2.5/README.md), [2.6](objections/2.6/README.md), [2.7](objections/2.7/README.md), [2.8](objections/2.8/README.md), [3.3](objections/3.3/README.md) |
| Interpretation and smaller wording changes | Refine diagnostic and bias-direction claims, explain evidence needed for treatment-induced frailty, clarify Cox benchmark and uncertainty bars, scaling and curve shape, clarify fitted grouping and disclosed empirical example selection. | [1.1](objections/1.1/README.md), [1.4](objections/1.4/README.md), [1.5](objections/1.5/README.md), [1.6](objections/1.6/README.md), [2.1](objections/2.1/README.md), [2.9](objections/2.9/README.md), [2.10](objections/2.10/README.md) |

The ancillary manufacturer workbook (not used in the paper) now uses the ordinary fixed-enrollment estimator and resampling; its source product groups must reconcile to the ordinary dose groups. [14.1 and 14.2](iterative%20review/evaluations/14/manufacturer-batch/README.md).

See [full correction table](PR-SCIENTIFIC-EVIDENCE.md#5-implemented-mathematical-software-and-validation-corrections) and [smaller manuscript changes](PR-SCIENTIFIC-EVIDENCE.md#6-manuscript-clarifications-and-smaller-rephrasings).

## 7. Scientific conclusions and approval requested

The primary recommendation is the common-parameter all-window fitter. The theoretical work establishes a way to reason about frailty and endpoint precision, rather than expecting an optimization change to solve a lack of information. The exact nonlinear and proportional entry candidates address separate, demonstrated model-consistency errors. All of these have matched numerical evidence; they also have limitations.

The lead author should decide on:
- adoption of the new all-window fit and final interval pipeline;
- inclusion of the theoretical precision limits and higher-information contamination companion;
- adoption of the nonlinear survivor-law and entry corrections, including the deficit attribution convention;
- final empirical reporting and sensitivity-only interpretation of alpha.
- treatment of fits near the frailty ceiling: a noiseless fit at variance 99.95 was accurately recovered but reset to zero, moving KCOR from 1 to 0.314. Retaining the constrained estimate corrects that case without resolving genuinely weak information. Matched Czech results reduce the youngest dose-2 bound hits from 287/999 (cap 100) to 7/999 (cap 200) and zero (cap 500); a wider bound is a candidate, not an identification guarantee. [Decision evidence](objections/8.4/README.md).

The implemented corrections and proposed candidates are kept separate. The numerical results above are properties of the documented studies and models, not evidence of treatment effects in the Czech registry.

**Read next:** [Detailed scientific comparisons and evidence](PR-SCIENTIFIC-EVIDENCE.md) · [Clean manuscript](<iterative review/KCOR-clean.pdf>) · [Cumulative linked redline](<iterative review/KCOR-redline.pdf>) · [Objection-by-objection evidence](README.md).

The two PDFs present the current implemented edition; experimental changes that await approval are described in the evidence rather than silently included in the paper.
