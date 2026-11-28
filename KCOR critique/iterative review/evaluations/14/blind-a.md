# Independent review of KCOR, candidate m169

The core proportional identity checks out. For a mean-one gamma frailty, survival at cumulative reference exposure B is (1 + θB) raised to −1/θ. Its inverse is the expression in §2.4.2. The fixed-Gompertz differential identity in §2.4.4 also follows by differentiation. Both statements require the stated model. The tests below uncovered errors in distributed outputs that implement or present the method.

## 1. The ordinary workbook labels a pre-enrollment, unadjusted manufacturer comparison as KCOR

**High severity for the three manufacturer sheets and their log summaries.** The manufacturer path in `code/KCOR.py` lines 9089–9182 assigns time zero to the first source row, includes pre-enrollment observations, sums observed hazards without estimating either cohort's frailty variance, and substitutes a reference ratio of one when the selected baseline is undefined. The log independently repeats the calculation at lines 8694–8724. The shipped Czech input contains all three auxiliary source sheets. These outputs are separate from the manuscript's Table 8 and S6.

I reproduced the calculation directly from `data/Czech/KCOR_CMR.xlsx`, sheet `2021_24_MFG_D2`, aggregating the 1940–1949 birth years by manufacturer and week exactly as the workbook writer does. There are 67 pre-enrollment weeks. For the June 14, 2021 enrollment, the selected reference date is April 6, 2020, when the ratio is undefined. The code consequently exports the unanchored raw value **1.222234** for December 26, 2022, and produces finite ratios on 20 dates before enrollment. Restricting to enrollment onward and using the actual July 19, 2021 reference gives **0.920011** for that same endpoint using just the observed hazards. This latter number isolates the time-origin and anchor errors. It is not a corrected KCOR or an intervention-effect estimate.

**Fix.** Compute manufacturer contrasts from the validated frozen enrollment groups through the same fit, normalization, reference selection and reporting path used for other two-cohort comparisons. Use that output for both the workbook and its log. An observed-hazard comparator can be retained with an explicit raw label and the correct dates. A missing reference makes its anchored statistic unavailable.

## 2. The manufacturer converter counts deaths outside the specified dose group

**Moderate severity, upstream of finding 1.** In `code/KCOR_CMR.py` lines 1711–1732, a person contributes a death to dose d if the date of dose d precedes the death week. Someone already on dose d+1 at enrollment also meets that test. The denominator calculation subtracts transitions to d+1 (lines 1743–1776). For example, someone already on four doses can therefore contribute a death to the dose-three group despite being excluded from its risk set. The ordinary CMR converter instead assigns post-enrollment deaths using the fixed enrollment dose at line 867.

There is a directly observable reconciliation failure in the released workbook. For birth years 1940–1949 at enrollment 2022-06, summing **all** available manufacturer labels on `2022_06_MFG_D3` gives an opening count of **527,477** and **44,030** deaths from enrollment through the last week. The corresponding ordinary `2022_06` dose-three cohort gives **527,414** and **44,026**. These discrepancies establish that the supplied populations differ. The code establishes the mismatched death rule separately. I did not attribute the entire observed count discrepancy to that one rule.

**Fix.** Split the already established fixed exact-dose enrollment risk sets and their deaths by manufacturer, counting every eligible person once in the same product category for the entire post-enrollment trajectory. Reconcile the summed category histories to the ordinary source by enrollment week, age, dose and date. A two-person case with one subject already on the next dose catches the numerator error.

## 3. Two ancillary outputs still disagree with the ordinary estimator on reference or endpoint support

**Moderate for a missing negative-control endpoint, low for the numerical edge case.** The optional negative-control report in `code/KCOR.py` lines 8199–8226 drops unsupported dates, selects the closest remaining date and exports its value without a date column. If an age group exhausts its risk set before the reporting date, this presents an earlier result as the endpoint summary. A four-week array with the numerator's cumulative hazard unavailable in the final two weeks made this branch choose January 10 for a January 24 endpoint. The ordinary workbook explicitly treats those later weeks as unavailable. This issue concerns the optional report writer, distinct from the Table 8 replay.

The sensitivity sweep at lines 6219–6234 and the same negative-control writer at lines 8208–8211 also retain the threshold `reference > 1e-12`. Section 2.8 and the ordinary estimator accept any finite positive reference when its quotient is representable. A reference of 5 × 10⁻¹³ and endpoint of 10⁻¹² gives anchored KCOR 2 and is rejected by the sweep. Direct division on that path can conversely label an overflowing quotient as OK. I found no evidence that this changes the published 1,323 Czech sensitivity cells.

**Fix.** Look up the requested endpoint exactly and export the requested date and support status. Use the existing finite-ratio helper for raw and anchored ratios in both consumers and mark a value OK only if the final quotient is finite. Test one exhausted endpoint and one small positive reference.

## Scope

This review used the candidate manuscript, supplement, production code, and released Czech workbook. The checks were local formula replays and targeted group-count reconciliations. I did not rerun the full person-bootstrap studies. No change to a manuscript efficacy result follows from the manufacturer findings. The Czech comparisons remain descriptive under their stated assumptions.
