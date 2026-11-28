# Independent review of KCOR candidate m167

**Question.** Please examine KCOR closely. What is wrong, if anything, and how would you fix it?

The gamma inversion and the fixed-Gompertz-slope identification identity are mathematically consistent under the stated proportional-survivor model. The following findings concern a validation claim, an empirical diagnostic omission, and a reproducible pooled estimate. All source locations refer to the candidate. I used its manuscript, supplement, implementation, inputs and analysis scripts. I did not modify the candidate or rerun a full bootstrap study.

## 1. The published benefit control triggers the paper's own audit-only rule — major for the validation claim

Section 2.5 says to flag the nonnegative gap-exposure assumption when the upper 95% percentile endpoint for a signed initial gap change is below zero, and to retain affected estimates for audit. Table 9 injects benefit in weeks 20–79. Those weeks lie between the early fitting anchors and the quiet period beginning in week 81. Section 3.2 treats 0.834 [0.774, 0.897] as a successful detection. The replay records numerical fit status and insufficient-signal flags but omits the signed gap diagnostic (`analysis/positive_control/replay.py`, functions `estimates` and `main`).

I reconstructed the original counts and paired resampling streams with `read_data()` and the published seed 33080. The sole re-entry is rebased week 79. I ran the production fitter on the point and on the first 99 matched treated-benefit draws, then evaluated only the same quiet-window seed, signed entry-rate identity and gap contrast for all 999 draws. The first 99 results agree at the reported precision. All 999 first-pass offsets were assessable in each group.

| Group | Signed offset at point | 95% matched bootstrap interval | Negative draws |
|---|---:|---:|---:|
| Treated, benefit | −0.0227414 | [−0.0422275, −0.0068647] | 997/999 |
| Treated, harm | +0.0210405 | [−0.0008843, +0.0377747] | 29/999 |
| Shared control | −0.0005130 | [−0.0192159, +0.0149087] | 506/999 |

The fitted benefit-group raw gap was −0.0280148 and its applied excess was clamped to zero. The week-80 estimate and interval are reproducible. The substantive problem is interpreting the direction as validation of benefit detection under the method's specified gates.

**Minimal fix.** Emit the matched gap results from the existing replay and qualify the benefit row and §3.2 as audit-only. The direction remains worth reporting. If a validation claim for benefit within the assumptions is needed, use a design with an adequately informative quiet period before the intervention and freeze the fitted parameters through the benefit follow-up.

## 2. The pooled workbook drops an actual zero contrast — moderate for early follow-up

Section 2.8 defines a weighted geometric mean of available anchored stratum ratios. `_finite_hazard_ratio` explicitly returns zero for zero numerator and positive denominator. The primary pool in `code/KCOR.py:4824` skips every `kstar == 0` before assembling the weights. For positive weights the geometric mean of nonnegative ratios is zero whenever any contributing ratio is zero. Renormalizing over other ages can turn this result into a value near one.

I used `test/test_round7_workbook.py`'s 20-week, two-age fixed-cohort fixture at enrollment `2023_01`. I set the 1940 dose-1 death count in the first accumulating week to zero and recomputed its opening risk sets from the death counts. Both strata had positive, available later anchor ratios and both normalization statuses were `ok`. In the first accumulating week the 1940 anchored ratio was 0, and the 1950 ratio was 0.997037. The workbook reported pooled KCOR 0.997037, with only 1950 contributing and retained weight 0.595106. The dropped age carries about 40.5% of the intended weight. The directly aggregated ratio is a different estimand. Because cumulative hazards are nondecreasing, this particular zero-ratio defect concerns weeks before a positive reference is reached.

**Minimal fix.** Retain finite zero ratios with positive weights in the contributor set, returning the pooled limit of zero before taking logs of positive values. Apply the same rule in point estimates and draws. Test the two-age fixture and the case where all finite ratios equal zero.

## 3. The Czech S6 fit table omits the material-gap assessment — moderate reporting issue

SI S6.1.2 lists numerical fit and residual-screen status, window-omission results, and 999-draw intervals. Its archived `analysis/seed_ceiling/s6/s6-all-window-draws.csv` has no signed gap-offset or seed-status fields. Table 8's reported gap assessment is for different data and cohorts. I refitted only the 14 point histories of S6.1.2, using the stated dates, groups and source. For ages 40–49, signed offsets were negative at each of the three re-entries for both doses. At rebased weeks 48, 97 and 141, dose 0 gave −0.00001510, −0.00034471 and −0.00049617; dose 2 gave −0.00000921, −0.00000234 and −0.00017517. Both point seeds had status `ok`. Point signs alone do not establish a material deficit. The other 12 fits had positive signed offsets at every re-entry.

**Minimal fix.** Record signed gap intervals and seed completeness for the same S6 person draws, and identify comparisons, if any, meeting §2.5's audit-only rule. Until assessed, label the gap diagnostic unassessed. No change to the displayed S6 point estimates follows from this check.
