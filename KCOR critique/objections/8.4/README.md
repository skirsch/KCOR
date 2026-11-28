# 8.4 - Upper-bound frailty fallback can erase an accurate fit

**Severity:** High for affected fitted contrasts

**Origin:** New exact counterexample to existing fallback policy

**Related issues / root causes:** 3.1, 5.11, 7.3: numerical limits versus statistical information

**Finding:** In a noiseless exact-model case, theta 99.95 is recovered to floating-point precision but reset to zero by the 99.9%-of-cap rule. Anchored KCOR becomes 0.3141 rather than 1. The fallback is flagged; empirical uncertainty can also be affected (287/999 youngest dose-2 S6 draws).

**Proposed scope (for discussion):** Separate genuine lack of identification from proximity to numerical bounds. Investigate the original safeguard and profiles, retaining unsupported fits as such. Any substantive fallback-policy change needs lead-author consideration; no automatic adoption of an expanded bound.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** approved candidate implementation (cap 500 and retain/flag near-ceiling fits); all empirical replays integrated in m142. Original m138 preserved. Independent review and lead-author approval pending.


## Original intent

Commit 4233f1ce (March 25, 2026), documentation/specs/KCORv7/insufficient_signal.md, explicitly introduced the 0.999*theta_max rule to catch optimizer-bound solutions in young/low-mortality cohorts. It treated them as ambiguous frailty signal and intentionally substituted zero rather than adopting an unreliable correction. The contemporary spec also conflated negative delta with insufficient signal; that earlier reset was corrected separately (3.1). The ceiling was not a biological upper bound. Current proportional and nonlinear fitting paths both retain the ceiling-triggered zero fallback. This is an intentional policy that has a demonstrated failure mode, not an accidental typo in the mathematics.

## Exact-model policy trial

80 noiseless Gompertz-gamma weeks, weekly baseline k=.001, gamma=.085/52, theta as shown. Comparator has identical baseline and zero variance; true fourth-week-anchored KCOR=1.

| True theta | Raw estimate, cap 100 | Current applied theta | Current KCOR | Retain constrained estimate | Diagnostic refit cap 200 |
|---:|---:|---:|---:|---:|---:|
| 99.80 | 99.80 | 99.80 | 1.000 | 1.000 | 1.000 |
| 99.95 | 99.95 | 0 | 0.314 | 1.000 | 1.000 |
| 100 | 100 | 0 | 0.314 | 1.000 | 1.000 |
| 120 | 100 | 0 | 0.289 | 0.790 | 1.000 |

The raw interior estimate 99.95 has RMSE 2.22e-18, so proximity to the ceiling alone is not evidence of lack of information. With a true value beyond the ceiling, merely retaining the constrained estimate does not fully correct the error. The expanded ceiling refit is a diagnostic using known truth, not a proposed automatic production bound. Profile results are deterministic scaled MSE with k profiled at each fixed theta; they are not uncertainty intervals.

## Matched Czech S6 replay

The existing 999 person-bootstrap risk/death draws were reconstructed with the archived generation order and seed, and archived raw theta used only where the bound guard had substituted zero, with all other fits unchanged. Current ratios reproduced before substitution; no model refits required. All cohort point estimates unchanged. In ages 40-49, 17/999 dose-0 and 287/999 dose-2 draws carried bound flags (296 distinct ratio draws changed), and anchored KCOR changed only in its bootstrap distribution: current 1.507 [0.824,2.712], retain constrained 1.507 [0.946,2.790]. Both intervals contain one; neither establishes a well-resolved true variance. Raw dose-2 frailty interval remains [0,100]. All six other age/aggregate intervals identical. Table 8 has zero near-bound fitted draws in either mode (maximum raw theta 18.868 and 14.813); therefore all Table 8 values and intervals unaffected by this policy-only trial.

## Recommendation for decision

Preserve the existing statistical purpose of the safeguard while distinguishing three facts: proximity to a numerical ceiling, an actual active bound, and inadequate statistical information. A well-fitting interior estimate should not be replaced by zero merely for reaching 99.9% of the cap. For active bounds, retain a clearly flagged constrained estimate for diagnostic use and do not characterize it as resolved; profile or targeted wider-bound sensitivity can help distinguish an imposed ceiling from a flat objective. Do not automatically increase production cap or claim the alternative is a calibrated interval. Any change to the treatment of genuinely unresolved cases requires lead-author approval. No candidate or manuscript changes made in this trial.

[Script](compare_bound_policy.py) | [Exact cases](exact-tests.csv) | [Profiles](exact-profiles.csv) | [S6 matched comparison](s6-policy-comparison.json) | [Matched draws](matched-draws.csv).


## Expanded-ceiling trial: 100, 200, 500

All 999 original person-bootstrap draws for each of the seven Czech S6 age/aggregate comparisons were matched. Caps 200 and 500 were freshly refitted with the same optimizer and original zero-at-bound policy; additionally recomputed KCOR retaining the constrained values. The 100-cap analysis is the verified original archive. 13,986 matched ratio rows (two widened limits x 7 groups x 999 draws) are finite, with no numerical failures. All point KCOR estimates agree within 1.94e-8.

Key group (ages 40-49, dose 2 versus dose 0):

| Ceiling | Dose-2 raw theta point [95% bootstrap percentile range] | Dose-2 near-ceiling draws | Current-policy KCOR [95% bootstrap CI] | Retain flagged constrained fit [95% bootstrap CI] |
|---:|---:|---:|---:|---:|
| 100 | 71.83 [0,100] | 287/999 | 1.507 [0.824,2.712] | 1.507 [0.946,2.790] |
| 200 | 71.83 [0,167.90] | 7/999 | 1.507 [0.946,2.822] | 1.507 [0.960,2.822] |
| 500 | 71.83 [0,167.90] | 0/999 | 1.507 [0.960,2.822] | 1.507 [0.960,2.822] |

Dose-0 bound hits decrease from 17/999 to zero with either widened cap. The seven Youngest dose-2 fits capped at 200 recover 200.865-238.764 with cap 500. The maximum raw estimate at 500 across all seven groups/doses/draws is 238.764. The central distributions at 200 and 500 agree closely; small paired optimizer differences also exist away from active bounds (maximum theta difference 2.763 in youngest dose 2), so not identical draw-by-draw. The other six point/CI summaries are unchanged to three decimals. The raw variance interval is wider than with cap 100, which was truncating the upper tail. A wide raw percentile range does not establish calibrated inference near the nonnegative boundary.

All 600 original low-variance synthetic datasets (200 each at 50k, 500k and 5M per cohort) were also freshly refitted at 200 and 500. No bound hits; max absolute KCOR difference vs archived 100-cap result is 1.83e-7 and 2.10e-7 respectively. For the 50k case, mean KCOR remains 1.0495 [1.0306,1.0684] and mean fitted low variance 1.548 [1.253,1.843] versus truth 0.5. Thus changing the ceiling is not a solution to the low-variance information problem.

Noiseless theta 120 is recovered at cap 200 or 500; theta 250 and 450 at cap 500. True theta 550 still reaches any tested ceiling. This confirms a wider ceiling resolves artificial truncation when the data identify the high variance; it does not guarantee resolution in general. Table 8 and the optional nonlinear module have not been fully refitted under the widened caps for this experimental comparison; original Table 8 raw values are below 19. Retaining an active bound flag remains important. Recommend cap 500 as a tested numerical candidate for this data set, not a universal scientific limit, and distinguish active constraint/weak information from a zero-variance finding. Production change remains pending.

[Expanded-ceiling script](compare_ceilings.py) | [Exact results](ceiling-exact.csv) | [S6 summary](ceiling-s6-summary.json) | [S6 draws](ceiling-s6-draws.csv) | [Bound transitions](ceiling-transitions.json) | [Synthetic summary](ceiling-synthetic-summary.json) | [Distribution plot](ceiling-distribution.png).


## Approved implementation (m139-review)

Following approval, default theta ceiling 500 in the proportional and nonlinear fitters. Preserve constrained estimates near the ceiling and flag THETA_UPPER_BOUND rather than treating proximity as insufficient signal and replacing theta with zero. Existing genuine insufficient-death/failed-fit handling preserved. This addresses the same exact counterexample at the new ceiling rather than simply shifting the discontinuity. Regression covers true theta 99.95, 120, 450, 499.95 and 550; the 499.95 fit is retained and flagged both in ordinary fit and marked alpha=1 wave, and a production workbook exact case reports KCOR 1 and THETA_UPPER_BOUND. The existing 24 noiseless solver and three nonlinear wave recovery tests pass.

All 6,993 S6 ratio draws freshly recomputed using the approved code, identical to the cap-500 trial; the displayed variance/KCOR intervals updated. Module-off Table 8 was fully recomputed (2,000 jobs), all six point/interval entries unchanged at three decimals. Largest paired draw change 0.002012, consistent with changed numerical optimization path away from the limit, not a change in the mathematical objective. Forced-alpha six point refits agree within 2.43e-7 with archived points; full 999-draw replay still running separately. The review snapshot explicitly labels the earlier forced-alpha interval archive and prior empirical figures. No claim that the forced-alpha intervals have finished refitting. The working m139 directory contains background replays; use m139-review for the frozen discussion checkpoint until they are integrated.

[Code regression](test_theta_ceiling.py) | [S6 final replay](s6-all-window.json) | [Current primary Table 8](identified_only-table8.json) | [Forced-alpha point check](forced-alpha-point-check.json).

**Background refresh integrated in m140:** age-shift, four S6 trajectories, quiet-window scans and all 1,323 sensitivity cells have completed at ceiling 500. Age-shift printed curve values and intervals unchanged; S6 all eight final levels/intervals unchanged (one interior 1940s lower band differed by up to 0.002806); sensitivity grid values differ at most 2.02e-8, unchanged at three decimals; both scans preserve 78 accepted and 63 residual flags and displayed fitted values. Full data archived in refresh/. Only full forced-alpha Table 8 remains running.

**Final forced-alpha replay:** all 2,000 original/paired draws completed under ceiling500, no flags/failures; all six printed values and intervals unchanged, max paired difference 0.00025491862. Fresh artifacts archived, to be included in next candidate.

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).
