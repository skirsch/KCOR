# Round 8 convergence - m135

Two fresh independent GPT-6.2 Astra High blind reviews; one independent targeted Astra High resolution review of m128 against m135. Original question verbatim: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" The proportional gamma inversion was sound in all three reviews. No full simulations/bootstrap reruns, method changes or PR update.

## Newly specific findings

- **[8.2](../../../objections/8.2/README.md)** - Weighted pooled workbook series exports only its final date. High for ordinary output. New specific defect. Blind A and the resolution reviewer independently reproduced 174 dates per stratum but only one weighted pooled date in the historical Table 8 input. The pooled row is emitted outside the date loop; the archived Table 8 point was reconstructed separately rather than taken from the workbook.
- **[8.3](../../../objections/8.3/README.md)** - Sensitivity sweep double-counts overlapping populations. High for sensitivity analysis. New specific defect. Blind B verified that the sensitivity branch sums disjoint birth bands with all-ages and younger totals. In Czech 2021-26, dose 0 genuine 883/5,859,062 deaths/risk becomes 1926/15,492,550; dose 2 genuine 630/1,837,004 becomes 1290/4,302,387. The path also uses row-number time and proportional inversion despite a possible forced nonlinear fit.
- **[8.4](../../../objections/8.4/README.md)** - Upper-bound frailty fallback can erase an accurate fit. High for affected fitted contrasts. New exact counterexample to existing fallback policy. In a noiseless exact-model case, theta 99.95 is recovered to floating-point precision but reset to zero by the 99.9%-of-cap rule. Anchored KCOR becomes 0.3141 rather than 1. The fallback is flagged; empirical uncertainty can also be affected (287/999 youngest dose-2 S6 draws).
- **[8.5](../../../objections/8.5/README.md)** - Overgeneralized necessity of an external effect scale for alpha. Moderate scientific claim, optional module. New specific counterexample to older alpha assumption. Blind A constructs a homogeneous reference and a gamma cohort with known quiet parameters; two aggregate weekly survival observations under a constant unknown multiplier yield a full-rank Jacobian for multiplier and alpha (determinant -5.5082e-5). Local identifiability under that special model is not a claim of Czech identification or useful precision.
- **[8.6](../../../objections/8.6/README.md)** - Mean frailty and stabilization-skip null benchmark. Moderate to high for interpretation. New explicit counterexamples, related earlier estimand concerns. Both blind reviewers distinguish mean frailty from relative heterogeneity. With actual-enrollment variances 10 and 0, common baseline exposure 0.02 during the skip, perfect post-skip inversion gives unanchored KCOR 5/6 and anchored 1 under a fixed absolute-frailty treatment null. Shifted mean frailty also changes reference baseline without changing theta. Raw cumulative hazard itself is not invariant to depletion.
- **[8.7](../../../objections/8.7/README.md)** - Frailty-misspecification experiments overclaimed. Moderate to high for validation reporting. New specific unsupported completed-study claim. Both blind reviewers could not find the claimed five-distribution by four-metric results for gamma, lognormal, two-point, bimodal and correlated frailty. Existing studies use different estimators and targets.
- **[8.8](../../../objections/8.8/README.md)** - Sensitivity figure provenance, scales and robustness claim. Moderate to high for displayed evidence. New specific reporting defect, linked to 8.3. Both blind reviewers observed different color bars despite common scale claim. B found 12/34 archived grids straddle 1 and the displayed figure does not exactly match the archived grid. The figure shows only selected 2021-24 comparisons.
- **[8.13](../../../objections/8.13/README.md)** - Near-zero frailty confused with linear cumulative hazard. Low, wording. New specific mathematical wording mistake. SI S6.1.2 associates near-zero frailty with linear cumulative hazard. At zero variance the observed cumulative hazard is Gompertz, not generally linear; exact linearity at positive slope can occur at k theta = gamma. Main proposition is correct.

## Returning round-7 objections (five partially resolved)

- **[8.1](../../../objections/8.1/README.md)** - Returning 7.2: Incomplete source-component histories. All three reviewers reproduced missing rows hidden by pre-fit aggregation. Identical two-dose data move from 1.000 to 0.982656 when one sex/week is missing; removing trailing rows with positive survivors gives 0.511294 and the bootstrap history validator still accepts. A missing last age row moves aggregate 0.999967 to 1.009796. These are deterministic point-only fixtures, not empirical confidence intervals.
- **[8.9](../../../objections/8.9/README.md)** - Returning 7.6: Diagnostic reporting guidance and reversed SI decision-table consequence. Czech numerical/residual/omission reporting is now correct, but general sections still say contrasts are never reported on diagnostic failure. SI S2 row I1 says that violating all-required-diagnostics-pass implies KCOR interpretable; I7 also uses inconsistent condition/consequence.
- **[8.10](../../../objections/8.10/README.md)** - Returning 7.8 plus new command defect: Reproducibility scope and Table 8 cached replay. SI S7 now accurately states archive scope but main section 2.13 still promises all studies reproducible. Table 8 runner loads checkpoints with all 2,000 flags true in each mode; documented refit schedules zero fits. Current stored estimates/percentiles checked out.
- **[8.11](../../../objections/8.11/README.md)** - Returning 7.9: Early anchor assumption contradicted elsewhere. Section 2.5 now explicitly describes the early baseline assumption; section 2.4.4 still claims only designated quiet bins are fitted.
- **[8.12](../../../objections/8.12/README.md)** - Returning 7.11: Remaining bias and prerequisite wording. Abstract, Box 2 and Table 3 corrected. Section 1.5 still says depletion biases common survival estimands and presents normalization as prerequisite.

## Prior round-7 objections resolved within scope (six)

- 7.1: numerical normalization failure reporting.
- 7.3: rare-event solver stopping; 24 noiseless regressions passed, stored Table 8 intervals verified.
- 7.4: overlapping age-shift total in ordinary path (separate sensitivity branch in 8.3).
- 7.5: historic/current Czech input provenance, 4,338 archived grouped input rows reproduced.
- 7.7: four Czech trajectory plots; all saved intervals and every rendered PNG reproduced.
- 7.10: historical/current Czech alpha attribution (different structural issue at 8.5).

## Common root-cause groups

- **Population/calendar/reporting integrity:** 8.1, 8.2, 8.3; ordinary vs alternate execution paths and fixed cohort components.
- **Numerical limits versus identification:** 8.4; related to low-information frailty work but the noiseless example specifically isolates the fallback policy.
- **Model/estimand and identification claims:** 8.5, 8.6, 8.11, 8.12, 8.13.
- **Evidence, provenance and reporting:** 8.7, 8.8, 8.9, 8.10.

13 consolidated discussion items: 8 newly specific items (some extend older root causes) and 5 direct carryovers (8.10 also contains a new specific command defect). No reviewer recommendation is adopted automatically. Absence of an objection in blind review is not proof of correctness.

Frozen input verification: all 1,248 files unchanged (no missing, added, or modified non-cache files). [Manifest](inputs.json) | [Verification](input-verification.json).
