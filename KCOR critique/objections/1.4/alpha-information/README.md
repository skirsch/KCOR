# Can the new survivor-law fitting model identify alpha?

This is a local theoretical information diagnostic, not a new KCOR estimator, an empirical estimate of alpha, or a new LLM review iteration.

The original Methods 2.7.2 estimates alpha from the cross-cohort excess-hazard ratio, requiring a fixed external intervention-effect scale. That is distinct from the possibility exposed by the exact nonlinear survival law: selection during a wave modifies survivor frailty and therefore the shape of later quiet mortality. The latter can in principle contain information about alpha even if every cohort has a freely estimated total wave exposure. It does not automatically disentangle causal treatment effects or establish identification under unspecified baseline changes.

The diagnostic adds alpha to the existing exact proportional/nonlinear-survival Fisher calculation. Under the reference model, use all actual configured quiet windows and observed aggregate survival transitions across gaps. Each cohort retains separate unknown k, theta and nonnegative gap/wave exposures. The scalar efficient information for alpha is the squared norm of its weighted hazard derivative after projecting out all cohort-specific nuisance derivatives. For a common alpha across disjoint age/dose cells within one enrollment, sum the efficient information; do not combine different enrollment dates as independent. No external VE or common wave amplitude is imposed. At zero frailty alpha has no effect on survival; near-zero/boundary references contribute no regular alpha information. Actual cohort input is the Czech CMR workbook, not the overlapping Table 8 pseudo-cohorts.

Reference alpha=1.18 is stipulated only. For the Czech design, the primary reference k/theta are from fitting the latest experimental survivor model at this stipulated alpha; nonnegative gap exposures are reconstructed from aggregate observed survival as in the earlier theoretical information study. A second reference assumes theta=0.5, maintaining the fitted k. This is an idealized regular, local model-based precision bound; the aggregate reference differs from the weekly deficit attribution in the actual experimental estimator. It is not an observed alpha profile or empirical confidence interval. The 2022 near-nonidentification results have very small information; focus on their order of magnitude, not exact digits. Derivative steps multiplied fivefold reproduce the estimates.

| Scenario | People at modeled origin | Local alpha SE (all nuisance parameters uncertain) | Local nominal 95% half-width |
|---|---:|---:|---:|
| Synthetic main (known generating reference) | 1,000,000 | 0.425 | 0.834 |
| Synthetic short-window stress | 50,000 | 4.51 | 8.84 |
| Actual Czech 2021_24; fitted-reference theta | 3,000,381 | 1.00 | 1.97 |
| Actual Czech 2022_06; fitted-reference theta | 2,962,695 | about 220 | about 432 |

At a stipulated reference of 1.18, the difference from proportional amplification 1 is 0.18. Even the main synthetic design has an ideal local SE of 0.425 from survival-shape information alone, and the actual 2021 schedule approximately 1.00. In the latter, the local separation is only 0.18 standard errors. The 2022 wave occurs before the first quiet window; almost none of the relevant information distinguishes alpha from the other unknown parameters in this design. A theta=0.5 reference gives still poorer resolution (2021 pooled SE 7.85). Near the boundaries, wide nominal symmetric Wald ranges are not meaningful confidence intervals.

Conclusion: exact survivor fitting creates a possible additional temporal information source, but this diagnostic does not support identifying alpha from the Czech survival shape alone. The existing external-scale identification procedure is a different source of information; the fully specified external-scale procedure has not been tested; the historical illustrative empirical objective has now been replayed separately (see the provenance update below). The reported value 1.18 should remain sensitivity-only unless identification is demonstrated with the approved model and prespecified assumptions. No official method or paper change.

[Full cohort information](cohort-information.csv), [pooled benchmark](pooled-information.csv), [reproducible diagnostic](diagnostic.py).

## Scope of the restricted current-config shape-only diagnostic

The original compressed Czech registry file contains 12,597,668 newline-delimited data records excluding the header (not a claim that all are distinct eligible persons). The reported pairwise alpha=1.190 and collapse alpha=1.185 are the real pooled Czech objective minima in manuscript section 3.4.2. The earlier two-enrollment temporal-information diagnostic above did not reproduce that original estimation population or its cross-cohort objective.

The more recent checked-in alpha sandbox configuration (test/alpha/params_alpha.yaml) selects birth decades 1930, 1940, 1950, doses 0 and 2, and seven enrollment dates. Two enrollments (2022_26 and 2022_47) have no overlap with the configured wave 2021-41 through 2022-12. The five overlapping enrollments supply six age/dose cells apiece that pass the minimum wave death/week count checks on the current CMR workbook. These counts are not a reconstruction of the historical run's exact fit eligibility or its primary excess-ratio filtering.

| Enrollment | Selected people at enrollment | People at modeled origin | Temporal alpha SE, fitted-reference |
|---|---:|---:|---:|
| 2021_13 | 1,766,838 | 1,762,756 | 1.355 |
| 2021_20 | 1,307,736 | 1,305,340 | 1.020 |
| 2021_24 | 1,676,974 | 1,674,661 | 0.915 |
| 2021_30 | 2,005,005 | 2,002,623 | 0.940 |
| 2022_06 | 593,373 | 591,422 | about 166 |

The sum of selected enrollment counts is 7,349,926 cohort memberships, not unique people: enrollment snapshots of the same registry overlap. Do not combine them as independent observations for Fisher precision. These are exact current CMR selection counts and local theoretical shape-only SEs using the same modeling assumptions as the earlier diagnostic. They do not reproduce the original pairwise or collapse estimator, which uses cross-cohort wave distortion with different identifying assumptions. The original analysis may have additional signal unavailable to these shape-only calculations. The real-data preferred region was near 1.18–1.19, but the manuscript's identification checks failed. Whether the new machinery changes that conclusion requires specifically reevaluating the pooled original estimator, with its externally supplied scale and consistent alpha-dependent frailty fit rather than fixing alpha=1.18 to fit then estimating alpha from those same fitted quantities. The historical illustrative objective has subsequently been rerun; see the provenance update below. A fully specified external-scale estimator has not been tested.

[Original-scope population counts](original-scope-populations.csv), [local information by enrollment](original-scope-information.csv), [cohort information](original-scope-cohort-information.csv), [scope audit code](original_scope.py).

## Historical provenance and pooled replay (subsequent result)

The original 1.190/1.185 minima were reproduced exactly from commit e50c23ca and its verified historical CMR workbook. That historical selection includes doses 0-3, not only the later configuration's doses 0 and 2. It has 10,361,968 overlapping enrollment memberships before eligibility (48 eligible cohorts), but only 62 positive-excess cohort-week rows from eight cohorts contribute after filtering. With the all-window and exact nonlinear survivor-law fits, both original-grid optima reach the lower boundary and identification remains unsupported. On the wider grid the exact optima again reach its lower boundary. The historical illustrative objective does not supply the external scale assumed for formal identification. See [full historical replay](../alpha-pooled-replay/README.md). The shape-only SEs in the earlier tables remain specific to their stated current-config selection; they are not historical pooled-estimator SEs.
