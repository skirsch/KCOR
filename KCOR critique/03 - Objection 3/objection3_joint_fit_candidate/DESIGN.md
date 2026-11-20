# Pilot design fixed before outcomes

This separate module tests a candidate change to KCOR's gamma-frailty parameter fit. It imports the corrected source snapshot for the unchanged hazard inversion and normalization. It does not edit production code, the PR, or completed objection 3 results.

## Candidate

For a single contiguous trusted quiet window, condition on people still alive at its first week. Model the sequence of conditional weekly deaths using exact discrete gamma-frailty survival with two unknowns: effective post-selection baseline rate at the window start and frailty parameter. The known Gompertz aging slope remains fixed. Estimate both unknowns together by conditional binomial likelihood of deaths given each week's risk set. Apply the existing KCOR gamma inversion to observed hazards and retain its normalization at week 5. The first two skipped weeks and weeks before the trusted window are not used to fit the candidate parameters.

For this pilot the trusted quiet window is weeks 6–40, the skip is weeks 0–1, and the reporting point is week 52. This candidate is **not yet a solution for multiple disjoint quiet windows**; they require a specified treatment of the untrusted gaps. Optional wave correction is disabled as in the preceding validation.

## Fixed checks and bounded pilot

1. Check the exact weekly gamma model on deterministic expected population counts for the earlier `unequal_large` scenario (true frailties 4 and 0.5, one million people, true normalized ratio 1). The candidate should recover the frailties and ratio to numerical precision using only weeks 6–40, from several starting values. Check the source model's prior exact-population result for context.
2. Check independent use of the corrected production inversion and normalization using a direct array calculation. Do not claim workbook equivalence for the *new* fitter, which does not exist in production.
3. Run a fresh, bounded pilot of 60 independent one-million-person datasets, each with 199 pooled person-bootstrap draws. The seed namespace is 672913. The original sample is excluded from bootstrap quantiles. Record candidate percentile and basic coverage, interval widths, bias, fit success, lower-bound frequency, and numerical failures. Refit both parameters in every bootstrap sample. The pilot uses one worker to limit laptop CPU use. Retain all draws and per-dataset results.
4. The pilot is exploratory and is **not** a definitive 95% coverage validation: 60 datasets have wide binomial uncertainty. If exact-population checks fail, stop before the pilot. If the candidate is numerically sound, report its bias-variance and coverage behavior without choosing a preferred interval based on these results. A later prespecified full validation with fresh seeds would be required before any production change.

The baseline comparator is the completed corrected-code validation, especially its 600-dataset large-population result. The candidate does not change the resampling unit, target, skipped weeks, or normalization. Reusing the old coverage seeds would not be fresh validation; this pilot uses a new namespace.
