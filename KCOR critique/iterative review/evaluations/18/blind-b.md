# Independent review of KCOR, frozen candidate m181

## Assessment

The core proportional gamma inversion and the paper’s conditional interpretation are defensible. For mean-one gamma frailty with variance θ, marginal survival is (1 + θH₀)^(−1/θ), which gives the stated inversion and its continuous limit at zero. With a known Gompertz slope the quiet-period cohort hazard satisfies h′ = γh − θh² under proportional selection. This establishes the model identity. A fit to observed deaths alone cannot establish that the model generated those deaths. The manuscript generally draws that distinction. I verified two errors in Monte Carlo workbook reporting. I found no evidence that either changes the manuscript’s Czech primary points.

## Corrections that affect reported workbook results

1. **The direct all-ages raw comparator can be wrong for every replicate in a batch.** I ran the ordinary Monte Carlo workbook entry point with the candidate’s 20-week fixed-cohort fixture, with two different mortality trajectories. I computed the all-ages anchored raw comparator independently by summing counts by replicate, dose and week, accumulating the post-skip weekly negative log survival, then dividing the endpoint ratio by its reference ratio. The two endpoint values were **1.5719147768** and **1.0027192274**. With the disjoint age rows supplied, both output cells for `KCOR_ns` were blank. With exactly the same trajectories supplied as all-ages totals, the written workbook instead gave **1.2862082260 for both replicates**. The normalized KCOR column retained the two separate values in this fixture. In `code/KCOR.py`, the all-ages branch of `build_kcor_ns_rows` groups by dose and date at line 5730, accumulates by dose alone, then emits rows without a replicate ID. The merge around line 8175 either cannot match these rows or broadcasts the combined estimate. Carry the replicate ID through aggregation, cumulative sums, numerator–denominator pairing, reference selection and output. Verify the raw comparator against direct arithmetic for both layouts. This affects comparisons between normalized and raw trajectories in simulation output.

2. **Partial Monte Carlo populations can be labeled “All Ages” even when the compared birth groups differ.** I also ran a mixed workbook with `KCOR_AGGREGATE_SCOPE=partial`. Replicate 1 provided only the 1940 group for the numerator and 1940 plus 1950 for the denominator, with no supplied full total. Replicate 2 provided both groups and full totals. Both direct aggregate rows said “All Ages,” with blank age-group and coverage-status fields. The first endpoint, **0.9992592**, compares the available groups on each side. The label gives it the same population description as the second replicate’s supplied total. The aggregate selection already distinguishes the replicates, but the reporting condition around line 6460 excludes Monte Carlo mode and stores coverage at sheet level. Apply the existing available-ages labeling and coverage rules per replicate and dose. This is a population reporting error even where the underlying arithmetic is correct.

## Scientific interpretation and verified sensitivity

I independently refit all seven S6 age rows using the archived 2021-24 CMR sheet, the same two-week skip, reference, quiet windows, ages and dose pair, and the production fitter. At the fixed annual Gompertz slope of 0.085, all seven estimates agreed with the archived primary points within 0.0000001. Changing only the shared slope to 0.065 or 0.105 gave the following examples.

| Age in 2024 | Slope 0.065 | Slope 0.085 | Slope 0.105 |
|---|---:|---:|---:|
| 40–49 | 1.5074 | 1.5075 | 1.5075 |
| 70–79 | 1.3399 | 1.3013 | 1.2636 |
| 90–99 | 1.0411 | 1.0122 | 0.9838 |
| All eligible ages | 1.1691 | 1.1356 | 1.1274 |

The direction of the oldest group’s point relative to one depends on the slope within this range. Its interval at 0.085 already spans one, [0.919, 1.114], so this check does not overturn a positive finding. Section 2.4.3 identifies its 0.4% sensitivity result as a 40-week example. If the empirical magnitudes are discussed, the matched real-data points above are a more relevant illustration of the slope assumption. These are sensitivity points. I did not repeat the full bootstrap at alternative slopes.

The empirical age shifts supply no known exact null target. The candidate reports residual flags in all 999 primary Table 8 draws for every comparison, the historical outcome-informed preference for a window, and the benefit positive control’s signed-gap failure. The S6 eligibility rule uses future vaccine products. The authors also report that 363,033 people were excluded solely for a disqualifying product in the enrollment week or later. These disclosures limit claims of external validation and causal attribution. They do not establish that the supplied arithmetic is wrong. As a narrow editorial correction, section 3.1.2 can describe the age-shift exercise without asserting that it operated “under model-consistent conditions,” since those conditions were assumed and the reported residual screen gave evidence of discrepancy. I see no need to replace the estimator or the paper’s model-conditional estimand.

## Verification scope

I inspected only the frozen m181 candidate. I checked the algebra, the reported diagnostics, the relevant code and the S6 point calculations, and reproduced the Monte Carlo results through the workbook entry point using temporary input and output files. The raw-comparator example used the input constructor in `test/test_round7_workbook.py` and the run helper in `test/test_round16_mc_pooling.py`, tripling deaths from week-index 8 onward in the 1940 numerator of replicate 1 and rebuilding its weekly risk sets. I checked both returned rows and written output for the raw comparator. I did not regenerate the published bootstrap ensembles.
