# Iteration 2. Weekly fit and Table 8

The comparison changes the weekly prediction used in fitting. The iteration 1 checkpoint and iteration 2 use the same Czech workbook, enrollment populations, subgroup overlap, fixed wave exponent (1.18), report date (December 26, 2022), pooling rule and person-bootstrap draws. The joint first-window fitter was selected explicitly in both runs. Adoption as the ordinary default still awaits lead approval.

## Paired comparison

Intervals are the 2.5th and 97.5th percentiles from 999 draws per enrollment. Changes below are computed before rounding.

| Enrollment | Pseudo-dose | Iteration 1 | Iteration 2 | Change | Iteration 1 interval | Iteration 2 interval |
|---|---|---:|---:|---:|---|---|
| 2021_24 | 1 vs 0 | 1.00905 | 1.00834 | -0.00071 | [0.9548, 1.0576] | [0.9553, 1.0563] |
| 2021_24 | 2 vs 0 | 1.03174 | 1.03137 | -0.00037 | [0.9709, 1.0940] | [0.9712, 1.0945] |
| 2021_24 | 2 vs 1 | 1.02249 | 1.02285 | +0.00036 | [0.9618, 1.0871] | [0.9622, 1.0871] |
| 2022_06 | 1 vs 0 | 1.05034 | 1.05393 | +0.00359 | [0.2332, 1.2987] | [0.3218, 1.2725] |
| 2022_06 | 2 vs 0 | 1.05750 | 1.05945 | +0.00195 | [0.2381, 1.2342] | [0.3312, 1.2203] |
| 2022_06 | 2 vs 1 | 1.00682 | 1.00524 | -0.00158 | [0.8059, 1.2093] | [0.8182, 1.2083] |

All six revised intervals contain 1. The largest point change was +0.00359 for 2022, 1 vs 0. The largest endpoint change was +0.09310 for the 2022 lower limit, 2 vs 0. The 2022 lower limits for both comparisons against pseudo-dose 0 rose appreciably; both intervals remain wide.

## Verification

The first-window fit and pooled refit use the exact cumulative Gompertz baseline at week entry and its increment through week end. The existing within-week offset is held fixed across those two boundaries. The N2 change also updates seed and final residuals. The existing gap-end indexing, resampling and wave correction are preserved.

Five zero/positive frailty values passed a forward-and-inverse round trip at three start exposures. A joint fit to independently generated one-window expected weekly hazards recovered the generating baseline rate 0.004 and frailty variance 4 to relative tolerance 0.00001. The 18 original-sample component fits were replayed under both versions. The largest applied frailty variance change was from 23.72927 to 24.56730 for the 2022 unvaccinated subgroup, pseudo-dose 2.

All six iteration 2 point estimates matched independently pooled output from the full workbook entry point at relative tolerance 1e-10. All 17,982 bootstrap component fits returned successfully, including configured fallbacks. Original-sample insufficient-signal flags remained on the same four of nine components in 2022 and zero in 2021. Across bootstrap draws, at least one component triggered that flag as follows.

| Enrollment | Iteration 1 draws | Iteration 2 draws |
|---|---:|---:|
| 2021_24 | 512/999 | 271/999 |
| 2022_06 | 998/999 | 999/999 |

The paired replay measures sensitivity of the reported results to N2. It does not estimate interval coverage.

From the iterative review directory run `py -3.11 ../2.2/check.py` for the mathematical and component-fit checks, then `py -3.11 ../1.6/replay.py --iteration-2` for the workbook replay and intervals. [Iteration 1 results](<../../iterative review/checkpoints/iteration-1/rounds/round-06-weekly-inversion/table8_bootstrap.json>) · [Iteration 2 results](table8_bootstrap.json) · [Paired component fits](original_fits.json)
