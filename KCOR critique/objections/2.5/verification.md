# Iteration 2. N5 quiet-window scan

The [corrected scan](scan.py) uses enrollment ISO week 2021-24, the 1940s and 1950s birth decades, and doses 0, 1 and 2. The workbook is pinned to Git blob `63c60f1f020d0ee59c4c03e6f86d30014cca107f`, the version present when the historical scan was recorded. Reapplying the historical fitting calculation reproduced the 78 saved baseline scales, frailty variances, cumulative-hazard RMSEs and post-normalization R² values within numerical tolerance.

Each 12-month window starts one calendar month later than its predecessor, from April 2022 through April 2023. All 13 start and end weeks and the 52 or 53 observations in each window match the historical archive. The revised calculation sums deaths and people at risk by birth decade and dose before fitting. It retains time since enrollment, applies the stabilization skip, and calls the iteration 2 joint first-window fitter with the Czech configuration. For each placement the final refit includes the four early post-skip anchor weeks. Each attempt contributes once to the pass-rate denominator.

| Birth decade | Dose | Historical passes / attempts | Iteration 2 passes / attempts | Iteration 2 fitted frailty variance, min to max |
|---|---:|---:|---:|---:|
| 1940s | 0 | 8/13 | 13/13 | <0.001 to 9.33 |
| 1940s | 1 | 3/13 | 13/13 | <0.001 to 7.70 |
| 1940s | 2 | 8/13 | 13/13 | <0.001 to 6.92 |
| 1950s | 0 | 13/13 | 13/13 | <0.001 to 18.25 |
| 1950s | 1 | 13/13 | 13/13 | <0.001 to 14.51 |
| 1950s | 2 | 13/13 | 13/13 | <0.001 to 21.81 |

The historical numerator is the count of passing flags in the archived window-level results. Its published summary had calculated the mean flag after restricting to passes. The historical and revised columns use the diagnostic criteria belonging to each fitter. Their difference therefore cannot isolate the effect of changing the fitting equation. All 78 revised fits converged with status OK. The 13 windows per group overlap, and the fitted frailty varied across placements despite passing the diagnostics.

All 78 sets of parameters and diagnostic outcomes were independently checked through the ordinary workbook entry point with the joint fit selected. Figure S5 in each PDF was verified to contain the same pixels as the saved PNG.

From the iterative review directory, run `py -3.11 ../2.5/scan.py` to regenerate the [window-level estimates and diagnostics](windows.csv), [six group summaries](summary.csv), and [revised figure](<../../iterative review/revised/documentation/preprint/figures/fig_si_quiet_window_theta_scan_czech_2021_24.png>). Run `py -3.11 ../2.5/check_workbook.py` to compare each window with the ordinary workbook entry point using the joint fit.
