# Verification of 3.1

## Table 8 before and after the gap correction

Both editions used the same 999 person-bootstrap draws per enrollment, the joint first-window fit, the archived Czech input and the December 26, 2022 reporting date. Every pooled contrast used all 999 draws, including draws that triggered a fallback. Limits are the 2.5th and 97.5th percentiles. Width is the difference between the unrounded limits.

| Enrollment | Pseudo-dose | Iteration 2 KCOR [95% interval] | Width | Iteration 3 KCOR [95% interval] | Width | Change in width |
|---|---|---:|---:|---:|---:|---:|
| 2021_24 | 1 vs 0 | 1.0083 [0.9553, 1.0562] | 0.1009 | 1.0083 [0.9564, 1.0584] | 0.1021 | +0.0012 (+1.2%) |
| 2021_24 | 2 vs 0 | 1.0312 [0.9711, 1.0943] | 0.1232 | 1.0312 [0.9739, 1.0944] | 0.1205 | -0.0027 (-2.2%) |
| 2021_24 | 2 vs 1 | 1.0227 [0.9618, 1.0868] | 0.1251 | 1.0227 [0.9640, 1.0889] | 0.1249 | -0.0002 (-0.1%) |
| 2022_06 | 1 vs 0 | 1.0462 [0.3218, 1.2680] | 0.9462 | 1.0302 [0.3219, 1.2358] | 0.9138 | -0.0323 (-3.4%) |
| 2022_06 | 2 vs 0 | 1.0677 [0.3351, 1.2130] | 0.8779 | 1.0724 [0.3414, 1.1928] | 0.8514 | -0.0265 (-3.0%) |
| 2022_06 | 2 vs 1 | 1.0205 [0.8224, 1.2083] | 0.3859 | 1.0410 [0.8472, 1.2133] | 0.3661 | -0.0197 (-5.1%) |

## Subgroup diagnostics

In the original 2021 sample both editions had zero insufficient-signal flags. In the original 2022 sample the two-dose subgroup at pseudo-dose 2 and the unvaccinated subgroup at pseudo-dose 0 triggered the gap reset in iteration 2. Both proceeded to a fit in iteration 3. The two-dose and all-persons subgroups at pseudo-dose 1 reached the upper variance bound in both editions.

| Enrollment | Draws with any insufficient-signal flag before | After | Unsuccessful fit calls after |
|---|---:|---:|---:|
| 2021_24 | 307/999 | 0/999 | 0 |
| 2022_06 | 999/999 | 932/999 | 0 |

All 2,217 revised 2022 insufficient-signal flags occurred when a subgroup fit reached the upper frailty bound. All revised 2021 and 2022 numerical fits returned successfully, including the configured fallback at that bound.

The following counts are per subgroup across 999 resamples. A gap set to zero can also be followed by a bound hit. Each occurrence is counted in both corresponding columns. A nonconverged fit is counted as unsuccessful even when its status field reads `ok`. The data include separate records for the original sample (draw 0).

| Enrollment | Subgroup | Pseudo-dose | Earlier gap resets | Earlier insufficient signal | Revised fits with any gap set to zero | Revised upper bound | Revised poor fit | Revised insufficient signal | Revised unsuccessful |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021_24 | two-dose | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021_24 | two-dose | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021_24 | two-dose | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021_24 | unvaccinated | 0 | 1 | 1 | 971 | 0 | 0 | 0 | 0 |
| 2021_24 | unvaccinated | 1 | 76 | 76 | 728 | 0 | 0 | 0 | 0 |
| 2021_24 | unvaccinated | 2 | 19 | 19 | 947 | 0 | 0 | 0 | 0 |
| 2021_24 | all-persons | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021_24 | all-persons | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021_24 | all-persons | 2 | 239 | 239 | 239 | 0 | 0 | 0 | 0 |
| 2022_06 | two-dose | 0 | 599 | 630 | 850 | 169 | 0 | 169 | 0 |
| 2022_06 | two-dose | 1 | 553 | 689 | 827 | 599 | 0 | 599 | 0 |
| 2022_06 | two-dose | 2 | 617 | 626 | 913 | 9 | 0 | 9 | 0 |
| 2022_06 | unvaccinated | 0 | 820 | 830 | 890 | 18 | 0 | 18 | 0 |
| 2022_06 | unvaccinated | 1 | 495 | 638 | 776 | 437 | 0 | 437 | 0 |
| 2022_06 | unvaccinated | 2 | 247 | 565 | 606 | 321 | 0 | 321 | 0 |
| 2022_06 | all-persons | 0 | 222 | 229 | 489 | 7 | 0 | 7 | 0 |
| 2022_06 | all-persons | 1 | 312 | 657 | 638 | 657 | 0 | 657 | 0 |
| 2022_06 | all-persons | 2 | 240 | 240 | 508 | 0 | 0 | 0 | 0 |

## Reproduction

From the repository root, run `py -3.11 -B "KCOR critique/objections/3.1/replay.py"`. The replay asserts agreement with all six archived pre-change points and their twelve percentile limits. The fixed-rate sensitivity analysis uses the same draws and is included in the data. Run `py -3.11 -B "KCOR critique/objections/3.1/check.py" --workbook` for the five noiseless gap tests and an independent comparison of the six corrected points with the ordinary joint-fit workbook.

[Summary and percentile limits](table8_bootstrap.json) · [All paired estimates by draw](table8-draws.npz) · [All diagnostic flags by subgroup and draw](subgroup-diagnostics.csv.gz) · [Earlier estimates and limits](../2.4/table8_bootstrap.json)
