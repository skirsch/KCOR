# 2.2 follow-up · Weekly Gompertz reference during a wave

The optional wave correction compares the observed cumulative hazard after inversion with a fitted no-wave reference. The earlier recurrence added the Gompertz rate at week entry as one week's baseline hazard. With starting rate $r$ and weekly slope $\gamma$, the baseline hazard for that week is $\int_0^1 r e^{\gamma u}\,du=r(e^\gamma-1)/\gamma$. The multiplier is 1 for a flat baseline and 1.00081775 at the configured Czech slope. Section 2.7.3 applies this increment to the first modeled wave week and each subsequent week within the window. SI §S2.1 describes the same calculation.

## Table 8

The before and after runs used the same archived Czech workbook, the same 999 person-bootstrap draws per enrollment, the joint first-window fit, wave exponent 1.18, and December 26, 2022 reporting date. The comparison includes every draw. The paired interval for a change is obtained by subtracting the earlier result from the revised result within each draw and taking the 2.5th and 97.5th percentiles. Original-sample estimates are excluded from the percentiles.

| Enrollment | Pseudo-dose | Before [95% interval] | After [95% interval] | Paired change [95% interval] | Width before / after |
|---|---|---:|---:|---:|---:|
| 2021_24 | 1 vs 0 | 1.008317 [0.956372, 1.058434] | 1.008322 [0.956378, 1.058436] | +0.0000050 [+0.0000022, +0.0000080] | 0.102062 / 0.102058 |
| 2021_24 | 2 vs 0 | 1.031238 [0.973882, 1.094428] | 1.031261 [0.973904, 1.094452] | +0.0000223 [+0.0000149, +0.0000270] | 0.120546 / 0.120548 |
| 2021_24 | 2 vs 1 | 1.022733 [0.963978, 1.088853] | 1.022750 [0.963995, 1.088868] | +0.0000170 [+0.0000094, +0.0000223] | 0.124876 / 0.124873 |
| 2022_06 | 1 vs 0 | 1.030160 [0.321918, 1.235755] | 1.030193 [0.321927, 1.235764] | +0.0000333 [-0.0000247, +0.0000564] | 0.913836 / 0.913837 |
| 2022_06 | 2 vs 0 | 1.072422 [0.341380, 1.192808] | 1.072358 [0.341336, 1.192704] | -0.0000636 [-0.0001484, +0.0000402] | 0.851427 / 0.851367 |
| 2022_06 | 2 vs 1 | 1.041024 [0.847185, 1.213328] | 1.040929 [0.847118, 1.213319] | -0.0000954 [-0.0001668, +0.0000248] | 0.366144 / 0.366200 |

All six revised intervals include 1. The original-sample insufficient-signal counts were 0/9 for 2021 and 2/9 for 2022. Across the 999 draws per enrollment, 0 and 932 draws respectively included at least one such flag, as before. All component fits returned successfully, including configured fallbacks.

## Verification and reproduction

From the repository root, run

    py -3.11 -B "KCOR critique/objections/2.2/wave-reference/check.py" --workbook
    py -3.11 -B "KCOR critique/objections/2.2/wave-reference/replay.py"

The boundary tests compare the production correction with numerical integration for positive, negative, zero and near-zero slopes. They cover a partial wave during the stabilization skip, a one-week wave, subsequent weeks, a known wave excess and the proportional identity. The full joint-fit workbook independently reproduced all six revised original-sample estimates. The replay checks archived draws against the [pre-change method](../../3.1/code/KCOR.py) before recomputing the joint and fixed-rate results.

[Evaluation finding](<../../../iterative review/evaluations/3/checks.md>) · [Revised estimates and intervals, including fixed-rate sensitivity](table8_bootstrap.json) · [Revised estimates for each draw](table8-draws.npz) · [Prior estimates and intervals](../../3.1/table8_bootstrap.json)
