# Iteration 2. N3 gap offset

Step 3 now reads the reconstructed and expected cumulative hazards at the start of the first quiet week after the gap. Both totals include the last gap week. The offset takes effect at this same boundary in the pooled refit and in the final fit diagnostics.

## Boundary checks

Using exact weekly observations, the saved N2 fitter and the N3 fitter both included a 0.010 increment in week 39 of a gap ending in week 40. Only N3 included the same increment when placed in week 40. Both excluded it when placed in quiet week 41. The no-pulse control gave zero to numerical precision. These are first-pass raw offsets. A full fit with separate increments of 0.010 and 0.006 in two final gap weeks, including a one-week gap, recovered both offsets, the generating baseline rate and frailty, and final residual RMSE below 1e-9. [Executable checks](check.py).

## Paired Table 8 replay

The reference is the N2 joint fit in iteration 2. The Czech workbook, pooled enrollment populations, draw seeds, 999 person-bootstrap draws per enrollment, wave exponent 1.18, and December 26, 2022 report date are identical. All six N3 point estimates matched the ordinary full-workbook replay with relative tolerance 1e-10.

| Enrollment | Pseudo-dose | Before N3 | After N3 | Change | 95% interval before | 95% interval after |
|---|---|---:|---:|---:|---:|---:|
| 2021_24 | 1 vs 0 | 1.008337 | 1.008354 | +0.000017 | [0.9553, 1.0563] | [0.9553, 1.0563] |
| 2021_24 | 2 vs 0 | 1.031373 | 1.031346 | -0.000027 | [0.9712, 1.0945] | [0.9712, 1.0945] |
| 2021_24 | 2 vs 1 | 1.022845 | 1.022802 | -0.000043 | [0.9622, 1.0871] | [0.9618, 1.0870] |
| 2022_06 | 1 vs 0 | 1.053926 | 1.053922 | -0.000004 | [0.3218, 1.2725] | [0.3218, 1.2710] |
| 2022_06 | 2 vs 0 | 1.059449 | 1.059428 | -0.000021 | [0.3312, 1.2203] | [0.3312, 1.2169] |
| 2022_06 | 2 vs 1 | 1.005241 | 1.005225 | -0.000016 | [0.8182, 1.2083] | [0.8185, 1.2046] |

The largest point change was a decrease of 0.0000433 for 2021, 2 vs 1. The largest interval endpoint change was a decrease of 0.00370 in the 2022 upper limit for 2 vs 1. All six intervals include 1. The counts below describe the number of original component fits with an insufficient-signal flag and the number of bootstrap draws with any such flag among their nine component fits.

| Enrollment | Original flags before | Original flags after | Flagged draws before | Flagged draws after |
|---|---:|---:|---:|---:|
| 2021_24 | 0/9 | 0/9 | 271/999 | 307/999 |
| 2022_06 | 4/9 | 4/9 | 999/999 | 999/999 |

All 17,982 bootstrap component fits returned successfully, including configured fallbacks. From the iterative review directory run `py -3.11 ../2.3/check.py` for the boundary tests and `py -3.11 ../1.6/replay.py --gap-boundary` for the workbook and bootstrap replay. The `--iteration-2` option reproduces the earlier N2 calculation with its saved fitter. [Before N3](../2.2/table8_bootstrap.json) and [after N3](table8_bootstrap.json).
