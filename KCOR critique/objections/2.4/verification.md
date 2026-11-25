# Iteration 2. N4 wave entry

The optional nonlinear wave correction now initializes its no-wave reference from the cumulative hazard immediately before the first modeled wave week. It then adds the fitted baseline increment for that week using the existing wave-path recurrence. When enrollment overlaps an ongoing calendar wave, the first modeled week follows the stabilization skip and starts from zero. Earlier wave exposure outside modeled follow-up is not estimated.

## Boundary checks

With a one-week wave, an opening cumulative hazard of 0.020, a fitted weekly increment of 0.001 and an observed inverted total of 0.026, the reference is 0.021. The N4 calculation passes 0.005 of positive excess to the existing rescaling and carries the rescaled exposure to wave exit. The N3 calculation assigned zero. Tests cover a two-week wave, a wave at the start of follow-up, partial and complete overlap with skipped weeks, observed totals at or below baseline, and the proportional identity. [Executable checks](check.py).

## Table 8

The same Czech workbook, 999 person-bootstrap draws per enrollment, reporting date, joint fitter, pooling rule and fixed wave exponent of 1.18 were used for both versions. All six original-sample estimates matched the full workbook run with relative tolerance 1e-10.

| Enrollment | Pseudo-dose | Before N4 | After N4 | Change | 95% interval before | 95% interval after |
|---|---|---:|---:|---:|---:|---:|
| 2021_24 | 1 vs 0 | 1.008354 | 1.008317 | -0.000037 | [0.9553, 1.0563] | [0.9553, 1.0562] |
| 2021_24 | 2 vs 0 | 1.031346 | 1.031238 | -0.000108 | [0.9712, 1.0945] | [0.9711, 1.0943] |
| 2021_24 | 2 vs 1 | 1.022802 | 1.022733 | -0.000069 | [0.9618, 1.0870] | [0.9618, 1.0868] |
| 2022_06 | 1 vs 0 | 1.053922 | 1.046204 | -0.007718 | [0.3218, 1.2710] | [0.3218, 1.2680] |
| 2022_06 | 2 vs 0 | 1.059428 | 1.067703 | +0.008275 | [0.3312, 1.2169] | [0.3351, 1.2130] |
| 2022_06 | 2 vs 1 | 1.005225 | 1.020550 | +0.015325 | [0.8185, 1.2046] | [0.8224, 1.2083] |

All six intervals include 1. The largest point movement was the 2022 pseudo-dose 2 vs 1 increase from 1.005225 to 1.020550. The 2022 enrollment began during the configured calendar wave; its modeled observation starts after the stabilization skip. The 2021 enrollment already had modeled follow-up before wave entry.

The fit precedes the N4 calculation. Original-sample insufficient-signal flags stayed at 0/9 and 4/9 in 2021 and 2022 respectively. The numbers of draws with any flagged component remained 307/999 and 999/999. All 17,982 bootstrap component fits returned successfully, including configured fallbacks.

From the iterative review directory run `py -3.11 ../2.4/check.py` and `py -3.11 ../1.6/replay.py --wave-entry`. The `--gap-boundary` option reproduces the earlier result with its archived N3 code. [Before N4](../2.3/table8_bootstrap.json) and [after N4](table8_bootstrap.json).
