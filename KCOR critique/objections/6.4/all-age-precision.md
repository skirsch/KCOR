# S6 ? matched frailty precision across all age bands

All 14 original frailty estimates matched the m120 manuscript replay. The 40-49 all-window draws match the previous targeted 999 draws to numerical precision. For each of seven age bands (including the full aggregate), 999 matched person-bootstrap samples were fitted by the current all-window method and by the very same fitter restricted to its first contiguous quiet window. The latter is an information comparison using the current model, not a reconstruction of every difference in the historical fitter. There are 13,986 paired-cohort fit results. All cohorts use the same 149-week endpoint and prespecified reference week; the actual disjoint 5-year birth-band/dose enrollment sizes are conditioned on, and the full aggregate includes unknown ages and reuses the same draws.

## Frailty intervals and theoretical low-variance resolution

Raw optimizer estimates and their 2.5/97.5 percentiles are shown because an applied estimate of zero can reflect the upper-bound fallback. The local information SE at reference theta=0.5 profiles the baseline and all available gap exposures, with actual fit rows and aggregate gap survival transitions. The local approximation measures resolution conditional on the model, and the boundary percentile intervals describe the constrained estimator's sampling distribution, not guaranteed confidence coverage for the underlying parameter.

| Age | Dose | All-window theta [95% raw interval] | First-window theta [95% raw interval] | SE first | SE all | Raw upper-bound flags all / 999 | Fitted deaths all |
|---|---:|---|---|---:|---:|---:|---:|
| 40-49 | 0 | 63.87 [29.60, 96.51] | ~0 [~0, 100.00] | 470.93 | 17.44 | 17 | 1095 |
| 40-49 | 2 | 71.83 [~0, 100.00] | ~0 [~0, 100.00] | 1712.97 | 55.15 | 287 | 182 |
| 50-59 | 0 | 26.82 [18.43, 35.48] | 85.35 [~0, 100.00] | 122.51 | 4.48 | 0 | 2037 |
| 50-59 | 2 | ~0 [~0, ~0] | ~0 [~0, 100.00] | 636.51 | 14.21 | 0 | 509 |
| 60-69 | 0 | 11.47 [9.37, 13.50] | 55.95 [5.20, 100.00] | 28.65 | 1.07 | 0 | 3787 |
| 60-69 | 2 | ~0 [~0, ~0] | ~0 [~0, 100.00] | 142.82 | 3.08 | 0 | 1848 |
| 70-79 | 0 | 5.16 [4.60, 5.71] | 18.95 [4.69, 34.89] | 7.79 | 0.30 | 0 | 6077 |
| 70-79 | 2 | ~0 [~0, ~0] | ~0 [~0, ~0] | 27.22 | 0.58 | 0 | 8325 |
| 80-89 | 0 | 1.70 [1.45, 1.94] | 13.99 [6.49, 21.54] | 3.60 | 0.13 | 0 | 5498 |
| 80-89 | 2 | ~0 [~0, ~0] | ~0 [~0, 2.36] | 7.41 | 0.16 | 0 | 12436 |
| 90-99 | 0 | 0.51 [0.39, 0.62] | 0.78 [~0, 4.29] | 1.80 | 0.06 | 0 | 3594 |
| 90-99 | 2 | ~0 [~0, ~0] | ~0 [~0, 0.51] | 2.60 | 0.06 | 0 | 8243 |
| All ages | 0 | 31.06 [29.38, 32.65] | 100.00 [74.95, 100.00] | 22.82 | 0.90 | 0 | 28013 |
| All ages | 2 | ~0 [~0, 0.28] | ~0 [~0, ~0] | 9.43 | 0.23 | 0 | 32056 |

## Endpoint KCOR with the actual fallback

| Age | First-window KCOR [95%] | All-window KCOR [95%] |
|---|---|---|
| 40-49 | 1.537 [0.929, 3.009] | 1.507 [0.824, 2.712] |
| 50-59 | 0.755 [0.668, 2.287] | 1.228 [0.913, 1.824] |
| 60-69 | 0.380 [0.118, 1.934] | 1.295 [1.086, 1.577] |
| 70-79 | 0.468 [0.121, 1.308] | 1.301 [1.173, 1.441] |
| 80-89 | 0.099 [0.016, 0.506] | 1.204 [1.108, 1.312] |
| 90-99 | 0.904 [0.176, 1.318] | 1.012 [0.919, 1.114] |
| All ages | 1.556 [0.518, 1.641] | 1.136 [1.086, 1.189] |

All 999 draws per band and mode yielded finite ratios. The point contrasts and intervals are descriptive and conditional on the cohort construction and model. First-window intervals can conceal uncertainty when the variance repeatedly reaches a bound. In the dose-2 40-49 group, the all-window information SE is lower by a factor of 31, yet the raw interval still spans the allowed range. Ages 50-59 and 60-69 dose 2 have bootstrap estimates piled at zero (991/999 and 979/999 below 1e-6), but their low-variance information SEs are 14.21 and 3.08. A boundary-piled bootstrap does not establish exact zero true frailty. Older cohorts have substantially more absolute curvature information.

## Why the 40-49 KCOR interval is wide

The normalized cumulative-hazard ratio at the reference is 0.50835 and at the end is 0.76634; their ratio is 1.50749. There were 116 dose-0 and just 13 dose-2 deaths in the four accumulating reference weeks. In the same draws, fixing both frailty values at their original estimates gives KCOR [0.911, 2.948]. Fixing the reference ratio at its original value while retaining the refitted endpoint gives [1.188, 1.707]. Both are diagnostics under artificially fixed quantities, not valid uncertainty intervals for the original estimator. The original interval is [0.824, 2.712]. The fixed-reference experiment demonstrates substantial reference uncertainty; dependencies between theta and the reference prevent simple additive attribution of variance. The previous targeted bootstrap and these exact recomputed draws agree.

[Full results](all-age-precision.json) | [Matched draws](all-age-precision-draws.csv) | [Replay](all-age-precision.py) | [Anchor diagnostic](anchor-precision.json) | [Anchor replay](anchor-precision.py). No manuscript or method change in this investigation; discuss reporting of weakly resolved estimates with lead author. Related to 2.7 and 5.11.
