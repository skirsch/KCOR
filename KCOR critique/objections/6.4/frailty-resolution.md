# Frailty precision in the Czech 40-49 age band

The S6.1.2 replay gave a dose-2 frailty variance of 71.83. Its numerical fit passed the descriptive residual screen. This targeted investigation uses the same age grouping (birth years 1975-1984), 2021-24 enrollment, module-off current fitter and follow-up through 2024-16. The original point estimates match the manuscript replay to numerical precision.

| Metric | Dose 0 | Dose 2 |
|---|---:|---:|
| Enrollment population | 767,923 | 168,932 |
| Deaths in 149 follow-up weeks | 4,075 | 699 |
| Deaths in 41 fitted weeks | 1,095 | 182 |
| Fitted weekly baseline rate | 0.00003679 | 0.00002711 |
| Fitted frailty variance | 63.87 | 71.83 |
| Local SE at theta=0.5 | 17.44 | 55.15 |
| Local SE at fitted theta | 16.90 | 53.72 |

The local Fisher calculation uses all actual fitted rows and gap survival transitions, profiling the baseline rate and cumulative gap offsets. The low-frailty reference for each cohort is theta=0.5 with the baseline rate estimated conditionally at that reference. These regular model-conditional local limits are not coverage-guaranteed intervals at the variance boundary.

## Person bootstrap

999 draws with seed 20261126, conditioning on observed opening size of each five-year birth-band/dose group. Verified death-only fixed histories permit multinomial resampling of each person's death week or end-of-follow-up survivor category. Age bands are combined before each refit. Both the raw optimizer estimate and the applied estimator are retained, including the existing upper-bound fallback.

| Metric | Dose 0 | Dose 2 |
|---|---:|---:|
| Raw theta 95% percentile interval | [29.60, 96.51] | [0.00, 100.00] |
| Applied theta 95% percentile interval | [24.03, 93.14] | [0.00, 95.58] |
| Raw theta reached upper-bound flag | 17/999 | 287/999 |
| Raw theta below 1e-6 | 2/999 | 78/999 |
| Applied theta below 1e-6 | 19/999 | 365/999 |

Anchored dose-2/dose-0 KCOR at week 2024-16: 1.5075 (95% percentile interval [0.8239, 2.7119]). All draws are retained. This describes the configured comparison, not a causal contrast.

The dose-2 variance is poorly resolved. The residual screen can accept a fitted mortality trajectory even when its curvature parameters are imprecise. The raw bootstrap spans effectively zero to the optimizer's upper limit of 100, and the latter triggers the existing zero-variance fallback in 287 draws. Neither the original 71.83 nor an imposed zero can be inferred precisely from these data. The result is linked to the established low-frailty information limit (2.7) and the unresolved fallback policy (5.11). No fitter or manuscript changes made in this investigation.

[Script](frailty-resolution.py) | [Results](frailty-resolution.json) | [Draws](frailty-resolution-draws.csv).
