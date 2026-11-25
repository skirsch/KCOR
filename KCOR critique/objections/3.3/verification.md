# Table 9 · Positive-control replay with the current joint fitter

The input was the archived positive-control workbook with its original counts, injections and dates. Each scenario was copied into a separate workbook with the enrollment sheet name `2021_24`, which the ordinary workbook reader accepts. The fit used the revised edition after 3.1, the positive-control dataset defaults, a joint first-window fit of rate and frailty, and dose 1 versus dose 0 for the 1950 birth cohort with both sexes pooled. The optional wave correction was off. The fitted endpoint and both cohort fits were cross-checked against the ordinary workbook run for each scenario.

| Scenario and injected weekly multiplier | Historical raw ratio [95% interval] | Raw on fitted timeline [95% interval] | Fitted KCOR [95% interval] |
|---|---:|---:|---:|
| Benefit, 0.8 | 0.8246 [0.7556, 0.8984] | 0.8225 [0.7440, 0.9014] | 0.8363 [0.7337, 0.9473] |
| Harm, 1.2 | 1.1070 [1.0120, 1.2091] | 1.1132 [1.0125, 1.2292] | 1.1852 [1.0342, 1.3680] |

The injection covers zero-based weeks 20–79. The endpoint is week 80 (December 26, 2022). The historical raw calculation starts in week 0 and anchors at week 4. The second and third columns of estimates skip weeks 0–1 and anchor at week 5. Comparing the raw and fitted estimates on that common timeline isolates the frailty adjustment. The 52 quiet weeks are weeks 81–132 (calendar 2023), after the endpoint. The final refit also includes four early post-skip anchor weeks. This is a retrospective directional check. The injected multiplier acts on the treated cohort’s weekly hazard during the injection period, while the reported ratio is cumulative and anchored.

| Scenario | Dose | Generator variance | Applied variance [95% interval] | Fit status | Flagged or unsuccessful bootstrap fits |
|---|---:|---:|---:|---|---:|
| Benefit and harm (shared control) | 0 | 0.5 | 1.420 [0.586, 2.225] | ok | 0/999 per scenario |
| Benefit | 1 | 1.0 | 2.033 [1.097, 3.000] | ok | 0/999 |
| Harm | 1 | 1.0 | 2.026 [1.037, 2.922] | ok | 0/999 |

All four original-sample fits converged without an insufficient-signal flag or upper-bound hit. The generating baseline hazard is flat and the fitted model uses its default Gompertz slope of 0.085 per year. Directions were recovered with these settings. The fitted variances exceeded the generating values.

### How the intervals were obtained

The source workbook contains rounded deterministic expected counts. The intervals describe variation under resampling from the represented cohorts. For each scenario, 999 draws resampled 100,000 people per dose with replacement, preserving death week, nondeath exit week or survival to the end. Each draw used the same counts for its three estimates, with both cohorts refitted. The identical dose-0 cohort received the same draw in both scenarios. Limits are the 2.5th and 97.5th percentiles. These are newly calculated intervals for the archived counts.

The generator truncates deaths and risk-set sizes independently. On weeks where the next risk set is smaller than the current risk set less recorded deaths, the resampling reconstruction treats the difference as a nondeath exit at the end of the week. Across follow-up this accounts for 146 people in the control, 152 in the benefit cohort and 154 in the harm cohort. It preserves the source deaths and risk sets in the original estimates.

Run `py -3.11 -B "KCOR critique/objections/3.3/replay.py"` from the repository root. The [runner](replay.py) writes the [benefit workbook](fitted-benefit.xlsx), [harm workbook](fitted-harm.xlsx) and [full-precision endpoints, intervals and diagnostics](results.json). The historical Table 9 and figure remain the artifacts examined in the [Series 3 provenance check](<../../iterative review/evaluations/3/positive-check.md>).
