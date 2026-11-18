# Corrected versus original calibration

Fresh random seeds; identical scenarios and statistical rules. These are scenario-level comparisons, not paired datasets.

| Scenario | Interval | Old coverage | Corrected coverage | Accepted | Conditional coverage | Conditional assessment |
|---|---|---:|---:|---:|---:|---|
| equal_frailty | percentile | 93.8% | 97.7% | 600/600 | 97.7% | inconclusive_or_conservative |
| equal_frailty | basic | 84.5% | 98.0% | 600/600 | 98.0% | inconclusive_or_conservative |
| unequal_frailty | percentile | 83.5% | 96.2% | 600/600 | 96.2% | adequate_in_tested_scenario |
| unequal_frailty | basic | 81.5% | 98.7% | 600/600 | 98.7% | inconclusive_or_conservative |
| unequal_large | percentile | 13.3% | 83.7% | 600/600 | 83.7% | undercoverage |
| unequal_large | basic | 21.0% | 97.7% | 600/600 | 97.7% | inconclusive_or_conservative |
| zero_frailty | percentile | 95.3% | 98.0% | 600/600 | 98.0% | inconclusive_or_conservative |
| zero_frailty | basic | 85.5% | 96.2% | 600/600 | 96.2% | adequate_in_tested_scenario |
| post_quiet_increase | percentile | 83.5% | 94.7% | 600/600 | 94.7% | adequate_in_tested_scenario |
| post_quiet_increase | basic | 78.8% | 97.7% | 600/600 | 97.7% | inconclusive_or_conservative |
| post_quiet_decrease | percentile | 82.5% | 94.5% | 600/600 | 94.5% | adequate_in_tested_scenario |
| post_quiet_decrease | basic | 80.7% | 98.0% | 600/600 | 98.0% | inconclusive_or_conservative |
| sparse | percentile | 95.2% | 97.3% | 0/600 | undefined | insufficient_accepted_analyses |
| sparse | basic | 87.3% | 90.0% | 0/600 | undefined | insufficient_accepted_analyses |
| quiet_contamination | percentile | 84.0% | 93.5% | 600/600 | 93.5% | inconclusive_or_conservative |
| quiet_contamination | basic | 80.2% | 90.2% | 600/600 | 90.2% | inconclusive_or_conservative |

Quiet-window contamination is a separately labelled assumption-violation stress. Boundary and low-information cases are identified in the JSON scope fields.

Coverage against the fitted population output is a secondary diagnostic only; it does not replace the model-defined target.

This run does not resolve the separate manuscript-versus-code resampling description or identify historical published-output provenance.
