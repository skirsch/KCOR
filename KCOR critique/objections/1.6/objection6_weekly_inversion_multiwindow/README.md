# Objection 6: effect of the weekly approximation in multi-window fitting

This isolated check compares the frozen production fitter with a variant that changes only its two internal weekly reconstruction lines to the exact gamma-frailty weekly increment. It makes no production-code changes. The [script](check.py) checks that each old line occurs exactly once before substituting it in memory and records the source-function hash in [the results](results.json).

Each version receives identical **expected** weekly death counts from the gamma-frailty model, with \(\theta=4\) and \(0.5\), 4 million total people, and \(\gamma=0.085/52\) per week. One case has a single quiet window as a no-gap control. Three cases use quiet weeks 6–25 and 41–55: no intervening wave, a 30% hazard increase in one arm during weeks 26–40, or the same increase in both arms. The measured outputs are fitted frailty, the diagnostic gate, and anchored KCOR at week 80. Expected counts remove sampling noise so any difference comes from this one code change.

| Case | First-order diagnostic | Exact-weekly diagnostic | First-order point | Exact-weekly point |
|---|---|---|---:|---:|
| One window | Both pass | Both pass | 1.00523 | 1.00523 |
| Two windows, no wave | Both arms rejected | Both pass | 1.12762* | 1.00547 |
| Two windows, one-arm wave | One arm rejected | Both pass | 1.21391* | 1.06211 |
| Two windows, wave in both arms | Both pass | Both pass | 1.00451 | 1.00473 |

\* The adapter computes a point for audit even on rejected fits; this is **not** a reportable KCOR estimate. The no-wave case has true anchored KCOR 1; the one-arm wave case has true anchored KCOR 1.05618; the both-arm wave case has true anchored KCOR 1. The check is about whether the implementation's approximation changes its output, rather than an evaluation of its overall bias.

The decisive mechanism is a hard diagnostic boundary. In the no-wave case, the first-pass reconstructed gap offset for the high-frailty arm is \(-3.26\times10^{-5}\) with the first-order update and \(+1.81\times10^{-4}\) with the exact update. The code treats any nonpositive first-pass offset as insufficient signal and sets the fitted frailty to zero. The low-frailty arm similarly changes from \(-4.88\times10^{-6}\) to \(+2.48\times10^{-5}\). In the both-arm wave case, both offsets are safely positive, and the effect on accepted KCOR is only about 0.000214. Thus a small arithmetic approximation can have a large discrete effect near the diagnostic boundary and a small effect away from it.

The tested fitter is the frozen corrected production source from the objection 3 validation module. The current `code/KCOR.py` still contains the same two approximate reconstruction lines; this script verifies their presence. The paper's stated joint seed fit differs from this production implementation, so these point results should **not** be assigned to the complete paper method. The check does not measure confidence-interval coverage or establish how often the boundary is crossed in real datasets. The appropriate next implementation decision is whether to replace the internal weekly update and then rerun relevant multi-window validation and diagnostics; no production source is changed here.

## Follow-up on simulated and saved Czech counts

[sensitivity.py](sensitivity.py) compares both updates on the same input in each run and checks that the frozen fitter has the same Python syntax tree as the current fitter. Run it with `simulation` or `replay` as the argument. Both runs use the fit gate from the earlier validation adapter. The Czech replay uses saved weekly death and risk counts for enrollment weeks 2021_24, 2022_06, and 2022_47, grouped by birth decade and configured cohort pairs. The KCOR comparison uses exact final inversion with the configured anchor and reporting date. Its scope is the core fit and contrast before the optional wave adjustment and workbook reporting.

| Data | Rejected to accepted | Accepted to rejected | Passed both | Largest absolute KCOR change among paired points |
|---|---:|---:|---:|---:|
| Simulation, 720 datasets | 28 | 0 | 418 | 0.001639 |
| Czech replay, 108 cohort fits | 0 | 0 | 39 | — |
| Czech replay, 126 cohort comparisons | 0 | 0 | 33 | 0.000562 |

The simulations used two quiet windows, two total population sizes (400,000 and 4 million), three baseline mortality and frailty settings, and 40 datasets per setting and gap level. The decisions changed in 18 of 240 datasets with no excess hazard in the gap, 10 of 240 with a 3% increase, and 0 of 240 with a 30% increase. The percentages describe this fixed simulation grid. The Czech replay describes the selected workbook and fit path.

The exact update eliminates the approximation as a cause of crossing the zero threshold. It retains the threshold rule and the fixed-rate seed fit. The median absolute KCOR change among paired points was 0.000198 in simulation and 0.000172 in the Czech replay. The [simulation record](simulation_sensitivity.json), [replay record](replay_sensitivity.json), and [isolated software patch](<../../../software-fixes/weekly_reconstruction/README.md>) support the proposed correction.
