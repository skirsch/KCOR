# Paper Step 1 on sampled cohorts

The manuscript's explicit joint least-squares Step 1 was applied to **all 600 saved one-million-person datasets** from the corrected objection 3 validation. The two procedures therefore used the same sampled death histories. Frailty-generating values were 4 and 0.5, the model-defined normalized contrast was 1, and one quiet window covered weeks 6–40. The paper predictor and exact cumulative inversion were used; the scalar contrast uses the week-5 reference in the saved validation design. The fit was repeated from four starting values in each arm. This evaluates point estimates, not the manuscript's block-bootstrap interval or full diagnostic gate.

| Point estimate | Mean | Standard deviation across 600 datasets | Root mean squared error from 1 |
|---|---:|---:|---:|
| Paper's joint Step 1 | 1.00567 | 0.03808 | 0.03846 |
| Tested fixed-rate code on those same datasets | 1.00986 | 0.01058 | 0.01445 |

All 600 joint fits returned numerical success. In the smaller-frailty arm, 152/600 fitted frailty values were below 0.01; no such values occurred in the larger-frailty arm. The tested code's operational diagnostic acceptance cannot be transferred to the paper fit, whose full diagnostic gate was not reproduced here.

On **exact expected counts** the paper Step 1 gives 0.9999999663, essentially the true value 1; the fixed-rate code gives 1.00461455. Thus the paper objective removes the tested code's deterministic offset in this one-window model. The sampled point estimates nevertheless vary substantially more, and their mean remains about 0.00567 above 1. The current code has lower root mean squared point error on these 600 datasets. This does not determine which method has better calibrated intervals: the paper's aggregate-block bootstrap remains operationally unspecified.

The [machine-readable summary](saved_600/summary.json), [audit](saved_600/audit.json), [frozen manifest](saved_600/manifest.json), and 600 per-dataset JSON records retain every paired point, fitted parameter, and source count hash. Three selected datasets were replayed from their saved count tables, and all 600 original point values and count hashes were matched to the earlier run. One worker with reduced priority was used.
