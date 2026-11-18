# Paired Monte Carlo resampling comparison

This module uses the **same 200 simulated datasets and the same paper joint point estimator** as the [four-week aggregate-block study](../block_bootstrap_operationalization/ASSESSMENT.md). For every dataset it builds a second 95% percentile interval using the software's actual **person-resampling** Monte Carlo rule. The original data, paper point estimate, true value, target week, and number of draws are paired; only the resampling rule changes.

The person resampling is implemented by the already retained `simulation.bootstrap` function in the corrected validation module. It draws a multinomial count over each person's arm and death-week category, mathematically equivalent to sampling individual records with replacement. The [frozen design](design.json) and per-dataset seeds make this comparison repeatable. This is a comparison involving the **paper's estimator**, not a claim that the current production fixed-rate fitter is the paper's estimator. Existing production-code interval results remain a separate third reference in [Objection 3](../../README.md).

Start with [ASSESSMENT.md](ASSESSMENT.md) once the paired run is complete. The paper's exact aggregate-block implementation remains unspecified, so the block side retains the limitations documented in its own assessment.
