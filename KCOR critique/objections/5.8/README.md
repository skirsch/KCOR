# 5.8 - Round 5 finding

**Severity:** Moderate

Proportional gap diagnostic reports pre-optimization offsets rather than final offsets; shape diagnostics use straight elapsed time and ordinary inverse rather than the active Gompertz/offset model. Status vs failure policy needs clear distinction.

**Related issues:** 3.1, 4.4, diagnostic alignment

**Review evidence:** A B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** first two diagnostic inconsistencies corrected in m110; m109 preserved. Zero-death safeguard remains a separate decision.

1. Final raw and applied proportional offsets and sign flags are now exported from the same recurrence used at the final fitted parameters, not the initialization. Nonlinear exports use the final survivor-profile gap discrepancies (observed cumulative-hazard units) and applied baseline-exposure increments, explicitly labeling the raw units.
2. Shape diagnostic uses the actual reported normalized cumulative-hazard trajectory. For each contiguous eligible quiet window, subtract its opening cumulative exposure and compare the increase to the fitted integrated Gompertz curve. Report residual mean, SD, maximum absolute deviation, and signed residual drift (largest absolute per-window slope); per-window metrics are exported in gamma_frailty_fit. No new fit, threshold, or acceptance rule.
3. The current zero-death relative-RMSE guard is unchanged. A small known-truth example demonstrates the problem: N=1,500, 60 weeks, 89 deaths, 15 zero-death weeks (the true probability of zero in those weeks is 22.4--25.1%). The current relative RMSE is 5.07e8 and flags insufficient_signal, resetting fitted theta from 6.87 to zero. Pearson residuals in those zero weeks are only -1.33 to -1.11, and sum of squared Pearson residuals divided by nominal 58 degrees of freedom is 1.03. This is illustrative, not a calibrated test. Conversely, in the earlier 310-death counterexample, the fitted expected first-week death count was 9.13 and zero probability about 0.000108, so zero observations need not always be accepted without scrutiny. A count-based standardized residual or binomial deviance diagnostic can reflect expected sampling variation, but its calibration and decision policy remain for discussion. [Exploration](zero-death-exploration.json), [script](explore_zero_deaths.py).

Verification: [old](m109-diagnostic-check.json) and [new](m110-diagnostic-check.json) reproduce identical fitted k/theta while the reported gap changes 0.01277145 -> correct 0.00834640. Exact Gompertz within-window residual is <2e-17 even with a persistent gap offset; a deliberately perturbed second window alone is detected. Nonlinear raw/applied outputs match the actual profile gaps. [Workbook verification](workbook-verification.json): all point and interval rows unchanged in both modes (five-draw smoke), and actual shape window metrics exported for both age-specific and all-ages cohorts. No full 999-draw replay required for unchanged fitting equations. Status/fallback policy and zero-death safeguard unchanged.

Main text section 2.6.2 and supplement S5.1 updated. [PDF](KCOR-clean.pdf), [main patch](paper.patch), [supplement patch](supplement.patch), [code patch](KCOR.py.patch), [survivor patch](kcor_survivor.py.patch), [regression](verify.py). Existing draft PR unchanged.

Related 5.2 regression: even after removing the zero-early initialization veto, the existing relative-RMSE diagnostic divides a positive predicted hazard by 1e-12 at a zero observed hazard. Both module-off and forced-alpha examples with 310 quiet deaths flag insufficient_signal. See [5.2 evidence](../5.2/README.md). Diagnostic policy unchanged pending discussion.


## Sampling diagnostic implementation (m111)

Replaced the relative-RMSE reset with reporting-only binomial count residuals and doubled smaller inclusive tail probabilities, using actual opening risk sets in the ordinary age-specific and all-ages fitter. Exploratory threshold 0.05/m; fitted parameters make this a screening rule, not calibrated inference. Legacy relative-error flag retained for comparison. Other existing bound and low-death safeguards retained. The sparse example retains theta 6.8695, no sampling flag, though the legacy flag is true. Zero when one death is expected is not flagged; zero when 20 are expected is flagged. Forced-alpha workbook exports 148 weekly diagnostic rows. [Verification](sampling-verification.json), [test](verify_sampling.py), [paper patch](sampling-paper.patch). No full 999-draw replay; results can change where the old reset fired.
