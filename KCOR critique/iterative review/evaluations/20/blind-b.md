# Blind independent review of m184

## Verdict

I found one reproducible validity-affecting implementation problem in the batched Monte Carlo route and no demonstrated algebraic defect in the core proportional estimator. The core is an intentionally conditional, descriptive working-model contrast. The fixed-cohort and post-skip mean-one frailty definition, gamma inversion, known-slope logistic hazard identity, exact integrated weekly prediction, and cohort-specific refit are coherent. The current manuscript is appropriately cautious about mean frailty, causal interpretation, quiet-window contamination, baseline misspecification, weak identification, signed-gap diagnostics, non-gamma survivors, external effect scale, and the lack of a calibrated general bootstrap. Those are restrictions on interpretation, not reasons to replace the method.

## Substantive reproducible defect: batched calendar inconsistency

`code/KCOR.py:6634–6648, 6743–6779` groups Monte Carlo rows by `ISOweekDied` and determines quiet-window membership from that column. But `code/fixed_cohort_history.py:141–188` checks history/calendar completeness using `DateDied`, and KCOR uses `DateDied` to determine elapsed event week and the presentation anchor (`KCOR.py:6699–6703`). Ordinary estimation instead derives ISO weeks from `DateDied`. There is no check that the two calendar representations agree. Thus contradictory calendar metadata can place actual epidemic weeks in the fitting set and omit actual quiet weeks, without violating the fixed-cohort history or triggering the numerical fit safeguards. It can also cause premature aggregation in a batched workbook with duplicated ISO labels.

I reproduced this with the production `process_workbook(..., return_rows=True)` route, without a bootstrap. Construct a 40-week cohort beginning Monday 2023-01-02; birth-year 1940, doses 0 and 1, each enrolling 100,000 people, sex F, `mc_id=0`. Weekly deaths are rounded from 0.002 of opening risk for both cohorts, except actual weeks 8–14 of dose 1 have 0.008 risk. Each subsequent risk set is exactly the preceding risk set minus deaths. Configure quiet weeks 2023-14 through 2023-35, ordinary identified-only module mode; batched MC enrollment 2023_01. Compare truthful calendar labels against labels calculated from `DateDied + 8 weeks`, changing no actual dates or counts.

| Input ISO labels | Last 1940 anchored KCOR | Fitted theta, dose 1 | Fitted theta, dose 0 | Normalization status | Both fit statuses |
|---|---:|---:|---:|---|---|
| Correct | 1.553030 | 0.482487 | 0.816595 | ok | OK |
| Shifted +8 weeks | 2.116018 | 5.747843 | 0.814656 | ok | OK |

The 36% endpoint shift is purely calendar metadata, not a scientific change. Exploratory residual diagnostics do not solve this contradiction in the input. I have not established that any published input contains discordant calendar metadata; this is a conditional defect in the public batched route, not evidence that existing numerical results are wrong.

**Fix:** derive the canonical ISO year/week from validated `DateDied` everywhere, including batched grouping and fit masks, and validate any separately supplied ISO labels against that canonical representation before aggregation (accept documented equivalent separators). Prefer rejecting a conflicting replicate/group with an explicit unavailable reason to silently trusting malformed metadata. Add a narrow test with contradictory but unique ISO labels and another with duplicated labels; valid calendar inputs must remain numerically unchanged. No change to the mathematical estimator is needed.

## Other observations and non-defects

- Negative-control and real registry illustrations do not prove exchangeability or identify causal vaccine effects. The manuscript now explicitly describes outcome-dependent window provenance, residual flags, and the unknown exact target for the empirical age-shift controls. I would retain that conservative interpretation rather than treat these as fresh undisclosed flaws.
- The optional nonlinear law is a separate specified-exponent sensitivity; the distinction between instantaneous wave-component cancellation and approximate weekly localization, and the requirement of an external effect scale for the historical localization procedure, are clearly articulated. No evidence here warrants requiring a different wave model or estimating additional free effects.
- The percentile bootstrap is sampling-variability propagation under the fitted model; its lack of universal coverage is already qualified. The current fitter's coverage is not conclusively established by historical studies, as the paper explicitly acknowledges. A direct prospective coverage study would strengthen validation but is not a prerequisite to the stated diagnostic estimand or a reason to redesign the estimator.
- Minor wording: §2.3 says the log survival transform is standard under a piecewise-constant hazard assumption; the negative log of the conditional weekly survival probability gives the integrated hazard without needing a constant within-week hazard. Clarify if desired. §2.6 says normalization factors out the contribution of both fitted k and theta; the proportional inversion itself uses theta and retains baseline level k (the manuscript otherwise correctly distinguishes level from curvature). These are wording corrections, not numerical defects.

Review limited to the frozen m184 candidate and a small external scratch workbook; no full bootstrap, prior-version comparison, or external critiques consulted.
