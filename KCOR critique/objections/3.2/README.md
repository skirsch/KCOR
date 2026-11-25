# 3.2 · A shock before the first quiet window changes the rate recovered by the seed fit

The first fit assumes zero cumulative offset. If a prior shock leaves a constant positive offset in individual cumulative baseline hazard, the first quiet window can fit perfectly at a lower effective baseline rate. The joint seed can still recover the generating frailty. A later pooled fit can move frailty because the rate is held fixed. The sensitivity of the Czech estimates to one alternative first-window calculation is measured below. Their underlying baseline and wave exposures are unknown.

The original model and previous discussions contain antecedents. Series 3 makes the rate equivalence explicit and tests the joint fitter. This issue is distinct from [3.1](../3.1/README.md), which concerns a subsequent gap.

[Derivation, replay and prior-art trace](<../../iterative review/evaluations/3/first-window.md>)

## Diagnostic experiment

The [experiment and results](experiment.md) compare the current fitter and a trial that uses survival to the first quiet week. This experiment is confined to this objection folder. It uses the current 3.1 edition as its comparator. Adoption as a KCOR procedure would require a separate decision.

## All-window follow-up

[Paired entry-aware all-window replay and early fitting-anchor diagnostic](all-window-entry/README.md). Proportional entry correction recovers true k, theta and KCOR; negligible Table 8 endpoint change. Nonlinear short-window problems are strongly linked to contaminated early fit anchors. Exclusion is diagnostic only; no method adoption.

## Exact nonlinear survivor-law fitting

[Complete matched survivor-fit study](nonlinear-survivor-fit/README.md). Expected-count k, theta and KCOR recover exactly for labeled nonlinear waves, with sampled main mean 1.00093 [0.99596,1.00591] versus 0.98982 [0.98493,0.99471] under the earlier gamma working fit. Coverage remains 90/100 and 48/50 in main and stress. Partial-wave and clean-anchor limitations noted. Experimental, approval pending.
