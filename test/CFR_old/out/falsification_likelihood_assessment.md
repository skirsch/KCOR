# VE-Death Likelihood Assessment

Date: 2026-04-09

This is a qualitative synthesis of the current CFR / falsification packet. It is not a formal Bayesian posterior or a causal estimate. It is a compact summary of what the current outputs appear to support after:

- age-stratified cohort comparisons
- old-vs-young breakpoint tests
- placebo breakpoint scans
- non-COVID negative controls
- multi-split age-cutoff sensitivity
- infection-naive-at-enrollment restriction
- incidence-vs-severity decomposition

## Current likelihood split


| VE-death range    | Rough likelihood | Why                                                                                                                                                                                                        |
| ----------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Slightly negative | 10%              | Not ruled out, but now looks less compatible with the age-stratified wave-period COVID death-rate results, the repeated COVID-specific old-vs-young break, and the infection-naive sensitivity.            |
| 0% to 10%         | 25%              | Still plausible because the design remains ecological and non-COVID background shifts remain present.                                                                                                      |
| 10% to 20%        | 30%              | Plausible and well supported by the combined packet: positive COVID death-rate signals survive age stratification, prior-infection restriction, and multiple age cutoffs.                                  |
| 20% to 40%        | 25%              | More plausible than before after adding the incidence/severity decomposition and naive-only results, but still not firmly established because residual age-specific epidemic confounding remains possible. |
| >40%              | 10%              | Possible, but the population-level ecological evidence is not clean enough to make a large death benefit the center of mass.                                                                               |


## Short interpretation

The current evidence no longer centers as strongly on "near-zero only." After the added falsification steps, the weight appears shifted toward a modest-to-moderate positive death benefit, while still leaving some room for near-zero and a smaller amount of room for slightly negative effect.

If a single headline sentence is needed:

> The current packet looks most compatible with a modest positive COVID death benefit, more likely in the 10%–40% range than below zero, but not identified cleanly enough to rule out near-zero impact.

## Main evidence behind the shift

- The COVID old-vs-young breakpoint survives all tested age cutoffs and remains placebo-rank 1.
- Restricting to infection-naive-at-enrollment does not weaken the COVID signal; if anything, it slightly strengthens it.
- The non-COVID negative control also moves, but the COVID jump is consistently larger.
- The incidence-vs-severity decomposition suggests the mortality signal is not only incidence-driven; severity contributes too, especially for 70+ vs <70.

## Main remaining doubt

The analysis is still ecological. The most important surviving alternative explanations are age-specific changes in exposure, variant impact, treatment context, or other wave-period age-varying factors that coincide with rollout.

## Quantitative Bound

The current transparent scenario-bound files are:

- `falsification_quantitative_scenario_bounds.csv`
- `falsification_naive_quantitative_scenario_bounds.csv`

For the full sample, the scenario ranges are approximately:

- `70plus_vs_under70`: `0.00` to `0.38`
- `60plus_vs_under60`: `0.00` to `0.32`
- `50plus_vs_under50`: `0.00` to `0.36`

For the infection-naive-at-enrollment sample, the scenario ranges are approximately:

- `70plus_vs_under70`: `0.00` to `0.50`
- `60plus_vs_under60`: `0.00` to `0.42`
- `50plus_vs_under50`: `0.00` to `0.45`

These are not identified causal intervals. They are transparent scenario ranges built from:

- a conservative lower bound that keeps `0` in play
- a negative-control-adjusted signal
- a coverage-calibrated implied population effect
- an optimistic cohort-based ceiling

## Why Stop Here

There is probably limited value in forcing a single final numeric VE estimate from this dataset alone.

Reason:

- The data source is observational and heavily confounded.
- Cohorts differ strongly.
- Wave conditions and natural immunity can vary in age-specific ways that are not fully identified here.
- A tighter single-number estimate would mostly reflect extra assumptions, not new information extracted from the data.

So the honest stopping point is:

- descriptive results
- falsification and sensitivity checks
- transparent scenario bounds

and not a pretend-precise single VE estimate.

## Practical Conclusion

The current packet does **not** show a convincing vaccine-harm signal in Czechia.

It does **not** prove a clearly positive death benefit either.

The most defensible summary is:

> With this single source and this level of confounding, the data are compatible with `0`, but lean more toward modest positive than negative effect; stronger claims would require stronger assumptions than the data themselves can justify.
