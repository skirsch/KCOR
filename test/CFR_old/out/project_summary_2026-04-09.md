# CFR Analysis Summary

Date: 2026-04-09

## Scope

This summary captures the work done in the `test/CFR` analysis pipeline around Czech COVID mortality, healthy-vaccinee effects (HVE), ecological old-vs-young comparisons, falsification tests, and practical interpretation limits.

## Main Question

The core question was whether the available Czech data support a meaningful COVID death benefit from vaccination, after accounting for strong cohort differences and likely healthy-vaccinee selection.

## Main Changes To The Pipeline

The following additions were made to the `test/CFR` workflow:

- Added a falsification framework for broad old-vs-young ecological comparisons.
- Added breakpoint tests and placebo scans around the configured Delta-wave onset.
- Added coverage-dilution calculations to estimate the size of any expected population effect under observed uptake.
- Added multi-split age-cutoff sensitivity checks:
  - `70plus_vs_under70`
  - `60plus_vs_under60`
  - `50plus_vs_under50`
- Added infection-naive-at-enrollment sensitivity analysis using a new prior-infection flag.
- Added non-COVID negative-control summaries.
- Added incidence-versus-severity decomposition using broad-group case rates, COVID death rates, and episode-style CFR.
- Added heuristic VE-death summaries and conservative scenario-bound outputs.
- Added test coverage for the new falsification and prior-infection logic.
- Added `pytest` to [requirements.txt](C:/Users/stk/Documents/GitHub/KCOR/requirements.txt).

## Key Output Files

Important outputs generated in [test/CFR/out](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out):

- [falsification_diff_break_tests.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_diff_break_tests.csv)
- [falsification_placebo_scan.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_placebo_scan.csv)
- [falsification_coverage_dilution.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_coverage_dilution.csv)
- [falsification_multi_split_summary.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_multi_split_summary.csv)
- [falsification_negative_control_summary.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_negative_control_summary.csv)
- [falsification_incidence_severity_summary.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_incidence_severity_summary.csv)
- [falsification_ve_death_bounds.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_ve_death_bounds.csv)
- [falsification_quantitative_scenario_bounds.csv](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_quantitative_scenario_bounds.csv)
- [falsification_likelihood_assessment.md](C:/Users/stk/Documents/GitHub/KCOR/test/CFR/out/falsification_likelihood_assessment.md)

Parallel infection-naive versions were also generated for the major falsification outputs.

## What The Data Seem To Say

### 1. Raw vaccinated-versus-unvaccinated comparisons are badly confounded

Baseline all-cause mortality differed sharply across vaccine cohorts before the wave, which is strong evidence of healthy-vaccinee / selection effects. This means simple vaccinated-versus-unvaccinated mortality comparisons are not trustworthy on their own.

### 2. The ecological old-versus-young design is more informative

The old-minus-young COVID mortality gap shows a structural change around the configured Delta-wave onset. This weakens the claim that absolutely nothing changed at population level.

### 3. The signal is not explained only by one arbitrary age cutoff

The main breakpoint signal survived all three age-cutoff definitions:

- `70plus_vs_under70`
- `60plus_vs_under60`
- `50plus_vs_under50`

### 4. Prior infection does not appear to be the main driver of the signal

Restricting to infection-naive-at-enrollment individuals did not remove the COVID old-versus-young breakpoint. If anything, the signal became slightly stronger.

### 5. The COVID signal is larger than the non-COVID negative control, but not perfectly clean

Non-COVID old-versus-young mortality also moves, which means background age-structured effects remain. However, the COVID break is consistently larger than the non-COVID break across the tested splits.

### 6. The ecological pattern appears to involve both incidence and severity

The added decomposition suggests:

- strong case-rate movement across all splits
- a severity component as well, especially for `70plus_vs_under70`

This means the mortality pattern is not just one-dimensional.

## Conservative Interpretation

The current evidence does **not** show a convincing vaccine-harm signal in Czechia.

It also does **not** prove a strong positive COVID death benefit.

The most defensible conservative summary is:

- `0` remains compatible with the data
- negative net COVID-mortality effect is not the best-supported reading
- modest positive effect is more plausible than negative effect
- large positive effect is difficult to defend strongly from this data source alone

## Likelihood Framing Used In Discussion

One rough working likelihood split used in discussion was:

- Slightly negative: `10%`
- `0%` to `10%`: `25%`
- `10%` to `20%`: `30%`
- `20%` to `40%`: `25%`
- `>40%`: `10%`

These numbers are heuristic judgments, not identified estimates.

## Scenario Bounds

The conservative scenario-bound outputs were intentionally built to keep `0` inside the range, because the full-population wave behavior does not show a clean enough shift to rule out no net effect.

Illustrative full-sample bounds:

- `70plus_vs_under70`: `0.00` to about `0.38`
- `60plus_vs_under60`: `0.00` to about `0.32`
- `50plus_vs_under50`: `0.00` to about `0.36`

Illustrative infection-naive bounds:

- `70plus_vs_under70`: `0.00` to about `0.50`
- `60plus_vs_under60`: `0.00` to about `0.42`
- `50plus_vs_under50`: `0.00` to about `0.45`

These are scenario ceilings and conservative ranges, not hard causal bounds.

## Why The Analysis Stops Here

There is only so much information that can be extracted from one observational data source with heavy confounding, changing waves, and strong cohort differences.

Further attempts to produce a single final VE number would mostly reflect assumptions about unmeasured bias rather than information identified by the Czech dataset itself.

What this dataset can do well:

- reject some simplistic stories
- show where strong claims are unsupported
- narrow the plausible range
- distinguish "clearly harmful" from "not clearly harmful"

What it cannot do cleanly by itself:

- prove a tightly identified causal VE against COVID death
- force the lower bound above zero
- fully eliminate time-varying age-specific confounding

## Related Reading From The PDF Review

The PDF [Spiked 10th March 2026.pdf](C:/Users/stk/Downloads/Spiked%2010th%20March%202026.pdf) was skimmed, especially pages `300` to `400`. The broad impression was:

- it adds arguments and selected examples that increase skepticism toward large claimed mortality benefit
- it does not provide decisive evidence of a clear harmful vaccine effect
- it mainly pushes interpretation toward "smaller and more uncertain" rather than toward "negative"

## Practical Bottom Line

The current Czech analysis is most consistent with:

- substantial uncertainty
- a likely effect somewhere between zero and modestly positive
- little support for a strong harm interpretation
- insufficient evidence from this source alone to claim a large mortality benefit with confidence
