# Conditional joint-fit candidate: bounded pilot assessment

This is a trial of a **new fitting step**, not a validated change to KCOR. The candidate jointly estimates effective baseline mortality and gamma frailty from weekly deaths and risk sets in one contiguous quiet window (weeks 6 through 40). It conditions on survival to week 6, uses the exact discrete weekly gamma model, and keeps KCOR's hazard inversion and week-5 normalization. Code and outputs are isolated in this module.

## What was tested

The generating model has one million people, equal arm sizes on average, gamma frailties 4 and 0.5, and a true normalized ratio of 1 at week 52. The first two weeks are skipped; the fitting window is weeks 6–40. This is the previously identified large-population, within-model scenario where the current fitter's nominal 95% percentile intervals covered only 502/600 datasets (83.7%). Optional wave correction is off.

The deterministic expected-population gate passed from three single starting values and a multistart fit. Both frailties were recovered to numerical precision and the candidate normalized ratio was 1.000000000 within about 2×10⁻¹¹. The current fitter gave 1.00461455 on the same expected counts. This establishes removal of the *deterministic fitting bias in this scenario*; it does not establish better finite-sample estimates.

A fresh, one-worker pilot then ran 60 independent datasets with 199 pooled person-bootstrap draws per dataset. Both parameters were refitted in every draw. The saved 11,940 draws, all intervals, and selected seeds passed the audit. The unchanged inversion and normalization were also independently reconstructed from the saved counts and fitted frailties.

## Results

| Measure | Current fitter | Joint-fit candidate |
|---|---:|---:|
| Mean point error on the **same 60 datasets** | +0.0070 | −0.0049 |
| Standard deviation of point estimates on those 60 | 0.0100 | 0.0383 |
| Root mean squared point error on those 60 | 0.0122 | 0.0383 |
| Percentile interval coverage | 502/600 (83.7%) in the completed earlier study | 56/60 (93.3%) in this pilot |
| Mean percentile interval width | 0.0496 in the completed earlier study | 0.1448 in this pilot |

The current fitter's interval comparison comes from a different, larger study with 999 bootstrap draws, so the coverage and width rows are descriptive rather than a paired comparison. The **point-estimate rows are paired** on exactly the same 60 datasets. The candidate pilot coverage has a 95% Wilson range of 84.1%–97.4%; 60 datasets cannot establish 95% calibration or a statistically secure improvement over the earlier 83.7% estimate. Candidate basic intervals also covered 56/60 and had the same average width in this pilot.

## Numerical status and interpretation

The raw optimizer status marked 3/60 original analyses and 887/11,940 bootstrap draws as failures. We replayed all 897 flagged arm fits. Every one had a projected gradient below 0.001; 716 landed exactly at the legitimate zero-frailty boundary. For 180 flagged interior fits, a retry with looser stopping settings changed the frailty by at most 1.1×10⁻¹⁶ and did not improve the objective. The three flagged original arm fits likewise had projected gradients below 0.001. Thus the raw status counts reflect an overly strict optimizer-success flag, not evidence that those saved point estimates failed to reach their fitted optimum. We retained the original flags and draws and documented the audit rather than editing the pilot afterward.

Even after separating status flags from optimization, **weak identification remains**: at least one arm's bootstrap frailty was near zero in 4,492/11,940 draws (37.6%), despite the generating values 4 and 0.5. That boundary behavior and the roughly 3.8-fold larger point-estimate standard deviation explain why removing deterministic bias did not improve point-estimate accuracy in this pilot. The candidate's intervals were roughly 2.9 times as wide.

## Decision

This candidate is mathematically sound for the single-window generating model and is useful for diagnosing the current fit's bias. The pilot does **not** justify replacing KCOR's fitter: the point estimates were substantially less precise, and the apparent coverage gain came with much wider intervals. Objection 3 remains a demonstrated calibration concern for the current percentile interval in the tested scenario. This candidate exposes a bias–variance tradeoff rather than resolving the objection. A revised method would need a prespecified way to stabilize the joint fit, then a fresh, adequately powered calibration study. No result here supports a claim about every KCOR scenario or about empirical data.

## Reproduction and records

- `exact_population/verification.json`: deterministic gate and original comparison.
- `pilot/manifest.json`, `pilot/????.json`, `pilot/????.npz`: frozen pilot design, points, diagnostics, counts, and bootstrap draws.
- `pilot/assessment/summary.json`, `replicates.csv`, `audit.json`: interval calculations and seed replay.
- `pilot/assessment/paired_points.json`: both point estimators on the same 60 datasets.
- `pilot/assessment/fit_flags.json`, `original_flags.json`, and `pilot/failure_probe.json`: optimization status audits.

The source snapshot, production functions, prior validation module, and existing KCOR code were not modified.
The pilot manifest hashes the fitting and simulation sources as they stood when the run began. The summarizer and post-run audit scripts were added after the run began, so they are recorded by their output files but are not in that initial hash list. No fitting source changed during the run.
