# Objection 3: paired Monte Carlo resampling with the paper's joint fit

## Main result

The **paper's joint baseline-rate/frailty estimator was held fixed**. For each of 200 independently simulated, model-consistent datasets, two 95% percentile intervals were computed with 199 draws each:

| Monte Carlo resampling rule | Coverage of true value 1 | Mean interval width | Mean bootstrap median minus dataset point | Dataset point outside its own interval |
|---|---:|---:|---:|---:|
| **People with replacement**, as in the current real-data software's Monte Carlo path | **189/200 (94.5%)** | 0.14589 | +0.00321 | 0/200 |
| **Four-week aggregate blocks**, one explicit implementation of the manuscript's outline | **38/200 (19.0%)** | 0.15163 | +0.10188 | 173/200 |

The datasets, original counts, paper point estimates, target, and number of bootstrap draws were identical between rows. The true normalized value was 1; the shared point estimates averaged **0.99769** with standard deviation **0.03610**. The approximate 95% Wilson range for person-resampling coverage is **90.4–96.9%**, which includes 95%. The corresponding block range is **14.2–25.0%**. The first 99 draws from each saved 199-draw series gave 189/200 person coverage and 36/200 block coverage, so the large difference is not an artifact of using the last 100 draws.

Paired outcomes were: both intervals covered on 33 datasets, person only on 156, block only on 5, and neither on 6. Of the 11 person misses, 4 intervals were entirely above truth and 7 below. All 162 block misses had their lower endpoint above truth. The intervals have similar average widths; the large difference is predominantly **where the intervals are centered**.

## Exactly what was compared

The scenario had one million people in total, two cohorts with generating frailties 4 and 0.5, quiet weeks 6–40, two skipped weeks, week-5 normalization, and a week-52 target under the gamma-frailty/Gompertz working model. There was no added effect and no optional wave correction. The paper's joint least-squares first fit was rerun in every bootstrap replicate, followed by its cumulative gamma inversion and normalized ratio. In this one-window setting, the paper's pooled Step 4 has the same quiet-window objective and no wave offsets.

The **person** version reused the corrected validation module's pooled person bootstrap: sample original people with replacement, preserving each sampled person's cohort and death week, then rebuild weekly counts and risk sets. It stores the result as a multinomial draw over cohort-by-lifetime categories; [the equivalence](WHY_MULTINOMIAL_EQUALS_PERSON_RESAMPLING.md) is exact for this single-stratum design. This is the resampling distribution used by the real-data software path, **combined here with the paper's joint estimator**. The current production fitter instead fixes the early baseline rate; its previously measured 502/600 (83.7%) coverage is a separate result and cannot be substituted for this paired study.

The **block** version copied source death fractions in four-week runs within broad pre-quiet, quiet, and post-quiet time segments and reconstructed each target risk set sequentially. This was a frozen, fully specified [operationalization](../block_bootstrap_operationalization/ASSESSMENT.md) of the manuscript's aggregate-block outline. Its block length, source-time pools, placement, and risk-set seam are choices supplied by this study because the paper does not define them. The manuscript also lacks numerical full-diagnostic acceptance thresholds. Therefore **19% is not a coverage estimate for the uniquely intended or historically run paper interval**.

## Verification and interpretation

All 200 person analyses began from byte-matched counts and identical paper point estimates to their block counterparts. All **39,800 person-bootstrap estimates** were finite, with no failed fits; the block study likewise had 39,800 finite estimates. Every interval was reconstructed from its saved draws. Selected datasets and first draws were regenerated from their seeds, and an [independent multi-start audit](main/optimization_audit.json) found a maximum selected-point difference below 2×10⁻⁸. Among the 129 datasets where both original frailty fits exceeded the exploratory 0.01 boundary proxy, person coverage was **124/129 (96.1%)** and block coverage was **21/129 (16.3%)**. This proxy is not the paper's full diagnostic gate.

The result shows that **resampling choice matters strongly even when the paper's point estimator is held constant**. Person-resampling intervals are compatible with nominal 95% coverage in this one scenario. The tested broad-pool block intervals fail under the same model, and their upward shift also appears on [exact expected counts](../block_bootstrap_operationalization/WHY_BLOCK_PLACEMENT_MATTERS.md). A different, better specified aggregate-block algorithm could behave differently; source-time placement alone changed the exact-count bootstrap center in a post-pilot probe.

This does not establish general interval calibration for KCOR: only one model-consistent, one-window, one-stratum scenario was tested here with 199 draws per dataset. It does not validate the full current fixed-rate production pipeline, optional wave correction, multi-window offset fitting, or the paper's unspecified diagnostic-pass conditioning. For objection 3, the next decision is editorial and algorithmic: if the authors intend the **person** Monte Carlo actually implemented, the paper should describe it and the paper's joint fit should be aligned with code before broader coverage validation. If they intend **aggregate blocks**, they must supply the generating algorithm and quantitative diagnostic gate, then validate that exact implementation across prespecified scenarios.

## Evidence

- [Frozen paired design](design.json), [person run manifest](main/manifest.json), [machine summary](main/summary.json), [draw audit](main/audit.json), [all 200 paired intervals](main/paired_intervals.json), and [retained per-dataset draws](main/README.md).
- [Block run assessment](../block_bootstrap_operationalization/ASSESSMENT.md), [machine summary](../block_bootstrap_operationalization/main/summary.json), and its retained per-dataset draws.
- [Person resampling source](../../objection3_corrected_validation/simulation.py), [aggregation verifier](../../objection3_resolution/verify_person_aggregation.py), [paper fit](../paper_step1.py), and the [source-provenance audit](../../objection3_step2_interval_provenance_2026-11-10/assessment.txt).

All new code and outputs are isolated in the critique tree. No production code, manuscript source, or earlier result record was edited.
