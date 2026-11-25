# Objection 3: paper-first aggregate-block interval assessment

## What the paper specifies

The manuscript specifies a **joint** first fit of baseline rate and frailty in the nearest quiet window, gamma-frailty inversion, an aggregate bootstrap of contiguous event-time blocks with reconstructed risk sets, and 2.5th/97.5th percentile endpoints. In this tested one-window, no-wave setting, its pooled Step 4 uses the same quiet-window objective with the fitted baseline rate fixed; there are no wave offsets or additional windows. The [earlier exact-count check](../exact_population.json) gave 0.9999999663 when the true normalized value was 1. On [600 sampled datasets](../POINT_RESULTS.md), its point estimates averaged 1.00567 with standard deviation 0.03808.

The manuscript and supplement do **not** uniquely specify the block length, source-time pool and placement, how a source risk set joins the reconstructed target risk set, or numeric thresholds for the full diagnostic acceptance rule. The repository contains no executable implementation of this stated aggregate-block bootstrap. These missing choices prevent an unqualified coverage number for **the paper's uniquely intended interval**.

## Explicit implementation tested here

The [frozen design](design.json) defines one implementation of the paper's outline. It samples four-week blocks with replacement within each arm and within the pre-quiet (weeks 0–5), quiet (6–40), and post-quiet (41–99) calendar segments. Each source week's death fraction is applied to the reconstructed target risk set, rounded to an integer death count, and the risk set is decremented. The recurrence for numbers at risk therefore holds exactly. Each bootstrap sample reruns the paper's joint fit and normalization; individuals are never resampled. The same simulator generated independent one-million-person datasets under the gamma-frailty/Gompertz working model, with cohort frailties 4 and 0.5, no added effect, true normalized week-52 value 1, and week-5 normalization. The optional wave module was inactive.

The primary run used **200 independent datasets and 199 bootstrap draws per dataset** on one reduced-priority worker. A prespecified eight-week version used 60 of those same datasets and the same 199-draw count. The 20-dataset, 99-draw pilot was for execution and early diagnosis; it was not combined with the primary coverage estimate.

## Results

| Result | Four-week primary | Eight-week sensitivity |
|---|---:|---:|
| Datasets | 200 | 60 |
| Percentile intervals containing truth | **38/200 (19.0%)** | **10/60 (16.7%)** |
| Approximate 95% Wilson range for coverage | 14.2–25.0% | 9.3–28.0% |
| Misses with lower endpoint above truth | 162 | 50 |
| Dataset point estimate outside its own interval | 173 | 57 |
| Mean point estimate | 0.99769 | 0.99667 |
| Mean bootstrap median minus dataset point | +0.10188 | +0.09483 |
| Mean interval width | 0.15163 | 0.12867 |
| Failed or nonfinite bootstrap draws | 0/39,800 | 0/11,940 |

On the **same first 60 datasets**, the four-week version covered 7/60, versus 10/60 for eight-week blocks. Thus block length changes individual intervals, but both tested lengths fail badly under this broad-segment placement rule. The sampled point estimates themselves remain approximately centered near the true value. The failure is primarily that block resampling shifts the percentile distribution upward, so many intervals sit above both the point estimate and the truth.

Numerical fit failure does not explain the result: all 200 original joint fits succeeded, all bootstrap draws were finite, no warm-start checks disagreed materially with the paper objective's four-start fit, and the [independent optimization audit](main/optimization_audit.json) found maximum selected point differences below 4×10⁻⁸. Excluding a simple near-zero-frailty proxy would not restore coverage: among the 129 primary datasets with both original fitted frailties at least 0.01, **21/129 (16.3%)** intervals covered. This proxy is **not** the manuscript's full diagnostic gate. Mean quiet-window fit root mean squared error was similar in covered and missed datasets (about 4.4×10⁻⁵ and 4.3×10⁻⁵, respectively), but the manuscript does not define a numerical pass threshold for it.

## Why this does not become an unqualified paper coverage result

On exact expected counts, with no random dataset noise, the joint point estimate is 0.99999997 while this four-week block bootstrap is centered near **1.1008**; eight-week blocks center near **1.0959**. This demonstrates a shift from the block rearrangement itself. A post-pilot check changed only the source-time pools: keeping blocks within roughly 12-week calendar segments moved the exact-count bootstrap median to **1.0113**. The [mechanism note](WHY_BLOCK_PLACEMENT_MATTERS.md) explains why preserving survivor accounting does not preserve the time-varying Gompertz and frailty geometry when blocks are moved across time.

That placement check is exploratory. It does not validate the narrower pools or identify the authors' intended algorithm. The paper could use a different time rule, seam rule, block length, or diagnostic rejection rule with different coverage. The main **paper-method finding** is therefore: the manuscript's joint point fit removes the deterministic offset seen in the tested fixed-rate code, but its described aggregate-block interval cannot be assigned a unique coverage rate from the available specification. One explicit, reproducible implementation of its outline has severe undercoverage under its own working model. On the **same 200 datasets**, the paper joint fit combined with the software's person-resampling law covered **189/200 (94.5%)**; [paired assessment](../paired_resampling/ASSESSMENT.md). This isolates a large resampling difference. It does not prove that every possible implementation of the manuscript's block outline fails or validate the complete fixed-rate production pipeline.

To resolve the remaining objection, the authors would need to identify the generating code or fix the missing bootstrap and diagnostic rules, then run fresh coverage under those exact rules. If that rule maintains close-to-nominal coverage among analyses it accepts, the calibration concern can be withdrawn for the tested scope. If it undercovers materially despite diagnostic acceptance, the objection becomes a direct failure of the specified paper method.

## Retained evidence

- [Primary summary](main/summary.json), [draw audit](main/audit.json), [optimization audit](main/optimization_audit.json), and [200 per-dataset results](main/README.md).
- [Eight-week summary](sensitivity/summary.json), [draw audit](sensitivity/audit.json), [optimization audit](sensitivity/optimization_audit.json), and [paired result records](sensitivity/README.md).
- [Exact-count block check](exact_counts_block_check.json) and [source-time placement check](exact_counts_placement_check.json).
- [Pilot summary](pilot/summary.json), [frozen design](design.json), [runner](run.py), and [auditing code](summarize.py).

The draw audit reconstructed every reported percentile interval from saved draws, verified source hashes, regenerated selected datasets and first bootstrap draws from their seeds, and checked exact risk-set recurrence. Saved manifests identify the tested source snapshot and scenario. No production or manuscript source was changed.
