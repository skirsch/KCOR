# Basic interval check at larger cohort sizes

The existing production fitter was tested without modification. The previously retained corrected-code study used 600 independent datasets and 999 bootstrap draws per dataset at each size; this separate, predefined screening extension used 40 fresh datasets and 199 pooled person-bootstrap draws at each new size. All scenarios have true normalized KCOR 1, frailty parameters 4 and 0.5, quiet weeks 6–40, and a week-52 target. Both interval formulas use each dataset's same saved draws, so they have equal width within that dataset.

| Cohort size | Percentile coverage | Basic coverage | Mean interval width | Accepted analyses |
|---:|---:|---:|---:|---:|
| 100,000, earlier full study | 577/600 (96.2%) | 592/600 (98.7%) | 0.2125 | 600/600 |
| 1,000,000, earlier full study | 502/600 (83.7%) | 586/600 (97.7%) | 0.0496 | 600/600 |
| 4,000,000, fresh screening study | 22/40 (55.0%) | 37/40 (92.5%) | 0.0212 | 40/40 |
| 16,000,000, fresh screening study | 13/40 (32.5%) | 17/40 (42.5%) | 0.00749 | 40/40 |

The 4-million-person basic interval result is uncertain: its 95% Wilson range is 80.1%–97.4%. At 16 million, the basic interval's 95% Wilson range is 28.5%–57.8%, clearly below nominal 95%. All 23 basic misses at that size had the lower endpoint above the true value. All intervals were finite and ordered; no fit failed in original or bootstrap analyses. Every interval was reconstructed from saved draws, and selected datasets and draws were replayed from their seeds.

The deterministic expected-count output of the production fitter is **1.0046145529** at both new sizes, whereas the generating model's target is 1. That offset does not shrink as the cohort grows. At 16 million people the mean basic interval midpoint is 1.00440 and the mean width is 0.00749, making it unsurprising that most intervals sit wholly above 1. This is also the expected large-cohort limit: sampling spread and either bootstrap interval's width shrink around the production fitter's population output, while its target offset remains. That limiting argument assumes the estimator and bootstrap concentrate around this deterministic output; the fresh screening results are consistent with it.

**Decision:** switching to the already implemented basic interval improves the one-million-person result, but does not resolve objection 3 across larger within-model cohorts. The joint-fit candidate removes the deterministic offset on exact counts but greatly increases finite-sample variation; it is not a validated replacement. The current objection remains specific: the production percentile interval undercovers in a model-consistent accepted scenario, and the available basic interval is not a general repair. Further work would require a revised estimator or a defensible target and scope, followed by fresh calibration. This module does not recommend changing production code.

The new screening estimates use only 40 outer datasets per size and fewer bootstrap draws than the full earlier study. They provide strong evidence of a large failure at 16 million, not a precise coverage curve. Optional wave correction, disjoint quiet windows, age pooling, and manuscript interval provenance were outside this test.

Records: [fixed design](DESIGN.md), [machine-readable summary](pilot/assessment/summary.json), [audit](pilot/assessment/audit.json), [manifest](pilot/manifest.json), and retained per-dataset count/draw files under `pilot/4000000/` and `pilot/16000000/`.
