# 14.4 - Inclusive aggregate reconciliation

Candidate m173, parent m172. Shared point/interval count reconciliation; no estimator change. Standard complete extracts must reconcile supplied all-ages and born-1960-or-later counts. A declared partial extract can retain a valid wider total but no joint resampling interval if constituent strata are missing. The selected scope is exported and logged. The S6 point replay explicitly declares its partial source layout.

## Verified

- Synthetic 20-week complete population: 1.802100 [1.563328, 1.875458]. Supplying the correct redundant total gives exactly the same estimate and interval; wrong supplied total has no anchored or unanchored point or interval. The 1940 age group remains 1.003326 [0.849748, 1.264642], unchanged.
- Explicit partial extract: correctly retains full-population point 1.802100, interval unavailable; displayed age-group intervals unchanged. Three draws are software regression checks, not a precision study.
- Born-1960-or-later mismatch remains isolated to that affected dose, smaller partial totals and invalid scope settings are rejected.
- Actual Czech S6 source extract: identical 1,192 point rows and all 39 pre-existing result fields before and after. Eleven Table 8, Table 9 and S6 archive files are byte-identical. No full 999-draw rerun.
- Round-12 complete-population parity test passed.
- Paper section 2.9.1 updated; clean page 18 and redline page 19. All 85 visible redline objection labels have valid PDF links.

[Clean paper](KCOR-clean.pdf) | [Redline](KCOR-redline.pdf) | [Verification](verification.json).

The broader legacy `test_round7_workbook.py` reaches the same final assertion failure in both m172 and m173 (forced failure of direct all-ages normalization versus separately weighted age pool; identical printed statuses). The calendar, missing-history and overlapping direct-population checks preceding that assertion pass in both. This pre-existing test expectation is not attributed to 14.4 or changed here.
