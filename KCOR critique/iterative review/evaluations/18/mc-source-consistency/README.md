# 18.1?18.4: validated full-source Monte Carlo aggregates

Persistent checkpoint m182, parent m181. These four changes repair the shared population-selection and reporting defects without changing the fitter, frailty correction, ASMR target, or ordinary estimator. No manuscript changes.

## Correction

- **18.1:** Resolve the direct all-ages population on each replicate's complete raw eligible history, after enrollment/dose filtering but before applying the individual-age display range. Select an existing inclusive total or reconstruct the missing total from the same replicate's complete disjoint input. The configured MC birth-year range still selects the individual-age and ASMR population; it no longer changes the direct all-ages source. In partial scope, do not mix a supplied wider total of one dose with an incomplete reconstructed total of another.
- **18.2:** Apply the existing fixed-cohort component-history and aggregate-reconciliation checks on each original replicate, with that replicate's own reporting horizon. Preserve optional aggregate metadata conventions and the existing exhaustion rule. Propagate invalid component/aggregate status to affected fits and raw comparisons only; do not drop invalid supplied totals and substitute another population. A hidden invalid component invalidates the associated reconstructed/exhaustive aggregate, not the displayed age group merely because it shares an age bucket. The ordinary validators are unchanged.
- **18.3:** The raw KCOR comparator now uses the same prepared direct all-ages rows in the existing replicate-aware dose-pair/reference loop. No second sheet-wide all-ages computation. Counts, cumulative hazards, pairing, reference and output are therefore replicate specific.
- **18.4:** In a declared partial extract with no wider supplied total, report "All available ages" and the original supplied birth-year groups plus explicitly zero enrollment groups separately by dose and replicate. Keep "All Ages" for the inclusive supplied-total population. The MC summary separates population signatures instead of mixing available and wider total comparisons under one label. Different coverage is disclosed, not prohibited; ASMR remains defined by its own configured age range.

## Tests and results

`test/test_round18_mc_sources.py` exercises the actual workbook pipeline, including writing and reading the workbook:

| Test | Result |
| --- | --- |
| Both, neither, numerator-only and denominator-only exact supplied totals, with a third raw birth group (1970) outside the display range | All four layouts have identical complete direct points and trajectories, at narrow 1940?1950 and default 1930?1960 display ranges. Identical dose histories give KCOR=1 at every defined week for all four layouts; the old one-dose points were 0.489556 and 2.042668. Non-null trajectories also agree across layouts and match the same raw input with wider display through 1970. The ASMR target still uses displayed ages only. |
| Raw comparator for the above reconstructed/supplied inputs; two distinct totals-only replicates | Every defined weekly `KCOR_ns` agrees with direct arithmetic from the full source counts (post-skip cumulative negative-log survival, anchored at the configured reference); written values retain separate replicate identity. |
| Internally valid but contradictory supplied exhaustive total; malformed hidden 1970 component; malformed displayed 1940 component | Affected direct normalized and raw comparisons unavailable; contradictory total marked `invalid_aggregate_counts`. Hidden/total failures do not propagate into otherwise valid displayed age/ASMR results. Displayed component failure propagates to its own contrast, direct aggregate and positive-weight pooled contrast, not unrelated age or replicate. |
| Mixed partial and full-supplied replicates, plus partial one-dose supplied total | Available-age replicate reports numerator 1940,1955,1970 (1955 explicitly zero) and denominator 1940,1950,1970 with `different_supplied_age_groups`; full supplied-total replicate has `All Ages` label. The partial supplied-vs-incomplete reconstructed comparison remains unavailable. Actual written workbook and MC summary preserve both population identities. Raw comparator again agrees with independent arithmetic. |
| Unequal replicate follow-up (20 vs 17 weeks), aggregate reporting opt-out | The shorter replicate retains exactly the same 17 rows in both modes; opt-out produces no direct aggregate rows. |
| Existing round-16 and round-17 MC regressions | Both pass exact standalone, batched, shuffled comparisons and written-workbook checks, including per-replicate weights/status propagation and full vs one-dose supplied totals. |

These are deterministic fixture/individual simulation realization checks, not estimated confidence intervals. No new 999-draw bootstrap or full simulation campaign was run.

## Ordinary parity

Independent reassembly of the actual retained Czech S6 input (`data/Czech/KCOR_CMR.xlsx`, 2021-24, dose 2 vs 0, 149 weeks through 2024-16, partial scope, existing reference and fit configuration) and ordinary point replays in m181 and m182 gave `1192 ? 46` output frames. `pandas.testing.assert_frame_equal(..., check_exact=True)` passed across every field, row order, missing value and status. Direct all-ages endpoint remained exactly 1.1355510686037023. No new interval was estimated for this point-only parity run. The replay script `ordinary_s6_parity.py` accepts a candidate root and scratch result path; it does not write to the candidate.

One unrelated legacy test (`test/test_round7_workbook.py`) fails its last test-only assertion identically in m181 and m182. Its forced inversion failure hits the direct -2 aggregate rather than the pooled contributor assumed by that assertion; all preceding date, missing-week, overlapping-population and failure checks print identical outcomes. This is not a regression from 18.1?18.4 and was not modified. The current replicate-specific controlled flag test passes.

Commands (relative to candidate): `python -B test/test_round18_mc_sources.py`, `python -B test/test_round16_mc_pooling.py`, `python -B test/test_round17_mc_totals.py`. Actual workbooks and ordinary replay outputs were created in scratch locations outside the candidate.

Candidate changes: `code/KCOR.py` and new `test/test_round18_mc_sources.py`; exact patches are stored here. An ignored bytecode cache produced by the syntax check is excluded from the scientific input manifest.
