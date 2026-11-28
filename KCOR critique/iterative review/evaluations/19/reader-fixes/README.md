# Candidate m184 reader corrections: 19.1, 19.6, and reader side of 19.7

The changes are confined to the MC input/output provenance and interval reporting, and to use of the existing finite-ratio helper for the raw cumulative-hazard comparator. The ordinary normalized estimator, cohort selection, age aggregation, and person-bootstrap method are unchanged.

## 19.1: common endpoint, source signature and complete roster

The weekly result table remains the trajectory for each iteration. The log summary now selects the original comparison's closest observed reporting week once and states the requested and actual dates; it never chooses another week for a draw. A missing original, missing exact week, undefined endpoint, missing comparison, missing input iteration, duplicate endpoint, or changed population signature leaves the interval unavailable. The point and dated draw outcomes remain available where supported. The roster is captured from input IDs before filtering, not inferred from the finite output rows. Producer input's optional `MCPlannedDraws` also declares draw IDs 1 through N. For external legacy workbooks, the supplied input IDs constitute the roster; absent unrecorded requested iterations cannot be determined without metadata.

Run from the candidate root:

```powershell
python test/test_round19_mc_reader.py
```

The Excel round-trip test used enrollment 2023-01 and requested 2024-01-01. The point and draw 1 reached that date; draw 2 ended 2023-04-24; draw 3 had zero reference numerator hazard; draw 4's input dose was filtered out; draw 5 was entirely absent from the input but declared by the plan. Its log shows:

```text
Actual reporting date: 2024-01-01
   Iteration         KCOR  Status
           0       1.8774  ok
           1       1.8774  ok
           2  unavailable  missing_reporting_week
           3  unavailable  undefined_reference_ratio
           4  unavailable  missing_comparison
           5  unavailable  missing_input_iteration
Point estimate (iteration 0): 1.8774
Bootstrap draws: 1 valid / 5 required
Unavailable draws (missing_comparison): 1; IDs [4]
Unavailable draws (missing_input_iteration): 1; IDs [5]
Unavailable draws (missing_reporting_week): 1; IDs [2]
Unavailable draws (undefined_reference_ratio): 1; IDs [3]
Interval unavailable: original or required draws not fully supported at a common reporting date and population
```

The test asserts that no percentile or basic bound is printed, and that no weekly trajectory is shortened or extended. It also checks a shortened original (actual date 2023-05-15) with two draws reaching that week: both reported intervals agree with independent quantiles from exactly the three recorded endpoints. A partial-source signature change is diagnosed and neither the original nor the unmatched draw-only signature generates unsupported intervals. The small draw counts are mechanical tests, not coverage calibration.

## 19.6: finite positive raw reference

A deterministic fractional-count Excel input with 1e-13 weekly deaths and opening population 20,000 exercises both a tiny positive numerator ratio and a tiny positive denominator cumulative hazard. The raw comparator (`KCOR_ns`) equals 1 at the configured reference for each tested individual age and direct all-ages comparison. This passes both the ordinary workbook-processing returned rows (the ordinary dose_pairs output omits the internal raw comparator by design) and the written MC workbook. The undefined zero-reference example above remains unavailable. Both active raw comparator paths use `_finite_hazard_ratio` rather than a fixed EPS cutoff; no normalization or fitted result is altered.

## 19.7: restricted-population and capped-dose provenance

Where input provides optional `MCSourceYOBStart`, `MCSourceYOBEnd`, `MCDoseCap`, and `MCPlannedDraws`, the weekly comparison workbook carries those values and cap-aware dose labels. Direct aggregates retain the existing numeric keys, with selected source bounds appended to the human-readable labels. A default bound of 1930-1960 yields “All Ages (source birth years 1930-1960)” and “-60 (born 1960+) (source birth years 1960)”. The summary explicitly identifies the capped dose as e.g. 1+ and explains it includes all doses at or above 1. Changed bounds/cap form a distinct population signature; the test verifies interval withholding for changed bounds. External inputs without metadata retain their existing age labels and population scope; no reader display range is treated as a record-level sampling restriction.

## Further regression

`python test/test_round18_mc_sources.py` passed the five existing prior-round cases: all four inclusive-total layouts, null/non-null trajectories with hidden raw ages, totals-only raw comparator, malformed total and component isolation, partial/full population coverage labels, unequal replicate horizons and aggregate opt-out.

These checks are local test workbooks; they are not a full Czech bootstrap rerun.

An exact ordinary-S6 worked-example replay used the prior round's `../../18/mc-source-consistency/ordinary_s6_parity.py` procedure against m183 and m184 in separate processes, with outputs in scratch pickles. Both runs returned (1192, 46) and endpoint 1.1355510686037023. `pd.testing.assert_frame_equal(old, new, check_exact=True, check_dtype=True)` passed; frame attributes also matched. This is exact equality of the complete ordinary point results for the worked example, not only of the final ratio.

