# Round 18 resolution review of m181 against m180

The original loss of Monte Carlo rows is repaired. The new one-dose reconstruction can, however, produce a large spurious contrast from two identical cohorts. In a written workbook, supplying an exact all-ages total for just one dose changed the anchored ratio from 1 to 0.490 or 2.043, depending on which dose supplied the total. The fractional-count correction passes. The requested section-number edit is present. The ordinary Czech S6 replay matched exactly across the two candidates.

| Round 17 objection | Independent disposition |
| --- | --- |
| 17.1. Direct all-ages rows lost in a mixed Monte Carlo batch | The original cross-replicate omission is resolved. With all source birth years in the analysis range, standalone, batched and shuffled results agree. The approved extension to reconstruction by dose introduces a population mismatch when the analysis range excludes a source group. |
| 17.2. Exact equality rejects valid fractional totals | Resolved. The original point is restored. Material fractional discrepancies and one-person integer discrepancies are still rejected. |
| 17.3. Incorrect stress-test section reference | The approved change from §3.4 to §3.3 was made in §5.2. The surrounding sentence is preserved. |

## 17.1. Original failure and controls

I reran the original written-workbook example on m180. A fixed replicate produced 20 direct all-ages rows alone and zero once a second replicate supplied its own totals. Its 20 age-standardized rows survived, with endpoint 1.428233994. In m181 the new workbook regression passed with exact totals supplied for both doses of the second replicate and with only its dose-0 total supplied. Each replicate retained 20 direct and 20 age-standardized rows. All in-memory fields matched processing alone, together and after a row shuffle. The written anchored and unanchored ratios agreed to Excel precision. The pre-existing age-standardized pooling and normalization-status regression also passed. The partial-scope source selector kept a supplied wider dose separate from incomplete components. [Original objection](../../../objections/17.1/README.md) · [Approved correction](../17/mc-totals/README.md)

## Regression in one-dose reconstruction

I used a single replicate with exhaustive disjoint histories for three birth years. Doses 0 and 1 had **identical** weekly risks and deaths in every birth year. The source contained all three groups throughout. The all-ages numerator and denominator therefore have identical histories, and their KCOR should be 1 at each defined time.

To reproduce, take the dose-0 histories for 1940 and 1950 from `test_round7_workbook.input_frame("2023_01")`. Add a 1970 group with 40,000 people at opening risk, 40 deaths per week in weeks 1–8 and 180 per week in weeks 9–20. Subtract weekly deaths from each subsequent opening risk. Duplicate the three histories as dose 1, assign `mc_id=0`, and form every supplied total by summing all three groups by dose and week. Process dose 1 versus 0 with aggregate reporting enabled, using the round 16 workbook test's 1940–1950 analysis range.

| Supplied all-ages totals | m180 direct endpoint | m181 direct endpoint | m181 unanchored endpoint |
| --- | ---: | ---: | ---: |
| Both doses | 1.000000 | 1.000000 | 1.000000 |
| Neither dose | 1.000000 | 1.000000 | 1.000000 |
| Dose 0 only | No direct rows | 0.489556 | 0.556041 |
| Dose 1 only | No direct rows | 2.042668 | 1.798430 |

All four m181 conditions wrote 20 direct rows labeled `All Ages`, with endpoint status `ok` and declared scope `exhaustive`. The default reader range of 1930–1960 also gave 0.489556 with only the dose-0 total. Expanding the reader range to 1930–1970 with the **same input file contents** restored both anchored and unanchored ratios to 1.

The reader applies the Monte Carlo birth-year filter before selecting aggregate sources, retaining supplied totals while discarding out-of-range individual groups (`code/KCOR.py`, lines 6355–6362). The new dose-specific source selection at lines 4315–4328 therefore compares a supplied three-group total with a reconstructed two-group total. The previous reader's early filtering also affected component-only aggregates. The new, non-unit one-dose contrast first appears in m181.

**Impact and correction.** A wider external Monte Carlo input, or a previously generated workbook reread with a narrower range, can report a false direction and magnitude with a successful status. The packaged generator restricts its master person population before resampling (`code/KCOR_CMR.py`, lines 647–653), so using the same range for generation and analysis avoids the demonstrated mismatch. For a full-input all-ages output, reconstruct the missing total from the complete source before filtering displayed ages. If the intended target is the selected birth-year range, construct both sides for that range and identify it in the output. Withhold a mixed comparison when the populations cannot be aligned. A regression test should exercise all four supplied-total layouts above with a source group outside the analysis range.

## 17.2. Fractional reconciliation

I reproduced the m180 rejection in the original 20-week fractional workbook. A discrepancy of 4.547473508864641 × 10⁻¹³ person between the correctly supplied risk total and the sum of its components suppressed both dose histories' direct all-ages point. Reconstructing those totals gave 1.003222772976552. On m181 the supplied and reconstructed written points both equal that value. The m181 test rejected a 0.001-person discrepancy and a one-person integer discrepancy, and preserved the compatible born-1960-or-later and individual-age points. The prior nested-total regression also passed, including its three-draw reporting check. Fractional expected counts in this comparison support point estimation. [Original objection](../../../objections/17.2/README.md) · [Approved correction](../17/fractional-total/README.md)

## 17.3. Manuscript pointer

The manuscript source differs at the single requested section number. The candidate's clean PDF gives §3.3 on page 38, and the redline marks the replacement on page 42. Sections 3.3.1 and 3.3.2 cover non-gamma frailty and quiet-window contamination. Section 3.3.2 also reports a population-size sweep. The explicitly named sparse-event comparison is in the simulation comparison table. The correction closes the approved section-number edit. [Original objection](../../../objections/17.3/README.md)

## Ordinary Czech results and verification scope

I ran each candidate's ordinary estimator on the same retained Czech S6 worked-example input with 2021 week 24 enrollment, doses 2 versus 0, partial aggregate scope, and 149 weeks. The returned frames matched exactly, including row order, missing values and all 46 fields in all 1,192 rows. The direct all-ages endpoint was 1.1355510686037023 in both runs.

The frozen manifests contain 1,380 files for m180 and 1,382 for m181. The only changes listed are two code files, the manuscript Markdown, TeX and PDF, and the two added tests. I separately checked the hashes of the changed and added files and 30 numerical or figure files in the Table 8, Table 9, S6 and associated assessment directories against the respective manifests. The targeted workbook runs used scratch output. No full bootstrap or simulation campaign was rerun. [Frozen m180 inputs](../17/inputs.json) · [Frozen m181 inputs](inputs.json)
