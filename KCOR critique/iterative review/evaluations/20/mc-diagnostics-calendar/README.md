# Round 20.1 and 20.2: Monte Carlo reporting and calendar verification

Candidate: m185, based on m184. No changes to the mathematical estimator or empirical archive.

## Approved implementation

- **20.1:** The batched summary now calls the same fit-flag predicate as ordinary bootstrap reporting. Finite flagged values remain in the original and bootstrap arithmetic; they are labeled fit-flagged and audit-only. Numeric completion, complete flagged draws, endpoint flags (including unavailable numbers), and missing endpoint assessments are counted separately. Percentile and basic intervals are marked audit-only and not evidence of identification if the original or any included draw is fit-flagged. Original reporting-week selection, population signatures, roster validation, missing-draw suppression, and both interval formulas remain unchanged.
- The batched workbook additionally exposes the ordinary residual-screen flag and unavailable state with replicate-specific parameter lookup, and the summary counts them separately from fit flags. The gap-assumption interval screen is not assessed in this path; that fact is stated explicitly rather than suggesting a pass.
- **20.2:** Within each iteration, compare the supplied ISO label to the canonical ISO year/week derived from the calendar date, at source age/dose resolution before reporting aggregation. A missing or malformed label in a supplied column is rejected, while hyphen/underscore and optional zero padding are accepted when equivalent; an absent label column is derived from the calendar. DateDied supplies the aggregation key and quiet mask. Contradictions propagate through the existing source validation and population dependencies, with explicit `invalid_iso_calendar` status, withholding anchored, unanchored, and raw comparison values; unrelated groups and replicates remain intact. Rows are not silently dropped or merged across weeks because of a contradictory label.

## Executed verification

Run from repository root (native Windows Python, written Excel workbooks):

- `python -B work/kcor-review-candidates/m185/test/test_round20_mc_diagnostics_calendar.py`: new regression. The five identical 20-week original/draw fixtures reproduce finite NOT_IDENTIFIED fallback output; all five endpoints identified as flagged, 4/4 numerically complete draws retained, and both interval arithmetic results preserved but qualified as audit-only. A separate well-identified 40-week fixture has clear fit status and unqualified intervals. Original-only and draw-only flag injections confirm each independently qualifies both intervals. Calendar tests compare complete actual written workbook output for equivalent separator/padding variants, and with an omitted optional ISO column. Three altered draws respectively have unique labels shifted by eight weeks, a duplicated week label, and malformed/missing labels. All 40 rows of the affected birth-year and its dependent all-ages/ASMR contrasts are unavailable, with explicit `invalid_iso_calendar`; both unrelated ages and two unrelated draws compare exactly. Summary withholds intervals when 3/4 required draws are unsupported.
- `python -B work/kcor-review-candidates/m185/test/test_round16_mc_pooling.py`: standalone/batched/permuted and written output parity; replicate-specific weights and flags.
- `python -B work/kcor-review-candidates/m185/test/test_round17_mc_totals.py`: full and one-dose totals; standalone/batched/shuffled and written workbook parity, partial scope.
- `python -B work/kcor-review-candidates/m185/test/test_round18_mc_sources.py`: all four total layouts, hidden ages, independent raw comparator, invalid source/total isolation, mixed and partial coverage, unequal horizons, aggregate opt-out.
- `python -B work/kcor-review-candidates/m185/test/test_round19_mc_reader.py`: exact reporting week, complete roster, all missing-draw cases, matched population/interval arithmetic, source metadata and raw fractional comparison.
- `python -B work/kcor-review-candidates/m185/test/test_round19_mc_producer.py`: actual converter, scopes, dose cap and absent categories, shared calendar, sampled records and aggregates; actual Makefile configuration-change and failed-producer retry.
- `python -B work/kcor-review-candidates/m185/test/test_round19_overlap.py`: prior impossible-history tests and archived Table 8 disjoint reconstruction (174 weeks/2,511,927 people and 140 weeks/2,450,449 people) plus fractional-history distinction.

**Valid-input MC exact parity:** separately ran the new regression's valid-input export mode against m184 and m185 (five iterations, three birth years, 40 weeks, quiet mask explicitly set to 2023-14 through 2023-35 after loading configuration). All 1,000 rows × 45 pre-existing fields passed `pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True, check_dtype=True)`. m185 adds only the two residual diagnostic fields (47 columns total). All five All Ages endpoints were exactly 1.754700159085343. This is the three-age regression's endpoint, not a claimed reproduction of the separate one-age review fixture.

**Ordinary S6 exact parity:** replayed `evaluations/18/mc-source-consistency/ordinary_s6_parity.py` in separate processes for m184 and m185 using temporary pickle destinations. Both: 1,192 rows × 46 columns; last All Ages KCOR = 1.1355510686037023. Full frames including dtypes compared exactly and attributes were equal.

No full person bootstrap or empirical MC coverage calibration was run. The five-draw fixtures are deterministic diagnostic tests, not evidence of calibrated coverage or new empirical results. No error in archived published inputs is asserted. The Windows converter regression covers its single-worker path.

## Changed code and regression

- `work/kcor-review-candidates/m185/code/KCOR.py`
- `work/kcor-review-candidates/m185/code/bootstrap_reporting.py`
- `work/kcor-review-candidates/m185/test/test_round20_mc_diagnostics_calendar.py`

