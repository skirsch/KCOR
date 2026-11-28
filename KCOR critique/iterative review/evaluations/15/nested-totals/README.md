# 15.3 - Nested inclusive population totals

Persistent candidate m176 from m175. The existing shared reconciliation helper additionally checks all ages versus born-1960-or-later totals per dose/week, including feasibility of the complementary risk/death history and the remainder after the known older and unknown-birth-year components are removed. The latter are outside the converter's known-born-1960+ subtotal. This check applies to both mixed and aggregate-only sheets. Conflicting totals are withheld; when the exhaustive disjoint source independently reconciles one total, that verified total is preserved. Valid unrelated age groups are unaffected.

## Verification

- Synthetic impossible nested totals with internally valid individual histories are withheld in the ordinary mixed/partial workbook and aggregate-only default workbook. Neither anchored nor unanchored values or intervals are available. Valid group points and intervals are unchanged.
- A compatible complete source gives 1.002719 [0.875642, 1.093971] (three draws). The unaffected 1940 group remains 1.003326 [0.849748, 1.264642] (three draws). These are smoke-test intervals, not precision studies.
- Valid wider nested partial totals pass, a single incompatible dose is isolated, a known older/unknown-birth complement that cannot fit is rejected, and a separately reconciled exhaustive total is preserved.
- Actual released Czech 2021_24 source: 280,140 post-enrollment rows; every supplied inclusive total passes reconciliation, including nested complement and COVID-death counts.
- Eleven main Table 8, Table 9 and S6 archive files and the two manufacturer-status workbooks are byte-identical to m175. No full bootstrap was rerun.

A one-sentence paper addition records the nested-total requirement. No change to the core estimator.
