# 15.2 - All available ages versus all ages

Candidate m175, parent m174. In explicitly partial input without a supplied inclusive total, the existing aggregate of supplied disjoint ages carries the visible population label All available ages, not All Ages. The separately supplied nested subtotal is not silently counted twice. The same label is exported in ordinary point, per-dose, diagnostic and summary outputs and checked across bootstrap draws. An independently supplied full total retains its original All Ages label. Partial and full populations are not mixed across doses. For a complete extract, missing dose-specific all-ages totals are reconstructed from the disjoint counts.

## Verification

- Synthetic full population: anchored 2.208449 [1.897007, 2.282762], three valid draws. The partial extract reports the older available disjoint cohort 1.003326 [0.849748, 1.264642], three valid draws, accurately labeled All available ages. Its interval exactly matches the 1940 group.
- Supplying a complete valid total for only one dose in an exhaustive input reproduces the full-population point and interval exactly.
- An explicit wider supplied total in a partial extract retains the full point 2.208449 with no joint interval, because constituent strata are missing. Supplying a full total for only one dose in a partial extract does not produce a mixed-scope contrast.
- Actual Czech S6 extract: all 1,192 rows and 40 prior result fields match exactly; supplied full total remains All Ages. Eleven Table 8, Table 9 and S6 archive files are byte-identical. No new full bootstrap.
- Three-draw intervals are regression smoke tests, not precision studies. No change to frailty fitting or core estimator. No change to nested-total validity in 15.3.

The short paper addition and cumulative linked redline describe the distinction. See the patch and verification files in this directory.
