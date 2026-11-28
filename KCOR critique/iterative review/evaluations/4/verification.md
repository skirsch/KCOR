# Integrated candidate verification

Table 8: ordinary workbook and first matched draw were replayed. The following 999-draw intervals are from the exact same survivor-model study, not rerun wholesale in this round.

| Enrollment | Pair | KCOR [95% percentile interval] |
|---|---|---|
| 2021_24 | 1/0 | 1.01277 [0.95551,1.06827] |
| 2021_24 | 2/0 | 1.03309 [0.97038,1.10560] |
| 2021_24 | 2/1 | 1.02006 [0.95776,1.09408] |
| 2022_06 | 1/0 | 1.01255 [0.95757,1.07539] |
| 2022_06 | 2/0 | 1.05566 [0.99626,1.12107] |
| 2022_06 | 2/1 | 1.04257 [0.97537,1.10948] |

Table 9: rerun all 999 draws with the integrated proportional route.

| Scenario | KCOR [95% percentile interval] |
|---|---|
| benefit | 0.83359 [0.77412,0.89705] |
| harm | 1.16170 [1.08652,1.24845] |

Quiet-window scan: 78/78 pass. Source and manuscript hashes for all four sealed editions verified. The full candidate inputs are preserved in the overlay and manifest.

## Core (module-off) comparison requested by reviewers

The same frozen code, dates, input and 999 paired person-bootstrap draws were used with the wave disabled in both fitting and inversion. These post-review-start checks are not part of the frozen blinded input.

| Enrollment | Pair | Core off [95% CI] | Sensitivity alpha=1.18 [95% CI] |
|---|---|---|---|
| 2021_24 | 1/0 | 1.01619 [0.95820,1.07220] | 1.01277 [0.95551,1.06827] |
| 2021_24 | 2/0 | 1.04560 [0.98225,1.11873] | 1.03309 [0.97038,1.10560] |
| 2021_24 | 2/1 | 1.02894 [0.96568,1.10269] | 1.02006 [0.95776,1.09408] |
| 2022_06 | 1/0 | 1.00661 [0.96182,1.05325] | 1.01255 [0.95757,1.07539] |
| 2022_06 | 2/0 | 1.01662 [0.96976,1.07188] | 1.05566 [0.99626,1.12107] |
| 2022_06 | 2/1 | 1.00995 [0.95708,1.06750] | 1.04257 [0.97537,1.10948] |

Both sets have all six intervals containing one; neither establishes exact equivalence in this empirical control. No module-off draw flags or failures. [Module-off code](table8_module_off.py), [results](module-off-table8.json).

## Ordinary-workbook interval check

Read the exact week-80 rows from the frozen fitted workbooks (1950 birth cohort, dose 1/0).

| Scenario | Same KCOR point | Ordinary analytic interval | Refitted person-bootstrap interval |
|---|---:|---|---|
| Benefit | 0.833593 | [0.814370,0.853270] | [0.774117,0.897055] |
| Harm | 1.161701 | [1.137278,1.186649] | [1.086518,1.248454] |

The ordinary workbook interval does not account for the full refitting procedure. This confirms the specific reporting mismatch without changing the frozen candidate.
