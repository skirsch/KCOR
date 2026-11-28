# 19.5 — shared Table 8 overlap feasibility validation in m184

The declared Table 8 layout retains the 1940 and 1950 constituent pseudo-cohorts and the inclusive 1960 pseudo-cohort. The 1960 label supplies the direct all-ages history; all three groups remain in the separately weighted pooled estimator. No estimator, fit, population definition or archived data were changed.

`code/fixed_cohort_history.py` now shares construction of the disjoint death-week and final-survivor cells with `code/bootstrap_reporting.py`. After component histories have passed death-only accounting, the inclusive group's death cells less the two constituent death cells must be nonnegative in every week, and its final surviving population less both constituent surviving populations must also be nonnegative. Together with the death-only risk recursion, these conditions enforce feasibility of the opening and every later risk set. Integer histories must reconcile exactly. Fractional expected-count point inputs retain roundoff tolerance; the person bootstrap separately requires integer raw counts. Omitted suffixes after complete exhaustion still use the existing rectangular-history reconstruction.

The point path checks each available inclusive dose independently, marking an impossible family at the inclusive (1960, dose) key. The existing source-history handling propagates that failure to the corresponding direct all-ages and pooled comparisons. Independent constituent histories and comparisons between unaffected doses remain available. Already-malformed source histories retain their original history diagnostic; an independently valid inclusive history is not declared inconsistent merely because a constituent history cannot be assessed. The person bootstrap still rejects malformed inputs globally rather than sampling a negative residual population.

## Verification

`python test/test_round19_overlap.py` passed. It checks both archived Table 8 input sheets, and reads back actually written `dose_pairs` sheets for a 20-week valid fixture and each malformed variant. Each malformed input retains individually consistent death-only histories; the only invalid group is `(1960, 1)`.

| Malformed variation | Point results involving dose 1 (1960, direct all-ages −2, pooled 0) | Bootstrap |
| --- | --- | --- |
| Inclusive opening risk 30,000 versus constituent opening risks totaling 40,000 | KCOR unavailable, `invalid_risk_history` | Rejected |
| Inclusive weekly death count less than the sum of constituent deaths, while risk and survivor totals are sufficient | KCOR unavailable, `invalid_risk_history` | Rejected |
| Inclusive opening risk 40,001 versus constituent opening risks totaling 40,000, but too few final survivors | KCOR unavailable, `invalid_risk_history` | Rejected |

All constituent and unrelated direct comparison KCOR values agree with the valid fixture; the unaffected 2/0 pooled comparison remains finite. (The existing pooled weights depend on the supplied histories, including other doses; the weighting rule is unchanged.) A fractional expected-count fixture passes point validation but is correctly refused by the integer person bootstrap.

| Archive sheet | Weeks reconstructed | Disjoint people | Point and bootstrap validation |
| --- | ---: | ---: | --- |
| 2021_24 | 174 | 2,511,927 | Accepted |
| 2022_06 | 140 | 2,450,449 | Accepted |

The reconstructed integer cell arrays are checked for element-wise equality with the original bootstrap reconstruction formula, not only equality of totals.

`python test/test_table8_sex_history.py` also passed, covering the original archives, the dedicated Table 8 replay input, a missing positive-risk component suffix, unaffected constituent output values, and genuine component exhaustion. `python test/test_round10_batch.py` passed (including equivalent exhaustion encodings and matched draw intervals); `python test/test_round11_batch.py` passed (including original dedicated Table 8 pooled point values `[1.0161931143408671, 1.0455992196501802, 1.028937516791173]` and the exhausted/sparse fixtures). The published full 999-draw bootstrap was not rerun; archived results are unchanged.
