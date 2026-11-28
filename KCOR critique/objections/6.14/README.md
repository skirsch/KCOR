# 6.14 ? Actual post-enrollment manufacturer eligibility exclusion

**Severity:** High for Czech population selection

A full registry check finds 363,037 otherwise eligible people with first disqualifying product in enrollment week or later, including 354,608 with no earlier dose among first four. Independent 100k check finds 3,002 and 2,935 respectively. The direction and size of endpoint effects remain unknown. Determine original study population intent before changing construction and disclose the specific Czech rule.

**Related issues:** 5.6.

**Origin:** Existing 5.6 only partially resolved; new quantitative evidence.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md) ? [B](../../iterative%20review/evaluations/6/blind-b.md) ? [R](../../iterative%20review/evaluations/6/resolution.md).

**Status:** addressed by application-specific disclosure in m128, preserving m127. Sensitivity to alternative eligibility remains an application-level question, not a general KCOR estimator defect.

## Original intent and actual rule

Commit f58d8a8f explicitly introduced exclusion of non-mRNA vaccine records. The builder examines recorded manufacturer codes in the first through fourth doses over the available history before constructing enrollment cohorts, retaining blank codes and the mapped Pfizer/Moderna product codes. The user confirmed the retrospective restriction is the intended Czech study population. It is retained unchanged. Neither a vaccine-safety conclusion nor an assumption that all non-mRNA products were withdrawn is made.

The paper states the scope and possible retrospective survivor selection, and identifies AstraZeneca and Janssen as examples of excluded non-mRNA COVID-19 vaccine products whose EU marketing authorizations were withdrawn in 2024. Sources: [EMA AstraZeneca](https://www.ema.europa.eu/en/medicines/human/EPAR/vaxzevria) and [EMA Janssen](https://www.ema.europa.eu/en/medicines/human/EPAR/jcovden).

## Independent aggregate audit

The read-only audit reconstructs the converter's infection-row, birth-band, survival-at-entry, date-consistency and manufacturer restrictions. The 100k sample reproduces the external review's counts exactly: 85,600 eligible before product filtering, 7,091 excluded, 3,002 future-only exclusions, including 2,935 with no prior recorded dose among the first four.

The full registry contains 10,779,090 otherwise eligible persons at the beginning of week 2021-24; 874,157 are excluded by the product rule. The remaining 9,904,933 exactly match the sum of the cached all-ages enrollment risk sets across doses in the supplied CMR workbook. Of those excluded, 363,033 have a disqualifying product in the enrollment week or later and none before it; 354,604 have no earlier recorded vaccination among the first four doses. Prior to the converter's additional first-dose-after-death consistency check, the respective counts are exactly the review's 363,037 and 354,608. The four-record difference is accounted for by that check.

The supplement reports the corrected counts and defines the eligible-population aggregate including unknown birth years. The effect on fitted frailty and KCOR under an alternative eligibility rule has not been evaluated; no claim that the intended retrospective eligibility rule eliminates survivor selection. No modification to converter, method, or results.

[Audit script](audit.py) | [Aggregate counts](counts.json) | [Cached input check](cached-input-check.json) | [Paper patch](paper.patch) | [Supplement patch](supplement.patch) | [Reference patch](refs.patch).
