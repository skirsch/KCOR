# 8.1 - Incomplete source-component histories

**Severity:** High for ordinary workbook validity

**Origin:** Returning 7.2

**Related issues / root causes:** 7.2, 4.6, 2.9: fixed-cohort and point/bootstrap calendar integrity

**Finding:** All three reviewers reproduced missing rows hidden by pre-fit aggregation. Identical two-dose data move from 1.000 to 0.982656 when one sex/week is missing; removing trailing rows with positive survivors gives 0.511294 and the bootstrap history validator still accepts. A missing last age row moves aggregate 0.999967 to 1.009796. These are deterministic point-only fixtures, not empirical confidence intervals.

**Proposed scope (for discussion):** Validate each disjoint source component before aggregation and against the requested horizon; distinguish actual exhaustion/explicit exits from lost records. Withhold only unsupported contrasts and match point and bootstrap validation. No evidence yet of affected published Czech histories.

**Review evidence:** A, B, resolution. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m137, preserving m136. Independent review not yet repeated.


The shared source-calendar validator checks before age bucketing or sex/DCCI pooling. It flags duplicate/missing weeks, invalid counts, unexplained risk-set increases, and truncation while positive risk remains. A truly exhausted component can end with zero remaining risk. Existing non-death exits between observed weeks are allowed. The sheet calendar defines the horizon in the absence of separate administrative-end metadata. Entirely omitted components or an undisclosed earlier common horizon cannot be inferred from the observed file alone.

The known Table 8 overlapping pseudo-birth-year layout intentionally relabels several histories to the same keys. Validate the stable original row multiplicity and combined pseudo-cohort histories instead of misclassifying these repeated keys as duplicates. Existing overlap-specific history reconstruction remains unchanged.

Regression tests reproduce interior sex-week omission, trailing sex-week loss, missing final age row and duplicate source row. All affected direct, pooled and reconstructed aggregate contrasts are unavailable; the independent age comparison is unchanged. Genuine exhaustion is accepted by both paths. All prior round-7 workbook checks pass. The bootstrap constructor rejects invalid sheets under its existing policy; unaffected point comparisons remain available but the existing policy withholds bootstrap intervals for a malformed sheet. Independently supplied all-ages histories are checked as their own input; reconstructed aggregates propagate component flags.

The actual four Czech input sheets (two archived age-shift sheets plus the 2021-24 and 2022-26 CMR sheets) have no invalid groups. All twelve Table 8 actual exported points reproduce stored results within 2.22e-16; archived 999-draw files and percentile summaries are byte-identical. No full resampling run was necessary since the estimator and validated subset are unchanged.

[Source validation](source-history-validation.json) | [Actual workbook verification](workbook-verification.json) | [Regression test](test_source_component_history.py) | [Shared validator](fixed_cohort_history.py).

**Round 9 independent disposition:** partial; see [9.1](../9.1/README.md). [Resolution](../../iterative%20review/evaluations/9/resolution.md).
