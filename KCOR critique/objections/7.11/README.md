# 7.11 - Remaining abstract and comparison-table bias/conditioning language

**Severity:** Moderate for estimand framing

**Origin:** Returning 6.7, additional passages

**Related issues / root causes:** 6.7, 2.1, 1.5.

**Finding:** The abstract still says depletion biases standard survival estimands without benchmark qualification. Table 3 says KCOR does not condition on survival although the input hazards use survivor risk sets; the final target is reference-frailty cumulative hazard. The prior verified Cox/RMST results and known-frailty disclosure are sound within the stated estimands.

**Proposed scope:** Qualify the benchmark in the abstract and distinguish survivor-based input from the cumulative target in the table/Box 2; no change to comparative experiments or method.

**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m135; paper wording only, estimator and simulations unchanged.

Abstract now qualifies the benchmark as fixed-frailty contrasts inferred from conventional survival summaries. Box 2 clarifies reference frailty Z=1 under the mean-one working model. Table 3 distinguishes the final target from survivor-risk-set inputs. The underlying Cox and RMST estimands and the previous comparative evidence are unchanged. [Patch](paper.patch).

**Round 8 independent disposition:** partial; remaining concrete issue tracked in [8.12](../8.12/README.md).
