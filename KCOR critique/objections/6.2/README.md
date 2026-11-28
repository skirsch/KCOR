# 6.2 ? Incorrect propagation of aggregate diagnostic flags

**Severity:** Moderate

The bootstrap reporting flag OR includes unrelated age strata for every nonpositive age label. A flagged 1930 cell marks both an independently clean -2 direct fit and the born-1960-or-later -60 aggregate. Use actual contributors or direct-fit status. This changes flag counts, not the computed percentiles in the reproduced example.

**Related issues:** 5.4, 5.7.

**Origin:** New reporting defect; independently reproduced by resolution reviewer.

**Review evidence:** [R](../../iterative%20review/evaluations/6/resolution.md).

**Status:** implemented in m118.

Implemented in m118. Direct and other independently fitted rows retain their own flags; pooled rows inherit flags only from their positive-weight, finite contributing age rows at the same date and draw. Workbook fixture with one flagged 1930 age and clean 1970 age verifies flags true for age 1930 and the pooled row, false for age 1970, -60 and -2. [Reporting patch](bootstrap_reporting.patch), [pooling patch](KCOR.py.patch), [verification](verify.py).
