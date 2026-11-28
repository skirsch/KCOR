# 6.10 ? Five-year age bands labeled single birth years

**Severity:** Minor

Czech configuration uses five-year birth bands and builder extracts lower-year labels. SI S5.6 calls curves single birth-year cohorts. Confirm actual input and label accordingly.

**Related issues:** 5.12.

**Origin:** New documentation issue.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md).

**Status:** implemented in m118.

Implemented in m118. The Czech five-year birth-band configuration and lower-year labels support the revised S5.6 text and three captions: 1930-1934, 1940-1944, 1950-1954. [Supplement patch](supplement.patch).
