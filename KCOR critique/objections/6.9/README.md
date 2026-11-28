# 6.9 ? Anchoring interpretation overstates post-reference isolation

**Severity:** Moderate

Anchor values 0.02 and 0.01 followed by equal 0.01 increments yield anchored ratio 0.75. Dividing cumulative ratios retains pre-anchor totals. Describe change relative to the reference ratio and verify reference week indexing. Defined estimator and invalid-anchor correction hold.

**Related issues:** 2.8, 5.3.

**Origin:** New interpretation issue.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md).

**Status:** implemented in m118.

Implemented in m118. The definition and repeated explanatory passages now describe changes relative to the reference cumulative ratio and explicitly retain pre-reference contributions. The misleading time-index sentence in the figure caption is removed; the historical figure provenance remains part of 6.8. No change to the calculation. [Paper patch](paper.patch).
