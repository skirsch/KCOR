# 6.7 ? RMST/Cox targets and shared-frailty comparative evidence

**Severity:** Moderate to high for comparator claims

Different frailty distributions produce different true marginal survival/RMST even at equal fixed-frailty treatment hazard. Review remaining bias language and unsupported shared-frailty numbers/extrapolation; ascertain the original comparator target and provenance. Oracle theta validation already disclosed.

**Related issues:** 1.5, 2.1.

**Origin:** Earlier interpretation issue resurfaces in additional passages.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md) ? [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** addressed in m124, preserving m123.


The shared-frailty table first appeared in commit 16c29ef4 (2026-01-06), which changed manuscript files and a coverage log, without adding a shared-frailty fit or results. The S7 generator and saved diagnostics do not contain shared-frailty Cox fits. The three-method table and the curve calibrated to 0.87 and 0.94 cannot be verified from the available repository evidence. They are removed rather than replaced by a new simulation. The standard Cox demonstration and known-frailty KCOR results (including its existing bootstrap intervals) are unchanged.

The paper now distinguishes a fixed-frailty treatment null from the actual marginal survival and survivor hazard estimands. The reproduced historical grid retains its numerical results; deviations from unity are not called estimator bias and no unmeasured sampling variances are asserted. The previous Table 11 becomes Table 10. No core KCOR method change. [Paper patch](paper.patch), [plot patches](plots.patch), [comparison patch](comparison.patch), [comparison data](comparison_table.csv).
