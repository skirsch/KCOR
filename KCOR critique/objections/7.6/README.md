# 7.6 - Czech diagnostic pass overstates the actual checks

**Severity:** High for validation claim

**Origin:** Returning 6.3, new window-omission evidence

**Related issues / root causes:** 6.3, 4.9, 5.11.

**Finding:** Two independent reviewers reproduce the omission comparison on the existing S6 inputs: youngest dose-2 frailty spans 24.89-83.70 versus 71.83 with all windows, anchored endpoint 1.3362-1.5551 versus 1.5075 (point-only sensitivity). All-ages endpoint is steadier, 1.1153-1.1341 versus 1.1356. These do not establish a universal stability threshold. The supplement acknowledges unevaluated omission stability, but the introduction and ?5.2 claim all criteria passed; blanket non-reporting also conflicts with deliberately flagged descriptive output.

**Proposed scope:** Narrow the claims to measured checks, report the point-only window sensitivity distinctly from uncertainty intervals and distinguish numerical acceptance, residual screening, precision, and substantive interpretation. No new automatic rejection rule.

**Review evidence:** [Blind A](../../iterative%20review/evaluations/7/blind-a.md) | [Blind B](../../iterative%20review/evaluations/7/blind-b.md) | [Resolution](../../iterative%20review/evaluations/7/resolution.md)

**Status:** implemented in m132 (manuscript and reproducible diagnostic evidence; estimator unchanged).

All seven full-window points independently match the current scaled fit. The omission ranges are point sensitivities, not confidence intervals; full estimates retain the 999-person-bootstrap intervals. No new threshold or rejection rule.

| Birth band | Full KCOR [95% CI] | Omission range |
|---|---|---|
| 40-49 | 1.507 [0.824, 2.712] | 1.336-1.555 |
| 50-59 | 1.228 [0.913, 1.824] | 1.205-1.248 |
| 60-69 | 1.295 [1.086, 1.577] | 1.272-1.295 |
| 70-79 | 1.301 [1.173, 1.441] | 1.277-1.330 |
| 80-89 | 1.204 [1.108, 1.312] | 1.139-1.217 |
| 90-99 | 1.012 [0.919, 1.114] | 1.004-1.035 |
| All ages | 1.136 [1.086, 1.189] | 1.115-1.134 |

[Full results](window-omission.csv) | [Detailed fits](window-omission-detail.csv) | [Reproduction script](window-omission.py) | [Paper patch](paper.patch) | [Supplement patch](supplement.patch).

**Round 8 independent disposition:** partial; remaining concrete issue tracked in [8.9](../8.9/README.md).
