# 7.9 - Fitting includes early anchor rows outside designated quiet windows

**Severity:** Moderate for nonquiet early fitting anchors

**Origin:** Returning older assumption, independently quantified

**Related issues / root causes:** 3.2, 4.5.

**Finding:** The production objective uses the first four accumulating weeks even outside configured quiet windows. In exact 100-week data with theta=4, k=.002 and all 60 designated quiet weeks unchanged, a 25% baseline pulse confined to the first four anchors moves fitted theta to 5.503, still OK. Earlier work already isolated the same assumption and tradeoff; this is not a new decision to abandon anchors.

**Proposed scope:** Clarify the early-row stability assumption in the fitting definition and applications. Preserve intentional anchor design unless the lead author approves a sensitivity or method change.

**Review evidence:** [Blind A](../../iterative%20review/evaluations/7/blind-a.md)

**Status:** addressed in m134; one approved sentence added to Section 2.5. No fitter changes.

The original anchors provide baseline-level information. The previous matched 3.2 analysis widened 2022 Table 8 intervals by approximately 30-35% when omitting the anchors, while demonstrating potential bias under contaminated anchors. The approved edit states the baseline-model assumption explicitly. The marked-wave exclusion documented in the next paragraph remains unchanged. [Patch](paper.patch).

**Round 8 independent disposition:** partial; remaining concrete issue tracked in [8.11](../8.11/README.md).
