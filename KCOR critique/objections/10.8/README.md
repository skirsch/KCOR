# 10.8 - Pooling weight formula not fully specified

**Severity:** Moderate for reproducibility

**Origin:** New reporting question

**Related issues / root causes:** 2.9, 9.8, 8.2.

**Finding:** Implemented weights average positive first-four-enrollment-week dose-week rates unweighted, then multiply by total person-weeks; not pooled deaths/person-time. Actual 2022 normalized weights .17343,.17560,.65096 vs death-share .16244,.17864,.65893.

**Proposed scope (for discussion):** Verify original intent; preserve existing weights if intentional and state exact formula, time origin, zero-rate fallback and actual weights. Changing target requires explicit approval.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** documented in m156; m155 preserved. Independent closure not yet evaluated.


Original intent confirmed by Steve Kirsch commit 7cf64460acdcb7e672343c22f5a89328839ab303 (2025-09-04), README Option 2+ and code: expected-deaths weighting superseded person-time-only weights to reflect death burden. Original implementation explicitly excluded zero rates and used minimum-positive-rate shrinkage. Smoothing was subsequently disabled in b061b80f; current MR_smooth equals MR. No weighting code changed. Section 2.8 documents exact empirical rule; Table 8 caption records both enrollment weight vectors. weights.json records the alternative observed-death shares for comparison only. Table 8 estimator and prior intervals unchanged.


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).
