# 6.11 ? Active baseline-rate bounds not flagged

**Severity:** Moderate for high event rates

The imposed k interval is [1e-8,0.1] per week. Noiseless k=0.2, theta=2 fits at k=0.1, theta=1.41361 with status ok. Retain intended numerical safeguards while disclosing supported range and flagging active bounds.

**Related issues:** 5.11.

**Origin:** New edge case.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** implemented in m118.

Implemented in m118. Both fit paths report k_hit_bound and a baseline_rate_bound status when otherwise accepted within 0.1% of the configured numerical rate bounds. The constrained fitted variance is retained. The ordinary fit-status mapping reports BASELINE_RATE_BOUND without overriding more severe statuses. The manuscript states the existing weekly bound interval. Noiseless high-rate example reaches 0.1 in both paths and is flagged; normal rate 0.001 recovers theta 2 with no flag. [Code patch](KCOR.py.patch), [paper patch](paper.patch), [verification](verify.py).
