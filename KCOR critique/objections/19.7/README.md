# 19.7 - Monte Carlo source population and capped dose labels omit their scope

**Severity:** Low reporting/provenance issue

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 19.2, 18.4, 16.1.

**Finding:** The producer intentionally resamples selected birth years and pools the top dose at or above the configured cap. Its aggregate labels and dose labels do not record those restrictions; with default bounds the 1960+ aggregate contains only 1960, and capped dose categories are not always exact-dose groups. Preserve the intended sample and cap rather than changing either.

**Proposed scope (for discussion):** Record actual source bounds and capped-dose meaning in generated outputs and summary labels/metadata. Support external full-population input separately.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/19/blind-a.md) | [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/reader-fixes/README.md).
The original birth frame and capped-dose definition remain unchanged; actual metadata is propagated and rendered in MC outputs. [Producer verification](../../iterative%20review/evaluations/19/producer-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).
