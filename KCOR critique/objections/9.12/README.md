# 9.12 - Forced-alpha origin inside ongoing nonlinear wave

**Severity:** Moderate for sensitivity only

**Origin:** New application-specific qualification

**Related issues / root causes:** 1.2, 4.7, 5.10, 7.10.

**Finding:** 2022-06 origin 2022-08 lies inside configured 2021-41 to 2022-12 wave. Fresh origin gamma assumption does not follow from nonlinear pre-origin selection; example skewness 3.762 vs moment-matched gamma 3.886.

**Proposed scope (for discussion):** Explicitly label fresh gamma assumption for affected forced-alpha sensitivity; do not infer unobserved earlier exposure or redesign module without approval.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m152 (wording only); m151 preserved. Independent closure not yet evaluated.

The Czech 2022-06 enrollment starts February 7, 2022; the two-week skip places the origin on February 21 (ISO 2022-08), inside the configured October 11, 2021 to March 21, 2022 marked wave. Section 3.1.2 now explicitly conditions the forced-alpha sensitivity on mean-one gamma frailty at that origin and states the prior nonlinear survivor selection is not reconstructed from post-origin counts. This is an application-specific assumption, not a change to the general proportional method or cohort definition. Source code and both Table 8 diagnostic/point/interval arrays byte-identical to m151.

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).
