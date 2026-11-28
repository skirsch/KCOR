# 6.13 ? Instantaneous wave moment identity versus weekly observed excess

**Severity:** Moderate for optional alpha localization

The conditional instantaneous wave component identity is valid, but a no-wave subtraction or weekly integrated ratio can differ as survivor composition changes. Review the scale of the localization claim; identification investigation remains deferred. Supplied-alpha inversion is a separate issue.

**Related issues:** 4.7, 5.10.

**Origin:** Sharpened deferred alpha objection.

**Review evidence:** [A](../../iterative%20review/evaluations/6/blind-a.md) ? [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** claim qualification addressed in m127, preserving m126; the broader alpha-localization investigation remains deferred.

## Verified distinction

The individual baseline hazard contribution at time t is b(t) z, and the additional wave contribution is A(t) z^alpha. Among actual survivors, the observed marginal hazard is b(t) E_actual[z] + A(t) E_actual[z^alpha]. Subtracting the marginal hazard of a hypothetical no-wave population gives the actual wave component plus b(t) (E_actual[z] - E_no_wave[z]), not the actual wave component alone.

For the review's exact unit-mean exponential frailty example with previous B = W = 1, alpha = 1 and current baseline and wave rates of 0.1, the actual survivor mean is 1/3, the no-wave mean is 1/2, and the actual wave component is 1/30. Actual total minus no-wave total equals 1/60, only half of that wave component. The instantaneous identity itself is valid.

For weekly quantities the numerator and denominator integrate A(t) against different time-varying survivor moments. Their ratio is not generally the instantaneous moment ratio. Proportional moments throughout the week are sufficient for cancellation. The historical difference from dose zero is yet another proxy, not an independently measured instantaneous wave component.

At a fixed time with equal weights, sum_(i<j) (r_i-r_j)^2 = n sum_i (r_i-r_mean)^2. The pairwise and collapse objectives therefore use the same information; weighting and exclusions can change the pooled optimum, but agreement does not provide independent identification.

## Approved paper-only change

Sections 2.7.1-2.7.2 identify the wave component conditional on actual survivors and state the within-week approximation limit. The workflow requires an appropriate instantaneous measurement or justified approximation. Sections 3.4 and S2.1 distinguish historical localization experiments from identification under the integrated survivor law and qualify agreement of the two objectives. The Czech non-identification result and sensitivity-only use of alpha=1.18 remain unchanged. Numerical inversion at a supplied alpha and ordinary KCOR remain unchanged. No new numerical analysis or redesign.

## Deferred scientific investigation

When the lead author approves further alpha-identification work, investigate estimation directly from weekly survival at both bin boundaries, sharing the weekly external wave exposure across cohorts via the prespecified external intervention-effect scales, rather than using a raw total-hazard difference as if it isolated wave exposure. Validate recovery against independent known-parameter simulations of the full survivor law before making a general claim. Related to 4.7 and 5.10. This is not a completed or approved change to the estimator.

[Paper patch](paper.patch) | [Supplement patch](supplement.patch).
