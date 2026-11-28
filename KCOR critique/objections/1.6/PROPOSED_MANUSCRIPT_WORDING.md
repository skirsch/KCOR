# Proposed manuscript wording for objection 6

These are reviewable replacements for the four mathematical statements in the [current manuscript](../../../documentation/preprint/paper.md). They are proposals in the critique folder; the manuscript and production code are not edited here. The derivations and checks are in the [scaling](objection6_constant_scaling/README.md), [weekly inversion](objection6_weekly_inversion/README.md), [shape](objection6_shape_claims/README.md), and [multi-window implementation](objection6_weekly_inversion_multiwindow/README.md) modules.

## 1. Constant hazard multiplier (§2.4.4)

> Under the Gompertz–gamma model with fixed Gompertz slope, multiplying an already marginal observed hazard by a positive constant \(c\) produces another curve in the model family with \(k^*=ck\) and \(\theta^*=\theta/c\). The numerical frailty parameter is therefore not invariant to this operation, although its parametric identifiability is preserved when the full curve is informative. If instead \(c\) multiplies each individual's baseline hazard before aggregation over frailty, the model has \(k^*=ck\) with unchanged frailty variance \(\theta\); the resulting marginal hazard is generally not \(c\) times the original marginal hazard. Any cancellation of a constant multiplier in an anchored KCOR ratio concerns the final contrast, not invariance of the frailty parameter.

This replaces the claim that scaling observed hazards leaves curvature and the identified \(\theta\) unchanged. Its last sentence assumes the same multiplier at reference and target times and valid model refitting.

## 2. Weekly reconstruction (§2.4.2 and Step 2 of §2.5)

> For an observed **integrated weekly hazard** increment \(h_j=H_{\mathrm{obs}}(j+1)-H_{\mathrm{obs}}(j)\), the exact gamma-frailty update is
> \[H_{0,j+1}-H_{0,j}=(1+\theta H_{0,j})\frac{\operatorname{expm1}(\theta h_j)}{\theta},\]
> with the \(\theta\to0\) limit \(h_j\). The update \((1+\theta H_{0,j})h_j\) used in the delta-iteration reconstruction is a first-order approximation. Final frailty normalization instead uses the exact cumulative inversion. Because the reconstruction supplies offsets to later quiet windows, its approximation error should be included in multi-window sensitivity checks, especially when a diagnostic depends on the sign of a small offset.

This describes the currently written and implemented reconstruction. If the internal code is changed to the exact update, the Step 2 description should instead state that exact update.

## 3. Parametric identification (§2.4.4 proposition)

> With fixed known \(\gamma>0\), \(k>0\), and \(\theta\ge0\), the Gompertz–gamma model implies
> \[\frac{1}{h_{\mathrm{obs}}(t)}=\frac{\theta}{\gamma}+\left(\frac{1}{k}-\frac{\theta}{\gamma}\right)e^{-\gamma t}.\]
> In the exact model, the intercept and coefficient of \(e^{-\gamma t}\) on a nondegenerate interval identify \(\theta\) and \(k\), even when the observed cumulative hazard is locally linear. Nonzero observed cumulative-hazard curvature is neither necessary nor sufficient for positive frailty: its sign is determined by \(\gamma-\theta k\). With finite weekly counts, precision and stability of the joint fit must be assessed separately; a curvature flag alone cannot establish identification or its failure.

This replaces the parenthetical “equivalently, depletion-linked curvature with \(\theta>0\)” and the omitted one-to-one curvature rationale. The exact-curve identification argument does not establish that the current finite-sample fitter is well calibrated.

## 4. Post-normalization shape diagnostic (§2.6.2)

> Under a correctly specified Gompertz baseline, depletion-neutralized cumulative hazard follows \(H_0(t)=k(e^{\gamma t}-1)/\gamma\) in a quiet window, plus any persistent offset carried from an earlier wave. Thus it should match the fitted Gompertz cumulative curve after accounting for the offset. Equivalently, it is linear in transformed time \(x=(e^{\gamma t}-1)/\gamma\), with an offset-specific intercept. Approximate linearity in ordinary time may be used as a short-window visual summary only with a tolerance that accounts for the curvature expected from \(\gamma\) and the window length. Residual departures beyond that expected shape warrant investigation of the fit, window choice, and offset model.

This preserves the intended diagnostic but compares it to the model's own predicted shape. With \(\gamma=0.085/52\) per week, the exact 52-week Gompertz cumulative curve has a midpoint departure from its straight chord equal to about 1.06% of the cumulative hazard at week 52.

## What this resolves

The four mathematical wording problems have explicit corrections. The isolated multi-window check also demonstrates that the internal weekly approximation can cross the current fitter's hard diagnostic threshold on a model-generated case, while it changes an accepted-versus-accepted KCOR point by about 0.000214 in another case. It does not estimate how often this occurs in applied data or establish interval coverage. These are separate implementation and validation questions.
