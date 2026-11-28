# 5.9 — Sensitivity of KCOR to a shared Gompertz slope

The proportional KCOR ratio cancels a shared fractional change in the two normalized cumulative hazards. A common change in the fixed Gompertz slope can produce different fractional changes. The distinction matters even with an exact quiet-window fit. This verification concerns the ordinary proportional inversion in candidate m111, paper §§2.4.3–2.6. The optional nonlinear wave uses a different survival law.

## Derivative of the reported ratio

Keep the observed cumulative hazards and the fit weeks fixed. Write $g$ for the weekly Gompertz slope, $H_d(t)$ for observed cumulative hazard, and $\theta_d(g)$ for the variance returned by the joint fit. For $H>0$, the normalized hazard and its log derivative with respect to variance are

$$
F(H,\theta)=\int_0^H e^{\theta u}\,du,\qquad
D(H,\theta)=\frac{\partial\log F}{\partial\theta}
=\frac{\int_0^H u e^{\theta u}\,du}{\int_0^H e^{\theta u}\,du}
=\frac{H}{1-e^{-\theta H}}-\frac{1}{\theta}.
$$

The continuous limit is $D(H,0)=H/2$. For the ratio anchored at $t_0$, set $\Delta D_d=D(H_d(t),\theta_d)-D(H_d(t_0),\theta_d)$. On a differentiable branch of the fit,

$$
\frac{d\log\mathrm{KCOR}(t;t_0)}{dg}
=\Delta D_A\,\frac{d\theta_A}{dg}
-\Delta D_B\,\frac{d\theta_B}{dg}.
$$

For the unanchored ratio use the endpoint $D_d$ in place of $\Delta D_d$. These identities include refitting. First-order cancellation occurs precisely when the two terms agree. For small $\theta H$, $D=H/2+\theta H^2/12+O(\theta^3H^4)$. Thus the leading anchored sensitivity depends on the observed hazard accumulated **after the anchor** in each cohort, multiplied by that cohort’s fitted variance response.

## What the weekly fit establishes

The instantaneous quiet-hazard identity $h'=gh-\theta h^2$ gives $\partial\theta/\partial g=1/h$ when the level and derivative **at one instant** are held fixed. When $h$ varies, the value $1/h$ varies across weeks. The all-window fit returns one variance per cohort. It minimizes the summed squared weekly residuals, jointly refitting the baseline rate and variance and recomputing the gap offsets. Formally, for its objective $L(p,g)$ with $p=(\log k,\theta)$, a smooth interior minimum with nonsingular Hessian has $dp/dg=-L_{pp}^{-1}L_{pg}$ by differentiation of the first-order conditions. This depends on the actual fit rows, hazards and offsets. At changes of bound, offset branch or fallback, a two-sided derivative need not exist.

One exact cancellation case follows directly from the weekly objective. Suppose B’s observed hazard history multiplied by $c>0$ equals A’s through the endpoint, with identical fitting weeks and corresponding interior solutions. Setting $k_A=ck_B$ and $\theta_A=\theta_B/c$ scales the reconstructed baseline exposure and every applied gap offset by $c$. The predicted weekly hazard and each residual scale by $c$, so the squared objective scales by $c^2$ and the minimizers correspond. Then $F(cH,\theta/c)=cF(H,\theta)$ for every slope. The unanchored ratio stays at $c$ and the anchored ratio stays at one. Agreement only within quiet weeks is insufficient for this result.

## Exact check against the fitter

Use the example in [review B](../../iterative%20review/evaluations/5/blind-b.md). Both cohorts have constant observed hazard $c=0.001$ per week for 20 fitting weeks. The exact solution is $k=c$, $\theta=g/c$ for every tested $g$. Indeed, the marginal instantaneous hazard is $c$ throughout, giving integrated weekly hazard $c$ exactly. Over the next 20 weeks, A’s observed hazard doubles and B’s stays constant. Their cumulative hazards reach $60c$ and $40c$, and their week-20 reference ratio is one. The exact week-40 anchored ratio is

$$
R(g)=\frac{e^{60g}-1}{e^{40g}-1}=1.5+15g+O(g^2).
$$

| Annual slope ($52g$) | Fitted variance, both cohorts | Week-40 anchored KCOR |
| ---: | ---: | ---: |
| 0.065 | 1.2500 | 1.51907 |
| 0.085 | 1.6346 | 1.52506 |
| 0.105 | 2.0192 | 1.53112 |

The m111 public fitter, called with four early anchors, the first 20 weeks quiet and the wave module off, reproduced all three ratios to within $3\times10^{-13}$ and reported an accepted fit for each cohort. These values come from specified hazards without sampling. Moving the annual slope by 0.020 in either direction changes the fitted variance by about 24% and the anchored KCOR by about 0.4% relative to the middle value. This quantifies sensitivity in the review example. Practical sensitivity for Table 8 remains unverified by this calculation.

## Manuscript revision

Revision m114 explains the slope sensitivity in §2.4.3 and quotes its magnitude in the 40-week example. This note preserves the derivation for both ratios and the details of the example. The [revision record](README.md) links to the manuscript, source change, runnable verification and earlier editions.
