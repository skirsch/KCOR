# Objection 6: curvature and linearity claims

This module checks two shape claims in the [paper](../../../documentation/preprint/paper.md) under its own Gompertz plus gamma-frailty model. Run `python check.py` to reproduce [the numbers](results.json). This is an algebraic check, not a simulation of statistical coverage.

With baseline hazard \(h_0(t)=k e^{\gamma t}\), baseline cumulative hazard is \(H_0(t)=k(e^{\gamma t}-1)/\gamma\). Observed hazard and its derivative are

\[h_{\rm obs}(t)=\frac{k e^{\gamma t}}{1+\theta H_0(t)},\qquad H''_{\rm obs}(t)=\frac{k e^{\gamma t}(\gamma-\theta k)}{(1+\theta H_0(t))^2}.\]

Therefore the sign of observed cumulative-hazard curvature depends on both Gompertz aging (\(\gamma\)) and frailty (\(\theta\)). With \(\theta=0\) and \(\gamma>0\), curvature is positive. With \(\theta=\gamma/k>0\), observed hazard is exactly constant and observed cumulative hazard is exactly linear. The paper's parenthetical equivalence between nonzero curvature and positive frailty, and the ensuing one-to-one curvature rationale, do not hold. This does **not** refute identification from the full trajectory. For fixed known \(\gamma>0\),

\[\frac{1}{h_{\rm obs}(t)}=\frac{\theta}{\gamma}+\left(\frac{1}{k}-\frac{\theta}{\gamma}\right)e^{-\gamma t}.\]

The intercept and coefficient of \(e^{-\gamma t}\) identify \(\theta\) and \(k\) from an exact observed hazard curve on a nondegenerate interval, including the flat-hazard case. Finite noisy data can still identify them weakly; that is an estimation question, not a logical requirement for nonzero observed curvature.

After **correct** frailty inversion, the cumulative hazard is \(H_0(t)=k(e^{\gamma t}-1)/\gamma\), which is curved in ordinary rebased time whenever \(\gamma>0\). It is exactly linear in the Gompertz time coordinate \(x=(e^{\gamma t}-1)/\gamma\): \(H_0=kx\). A persistent wave offset changes this to \(H_0=kx+\Delta\) within a later quiet window. The paper says *approximately* linear in ordinary time, so the issue is the unspecified tolerance and the inference that systematic residual curvature necessarily signals model or window failure. In the check's 52-week example, the exact baseline's midpoint lies 1.06% of the end cumulative hazard below a straight chord, despite a perfectly specified model.

Proposed corrections: replace the equivalence in §2.4.4 with the full-trajectory reciprocal-hazard identification argument above, and distinguish mathematical identification from finite-sample precision. In §2.6.2, compare corrected cumulative hazard with the fitted Gompertz curve, or test linearity after transforming to Gompertz time. If retaining approximate ordinary-time linearity, define the expected approximation error for the chosen window length and \(\gamma\).
