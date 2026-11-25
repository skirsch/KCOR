# Objection 2: the optional wave correction

The optional module models an epidemic wave as raising mortality more in people with greater underlying vulnerability. The exponent \(\alpha\) sets how that extra risk depends on frailty. At \(\alpha=1\), the wave and baseline hazards both scale in direct proportion to frailty. The manuscript says that ordinary KCOR handles this case. [Manuscript §2.7.1](../../documentation/preprint/paper.md#271-optional-non-proportional-hazard-nph-exponent-model)

## The identity case

Let \(B\) and \(W\) be cumulative baseline and wave hazards on the frailty-neutral scale. At \(\alpha=1\), a person with frailty \(z\) has cumulative hazard \(z(B+W)\). If frailty at enrollment is gamma with mean 1 and variance \(\theta\), observed cumulative hazard is \(\log[1+\theta(B+W)]/\theta\). Ordinary gamma inversion therefore returns \(B+W\) exactly under the model.

The paper defines its wave factor \(F(t;\alpha)\) as the mean of \(z^\alpha\) among survivors and divides the excess after inversion by that factor. At \(\alpha=1\), this mean is \(1/[1+\theta(B+W)]\), which is below 1 once there has been depletion. For example, if \(\theta=2\), \(B=0.20\), and \(W=0.05\), inversion returns \(0.25\). The factor is \(2/3\). With an exact baseline reference, applying the written adjustment would turn the wave contribution of \(0.05\) into \(0.075\), and the total into \(0.275\).

The implementation explicitly returns the original inverted curve at \(\alpha=1\). Its factor away from 1 also uses a gamma distribution with mean 1 and time-varying variance, whereas the original survivor frailty under ordinary proportional selection has a falling mean and fixed gamma shape. The code's factor has the identity limit around 1 because the moment of order 1 of a mean-one distribution is 1. The manuscript and code therefore use different definitions of the factor. This example establishes the inconsistency in the written derivation. [Manuscript §§2.7.2–2.7.3](../../documentation/preprint/paper.md) · [Software factor and correction](../../code/KCOR.py)

## Why getting the identity case right is just the first check

With baseline hazard proportional to \(z\) and wave hazard proportional to \(z^\alpha\), survival weights the original frailty distribution by \(\exp[-zB-z^\alpha W]\). At \(\alpha\ne1\), this generally changes the distribution away from gamma. That change can persist into the next quiet window. The implemented adjustment changes the level left by the wave and preserves subsequent increments of the original inverted curve. It therefore needs validation against a wave that changes who survives. The separate mathematical question is whether dividing the accumulated excess by a factor at its endpoint recovers the intended excess when the factor changes week by week. Neither point follows from recovering \(\alpha\).

The existing synthetic \(\alpha\) test generates excess directly from the assumed factor with added noise, using a supplied depletion path. It checks whether the fitting procedure can recover \(\alpha\) under that design. The manuscript reports the Czech \(\alpha\) as unidentified and treats the correction as inactive for that inference. [Synthetic generator](../../test/alpha/code/estimate_alpha.py) · [Manuscript §§3.4 and 5.4](../../documentation/preprint/paper.md)

## First test to isolate the correction

Generate expected weekly deaths and survivors from the individual frailty model and a known wave. Give the correction the true \(\alpha\), baseline, and starting frailty to isolate the adjustment from fitting error. Compare the corrected cumulative path with the independently known frailty-neutral baseline plus wave. Use \(\alpha=1\) as the control, then test values above and below 1 and check both during and after the wave. If the module is intended as a descriptive sensitivity transformation, state that target explicitly. A claim to recover the underlying frailty-neutral path calls for this recovery check and then a test with fitted inputs.
