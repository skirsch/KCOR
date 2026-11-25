# Exact error in the gamma correction after a nonlinear wave

With known starting frailty and a specified wave, the error in the baseline hazard after the wave can be calculated without simulating deaths or fitting parameters. It has a closed form for a quadratic wave (\(\alpha=2\)) and a simple approximation for weak waves at other exponents.

## Exact error for a specified wave

Let \(Z\) have gamma shape and rate \(a=1/\theta\), so its initial mean is one and its variance is \(\theta>0\). Let the individual hazard be \(Zb(t)+Z^\alpha A(t)\), where \(b(t)\) is baseline hazard and \(A(t)\) is the nonnegative wave intensity. Write \(B=\int_0^t b(s)\,ds\) and \(W=\int_0^t A(s)\,ds\). After the wave, \(W\) is fixed. Within this model the survivor mix at a given \(B\) depends on the total \(W\), regardless of how that total accumulated.

\[
S(B,W)=E\!\left[e^{-BZ-WZ^\alpha}\right],\qquad
m(B,W)=\frac{E\!\left[Z e^{-BZ-WZ^\alpha}\right]}{S(B,W)}.
\]

Here \(S\) is the fraction surviving and \(m\) is their true mean frailty. After the wave, the observed hazard is \(bm\). Ordinary gamma inversion of the observed cumulative hazard \(-\log S\) gives \(H_\Gamma=(S^{-\theta}-1)/\theta\). Differentiating with \(W\) fixed gives \(\widehat b=bm/S^\theta\). The factor \(S^\theta\) is the mean frailty implied by a gamma model with the same observed survival.

\[
\boxed{\frac{\widehat b-b}{b}=\frac{m(B,W)}{S(B,W)^\theta}-1.}
\]

For a full quiet week with true baseline cumulative hazard increment \(\Delta B>0\), the exact relative error is

\[
\frac{S(B+\Delta B,W)^{-\theta}-S(B,W)^{-\theta}}
     {\theta\,\Delta B}-1.
\]

The optional KCOR correction adds a fixed offset after the wave. That offset cancels when subtracting cumulative hazards at two times on the shifted curve. The error above therefore applies to its post-wave increments as well. It measures error in one cohort's increment with true starting frailty. Error in a ratio of cohorts also depends on both cohorts' errors and cumulative levels, and a real fit can change the estimated frailty. [Manuscript §2.7.3](../../../documentation/preprint/paper.md)

## Closed form for a quadratic wave

For \(\alpha=2\) and \(W>0\), let \(x=(a+B)/\sqrt{2W}\). Substituting \(u=\sqrt{2W}z\) in the gamma integral gives

\[
S=\left(\frac{a}{\sqrt{2W}}\right)^a e^{x^2/4}D_{-a}(x),\qquad
m=\frac{a}{\sqrt{2W}}\frac{D_{-a-1}(x)}{D_{-a}(x)},
\]

where \(D_\nu\) is the parabolic cylinder function. These expressions give the closed form for any starting gamma variance. [NIST integral representation](https://dlmf.nist.gov/12.5.E1) · [NIST notation for \(D_\nu\)](https://dlmf.nist.gov/12.2.E5)

For the simpler case \(\theta=1\), the same survival formula reduces to

\[
S=\frac{\sqrt{\pi}}{2\sqrt{W}}\operatorname{erfcx}
  \left(\frac{1+B}{2\sqrt{W}}\right),\qquad
m=\frac{1/S-(1+B)}{2W}.
\]

Here \(\operatorname{erfcx}(x)=e^{x^2}\operatorname{erfc}(x)\). The mean formula follows by integrating the derivative of \(e^{-(1+B)z-Wz^2}\). In the constructed example \(\theta=1\), \(B=0.20\), \(W=0.05\), with a true next-week increment of \(0.001\), the corrected week is \(0.000949484\), an error of about **−5.05%**. Under a proportional wave with the same inputs (\(\alpha=1\)), the exact result is \(0.001\). At \(\alpha=1.18\), direct integration gives \(0.000992588\), an error of about **−0.74%** for these same constructed inputs. \(W\) is a cumulative hazard for a person with \(Z=1\).

## Direction of the error and a simple approximation

For any positive wave, the sign of the error in this benchmark follows from matching the gamma model to the observed survival. Set \(r=a+B\) and \(R=aS^{-\theta}\). The gamma density that ordinary inversion implies for survivors has rate \(R>r\), mean \(S^\theta\), and the same normalizing mass \(S\) as the actual survivor density when both are formed from the original gamma density. The ratio of the actual survivor density to the matched gamma density is

\[
\exp[(R-r)z-Wz^\alpha].
\]

For \(\alpha>1\), this ratio is above 1 at low positive frailty and below 1 at high frailty, with one crossing. Because both densities have total mass one, the actual mean is smaller. Thus the ordinary inversion **underestimates** the true post-wave baseline increment. The direction reverses for \(0\le\alpha<1\). At \(\alpha=1\), the two densities coincide.

For weak waves, let \(M_\alpha(B)=\Gamma(a+\alpha)/[\Gamma(a)(a+B)^\alpha]\), the moment of order \(\alpha\) among survivors at baseline exposure \(B\) in the absence of a wave. Expanding survival and the mean to first order in \(W\) gives

\[
\frac{\widehat b-b}{b}=(1-\alpha)\,\theta\,W\,M_\alpha(B)+O(W^2).
\]

To see the coefficient, under the baseline-only survivor distribution \(S/S(B,0)=1-WM_\alpha+O(W^2)\) and the change in mean frailty is \(-W\operatorname{Cov}(Z,Z^\alpha)+O(W^2)\). The covariance is \(\alpha M_\alpha/(a+B)\). These expansions give the formula above.

[check.py](check.py) compares the closed form with direct integration, checks the weak-wave expansion and the sign rule, and reproduces the worked example. The example and derivation are specified model results. An applied error estimate would require defensible inputs for the cohort's baseline, starting frailty, wave exposure, and exponent.
