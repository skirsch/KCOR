# Objection 2: optional epidemic-wave correction

The original item 2 in [the independent review](../KCOR_Independent_Review_EN.md) questions the derivation of the optional wave correction, including the relationships between frailty moments, hazard amplification, and the final adjustment.

Status: **proposed resolution worked out, adoption pending.** The [initial wave audit](INITIAL_WAVE_AUDIT.md) documents the mismatch between the manuscript's wave factor and the software's behavior at \(\alpha=1\). The [post-wave error derivation](post_wave_error/README.md) calculates the exact error in the quiet weeks after a nonlinear wave with known inputs.

## Proposed resolution

Use a standalone correction when the optional non-proportional hazard (NPH) wave analysis is active. Assume gamma frailty \(Z\) once at the start of modeled follow-up. With a common exponent \(\alpha\) across waves, use the exact survival equation for each cohort:

\[
S(t)=E\!\left[\exp\!\left(-ZB(t)-Z^\alpha\sum_i W_i(t)\right)\right].
\]

Here \(B(t)\) is cumulative baseline hazard and \(W_i(t)\) is the exposure accumulated through time \(t\) from wave \(i\). During quiet weeks, earlier wave exposures stay fixed and continue to affect who survives. Given starting frailty, \(\alpha\), and cumulative wave exposure, solve for \(B(t)\) at each weekly boundary. Successive differences give corrected weekly increases. The paper should state what wave contribution the adjusted cumulative total retains before it forms the ratio and applies any reference anchor.

The equations are exact conditional on the specified model and inputs. General exponents call for deterministic numerical integration and a one-variable solve. The proposal starts after \(\alpha\) is obtained by the paper's existing procedure and takes absolute cumulative wave exposure as an input. Verification can use exact expected counts and known inputs. At \(\alpha=1\), the model including the wave reduces to ordinary gamma inversion. For one wave versus several waves with the same \(\alpha\), the final result should match when each cohort's total baseline hazard and wave exposure match and the reference value is the same. [Manuscript §2.7.2](../../documentation/preprint/paper.md)

For uncertainty intervals, apply the same correction to the original data and to each existing Monte Carlo resample before computing its KCOR ratio. The spread of corrected ratios propagates sampling variation through the correction. When \(\alpha\) and wave exposure are held fixed, the interval is conditional on those inputs. The deterministic verification and interval calibration answer separate questions.

