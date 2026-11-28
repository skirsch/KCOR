# Round 12 convergence - m161

Two independent blind GPT-6.2 Astra High reviews and one independent targeted Astra High review of m156 against m161. Original question: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" The proportional gamma inversion and fixed-slope identification within the working model remain sound. No new large refits or corrections adopted.

## Resolution of round 11

Fifteen resolved in original scope; three partial (11.1, 11.14, 11.15). See [targeted review](resolution.md). All six primary Table 8 percentile intervals reproduced from the archived 999 draws; S6 file unchanged. Full forced-alpha intervals remain from preceding seed implementation.

## Direct carryovers (three)

- **[12.1](../../../objections/12.1/README.md) - Post-2020 births excluded before all-ages reconstruction** (High for affected ordinary workbooks; A, B, R). Hard-coded 2020 cutoff removes valid births from reconstructed all-ages population but retains them in supplied totals. Three independent fixtures confirm divergent endpoints. Czech S6 eligibility ends in 2005, unaffected.
- **[12.2](../../../objections/12.2/README.md) - Identity wave branches miscount marked weeks and strict deficits** (Moderate for optional wave diagnostic; A, R). Alpha one returns zero marked weeks before counting; at zero frailty, cumulative attribution can mask a later weekly deficit. Three-week example has two strict deficits, not one zero-excess evaluation. Cumulative identity remains correct.
- **[12.3](../../../objections/12.3/README.md) - SI figure still labels finite deviation as asymptotic bias** (Minor; R). Embedded fig_si_kcor_bias_vs_theta.png and generator retain y-axis Bias in KCOR asymptote (KCOR minus 1), despite corrected caption.

## New or related specific findings (eight)

- **[12.4](../../../objections/12.4/README.md) - Protective first gap falsely localizes deficit to later ordinary gap** (High for per-gap assumption diagnostic; A). Exact signed offsets -0.046924 and 0 returned -0.101682 and -0.113628; second gap genuinely follows baseline. 50 matched person draws yield spurious second-gap interval [-0.12384,-0.10697]. Entry-rate seed clamps negative log multiplier before diagnostic baseline calculation.
- **[12.5](../../../objections/12.5/README.md) - Historical synthetic negative control misattributed to current gamma estimator** (High for validation claim; B). Figure source fits separate four-parameter log-hazard relaxation curves on entire follow-up of discrete mixture, not gamma quiet-window fitting or inversion. Replay reproduces endpoint 0.997533. Current SI S9 validation is separate.
- **[12.6](../../../objections/12.6/README.md) - Complete registry enrollment does not imply little frailty depletion** (Moderate scientific wording; B). SI S4.2.2 infers reduced depletion and weak variance identification from complete registry strata. Complete enumeration eliminates additional subsampling but not latent heterogeneity or selective deaths.
- **[12.7](../../../objections/12.7/README.md) - Undefined anchor differs between dedicated replay and ordinary pooling** (Moderate for affected replay histories; R). Moving first six weeks of 1940 dose-2 deaths to survivors yields ordinary available-strata points 1.209732 and 1.219862 with retained weight 0.704731, but dedicated NaNs. Previous exhaustion and sparse-weight fixtures remain fixed.
- **[12.8](../../../objections/12.8/README.md) - Tiny positive references accepted by pool but rejected by direct row** (Moderate numerical edge case; R). Valid one-stratum stress input with 10^13 people, one/two early deaths each week, pooled point 0.509146 but sole contributor and all-ages direct reference unavailable due 1e-12 threshold.
- **[12.9](../../../objections/12.9/README.md) - Pooled parameter status still copied from one stratum** (Minor output labeling; R). Pooled theta values now blank; neighboring status cells still copied from first finite contributor, not representative of all fits.
- **[12.10](../../../objections/12.10/README.md) - Cox caption attributes individual deviations solely to depletion** (Minor scientific wording; R). Equal zero-variance control has HR 0.988080 [0.968753,1.007793] nominal model CI despite no differential depletion.
- **[12.11](../../../objections/12.11/README.md) - Historical S7 mixture mass and initial risk disagree** (Minor historical provenance; numerical risk accounting if reused; R). Truncated quantile weights sum to .998 but initial opening risk uses nominal 2 million: 4,000 unexplained people in first-week accounting. Two skipped weeks exclude this denominator from fitted hazards.

## Interpretation

The core inversion is not newly challenged. The population-cutoff and gap-localization bugs affect particular inputs and diagnostics; the historical figure attribution affects the scientific validation claim. The published Czech numerical intervals were not shown incorrect by these tests. No large study was rerun.
