# KCOR Supplementary Information (SI)

```{=latex}
% Enforce Supplementary (S) numbering for figures and tables in the SI PDF.
\renewcommand{\thefigure}{S\arabic{figure}}%
\setcounter{figure}{0}%
\renewcommand{\thetable}{S\arabic{table}}%
\setcounter{table}{0}%
```

This document provides supplementary material supporting the KCOR methodology described in the main manuscript, including extended derivations, simulation studies, robustness analyses, and additional empirical results.

## S1. Overview

This SI is organized as follows:

- **S1**: Overview
- **S2**: Extended diagnostics and failure modes
- **S3**: Positive controls (injected harm/benefit)
- **S4**: Control-test specifications and simulation parameters
- **S5**: Additional figures and diagnostics
- **S6**: Extended Czech empirical application / illustrative registry analysis

## S2. Extended diagnostics and failure modes

This section describes the **observable diagnostics and failure modes** associated with the revised KCOR working assumptions and the corresponding identifiability criteria. KCOR is designed to **fail transparently rather than silently**: when an assumption is violated, the resulting lack of identifiability or model stress manifests through explicit diagnostic signals rather than spurious estimates. In particular, proportional hazard differences within quiet windows may not be distinguishable from frailty-induced curvature, and such cases are expected to trigger diagnostic instability or non-identifiability.

The KCOR framework separates **working assumptions**, **empirical diagnostics**, and **identifiability criteria**; these are summarized below in Tables @tbl:si_assumptions–@tbl:si_identifiability.

Table: KCOR working assumptions. {#tbl:si_assumptions}

| Assumption | Description | Role in KCOR |
|---|---|---|
| A1 Cohort stability | Cohorts are fixed at enrollment with no post-enrollment switching or informative censoring. | Ensures cumulative hazards are comparable over follow-up |
| A2 Shared external hazard environment | Cohorts experience the same background hazard over the comparison window, except for explicitly modeled extensions. | Prevents confounding by cohort-specific shocks |
| A3 Enrollment-time frailty | Selection operates through time-invariant latent heterogeneity summarized by cohort-specific enrollment-time frailty variance $\theta_{0,d}$, estimated separately for each cohort. Under minimal aggregated data, this interpretation is conditional on no constant multiplicative hazard effect operating inside the quiet-window identification regime. | Defines the parameter targeted by the estimator |
| A4 Adequacy of gamma frailty | Gamma frailty provides a reasonable approximation to observed depletion geometry. | Allows tractable inversion and normalization |
| A5 Adequacy of Gompertz baseline | A fixed Gompertz age slope is a reasonable working approximation over the estimation horizon. | Anchors the structured baseline used to identify $\theta_{0,d}$ |
| A6 Multi-window quiet-period validity | Prespecified quiet windows exist in which depletion can be identified without dominant external shocks. | Permits pooled identification across windows within each cohort |
| A7 Structured offset additivity | When the delta path is used, wave effects are additive in cumulative-hazard space and persist forward after the wave ends. | Permits alignment of quiet windows through $\delta_i$ and $\Delta(t)$ |
| A8 Optional NPH exponent model identifiability | When the optional NPH module is used, excess hazard is reasonably represented by a shared-amplitude frailty-dependent amplification model with a global exponent $\alpha$, and there is sufficient cross-cohort separation in depletion geometry to identify that exponent from relative structure. | Supports optional NPH preprocessing before inversion |

Table: Empirical diagnostics associated with KCOR assumptions. {#tbl:si_diagnostics}

| Diagnostic | Description | Observable signal |
|---|---|---|
| Skip-week sensitivity | Exclude early post-enrollment weeks subject to dynamic selection. | Stable fitted $\theta_{0,d}$ under varying skip weeks |
| Seed-fit quality | Evaluate the nearest-window Gompertz seed fit. | Small residuals and plausible $\hat{k}_d$ / $\hat{\theta}_{0,d}^{(0)}$ |
| Multi-window consistency | Refit after omitting or perturbing individual quiet windows. | Stable $\hat{\theta}_{0,d}$ and similar normalized trajectories |
| Delta plausibility | Check reconstructed offsets and failure flags. | Positive / coherent offsets or transparent fallback |
| Post-normalization linearity | Assess curvature removal in cumulative-hazard space. | Approximate linearity after normalization within quiet windows as a diagnostic check under the working model |
| Residual structure | Examine residuals in hazard space. | No systematic time-structure within or across windows |
| KCOR(t) stability | Inspect KCOR trajectories following anchoring. | Stabilization rather than drift, interpreted diagnostically rather than as proof |
| Optional NPH diagnostics | Compare pairwise and collapse estimators, theta propagation scales, excess-handling choices, and with/without-module results when relevant. | Estimator agreement, non-flat objectives, and limited dependence on reasonable NPH settings |

Empirical illustration of the "Quiet-window perturbation" diagnostic is provided in Figure @fig:si_quiet_window_theta_scan, which scans monthly-shifted 12-month windows in Czech registry data.

Table: Identifiability criteria governing KCOR interpretation. {#tbl:si_identifiability}

| Criterion | Condition | Consequence if violated |
|---|---|---|
| I1 Diagnostic sufficiency | All required diagnostics pass. | KCOR interpretable |
| I2 Curvature sufficiency | Mortality geometry contains enough curvature to distinguish $k_d$ from $\theta_{0,d}$, conditional on no constant multiplicative hazard effect being observationally confounded with that curvature in the quiet-window regime. | $\theta_{0,d}$ weakly identified |
| I3 Multi-window coherence | Quiet windows remain consistent after alignment. | Interpretation limited |
| I4 Delta applicability | Offset reconstruction is coherent when the delta path is used. | Fall back or treat as non-identifiable |
| I5 Anchoring validity | Post-normalization behavior is stable in the reference quiet window. | Anchoring invalid |
| I6 NPH signal sufficiency | When the optional NPH module is used, cross-cohort differences in depletion geometry and excess hazard are strong enough to identify $\alpha$ from relative structure. | $\alpha$ weakly identified; objectives flatten or estimators disagree |
| I7 Conservative failure rule | Any failure → diagnostics indicate non-identifiability. | Analysis treated as not identified; results not reported |

When diagnostics indicate non-identifiability, the analysis is treated as not identified and results are not reported; this does not invalidate the KCOR estimator itself. In minimal aggregated data, a constant multiplicative hazard effect within the quiet-window regime is observationally confounded with frailty-induced curvature over short horizons, so identifiability of $\theta_{0,d}$ remains conditional on the working model rather than assumption-free. When the optional NPH module is used, identifiability of $\alpha$ is likewise conditional: it depends on cross-cohort separation in depletion geometry and stable excess-hazard measurement rather than on any direct observation of the common external intensity.

### S2.1 Optional NPH exponent model: estimator and failure signatures

The optional NPH module extends KCOR only for prespecified periods in which an external hazard may interact with frailty in a non-proportional manner. Under the working model, excess hazard for cohort $d$ is represented as

$$
h_{\mathrm{excess},d}(t)=A(t)\,F_d(t;\alpha),
\qquad
F_d(t;\alpha)=E_d[z^{\alpha}\mid t],
$$

where $A(t)$ is an unknown common amplitude and the exponent $\alpha$ is treated as global across cohorts within the analyzed period. When $\alpha=1$, amplification is proportional to baseline risk and the NPH module is effectively inactive. The exponent $\alpha$ should be interpreted as a model-calibrated summary of frailty-dependent amplification under the working model, not as a uniquely identified biological or mechanistic constant.

Because $A(t)$ is common across cohorts, absolute excess hazards are not directly comparable. However, cross-cohort ratios eliminate the common amplitude:

$$
\frac{h_{\mathrm{excess},i}(t)}{h_{\mathrm{excess},j}(t)}
=
\frac{F_i(t;\alpha)}{F_j(t;\alpha)}.
$$

This invariance is the core identification logic for $\alpha$: the parameter is identified, if at all, from relative cohort structure rather than from absolute hazard levels. Identification therefore requires sufficient cross-cohort variation in depletion geometry and cumulative hazard. When cohorts are too similar, or when excess hazard is measured unreliably, the objective functions flatten and $\alpha$ becomes weakly identified.

The two operational estimators mirror the main text. The **pairwise estimator** minimizes

$$
\sum_{i<j,t}
\left[
\log e_i(t)-\log e_j(t)
-\left(\log F_i(t;\alpha)-\log F_j(t;\alpha)\right)
\right]^2,
$$

while the **collapse estimator** minimizes

$$
\operatorname{Var}_d\!\left[
\log e_d(t)-\log F_d(t;\alpha)
\right].
$$

Agreement between the two estimators is treated as supportive of identification under the working model; disagreement is diagnostic rather than something to be averaged away. Additional robustness checks vary the excess-hazard handling rule, baseline anchor choice, and theta propagation scale.

Failure of NPH identification manifests through observable signatures: flat objective curves, boundary-seeking estimates, strong dependence on arbitrary settings, or disagreement between the pairwise and collapse estimators. These patterns are treated as evidence of weak signal or model misspecification rather than as valid estimates of $\alpha$.

## S3. Positive controls

### S3.1 Construction of injected effects

The effect window is a simulation construct used solely for positive-control validation and does not represent a real-world intervention period or biological effect window.

Positive controls are constructed by starting from a negative-control dataset and injecting a known effect into the data-generating process for one cohort, for example by multiplying the *baseline* hazard by a constant factor $r$ over a prespecified interval:

$$
h_{0,\mathrm{treated}}(t) = r \cdot h_{0,\mathrm{control}}(t) \quad \text{for } t \in [t_1, t_2],
$$
{#eq:pos-control-injection}

with $r>1$ for harm and $0<r<1$ for benefit.

After gamma-frailty normalization (inversion), KCOR should deviate from 1 in the correct direction and with magnitude broadly consistent with the injected effect (up to discretization and sampling noise). Figure @fig:pos_control_injected and Table @tbl:pos_control_summary illustrate this behavior.

![Positive control validation: KCOR tracks injected effects in the expected direction. Left panels show harm scenario (r=1.2), right panels show benefit scenario (r=0.8). Top row displays cohort hazard curves with effect window shaded. Bottom row shows $\mathrm{KCOR}(t)$ deviating from 1.0 in the expected direction during the effect window. Uncertainty bands (95% bootstrap intervals; aggregated cohort--time resampling) are shown. X-axis units are weeks since enrollment.](figures/fig_pos_control_injected.png){#fig:pos_control_injected}

## S4. Control-test specifications and simulation parameters

### S4.0 Summary tables for control-test and simulation parameters

Table: Summary of control-test and simulation parameters referenced in Sections S4.2–S4.6. Values shown are representative defaults or grids used to probe the revised estimator; exact scripts and run settings are version-controlled in the repository. {#tbl:si_sim_params}

| Section | Item | Parameter | Value | Notes |
|---|---|---|---|---|
| S4.2.1 | Synthetic validation | Data source / scripts | Repository simulation scripts | Used for $\theta_0$ recovery and null-behavior checks |
| S4.2.1 | Synthetic validation | Frailty heterogeneity grid | Prespecified varying grid | Includes weak- and strong-curvature regimes |
| S4.2.1 | Synthetic validation | Baseline model in estimator | Gompertz with fixed $\gamma$ | Aligns with main-text estimator |
| S4.2.1 | Synthetic validation | Time horizon | Multi-year follow-up | Sufficient to probe identifiability |
| S4.2.2 | Empirical negative control | Data source | Czech admin registry data (KCOR_CMR) | Aggregated cohorts |
| S4.2.2 | Empirical negative control | Construction | Age strata remapped to pseudo-doses | Preserves a pseudo-null by design |
| S4.3 | Positive control | Effect multiplier | $r=1.2$ harm; $r=0.8$ benefit | Injected after the same preprocessing pipeline |
| S4.3 | Positive control | Frailty target | Enrollment-time $\theta_{0,d}$ | Estimated separately within each cohort, not fixed in the estimator |
| S4.4 | Sensitivity analysis | Perturbations | Skip weeks, quiet-window placement, omitted windows, anchor choice, excess handling, theta propagation scale, optional NPH settings | Probes robustness of the revised estimator |
| S4.5 | Adversarial selection geometry | Frailty distribution | Tail-mixture versus mid-quantile sampling | Tests robustness to non-gamma geometry |
| S4.6 | Joint frailty + effect | DGP | Frailty heterogeneity plus injected effect | Tests separability of depletion and effect windows |

### S4.1 Reference implementation and default operational settings

Table: Reference implementation and default operational settings. {#tbl:si_defaults}

| Component | Setting | Default value | Notes |
|---|---|---|---|
| Cohort construction | Cohort indexing | Enrollment period × YearOfBirth group × Dose; plus all-ages cohort (YearOfBirth = -2) | Implementation detail |
| Quiet-period selection | Quiet windows | Prespecified ISO-week intervals spanning follow-up | Used jointly after alignment, not one at a time |
| Early-period stabilization (dynamic HVE) | `SKIP_WEEKS` | 2 | Weeks $t < \mathrm{SKIP\_WEEKS}$ are excluded from accumulation; time is then rebased |
| Gompertz slope | `\gamma` | Fixed implementation value | Same value used throughout the estimator |
| Frailty estimation | Fit method | Seed Gompertz fit + delta iteration + pooled refit | Estimates $(k_d,\theta_{0,d})$ separately within each cohort in the main path |
| Anchoring | `NORMALIZATION_WEEKS` | 4 | Reference window for anchored KCOR plots |
| NPH module | Optional NPH exponent model | Off by default | Enabled only when cross-cohort NPH signal is plausible and diagnostics support identification of $\alpha$ |

### S4.2 Negative controls

Negative controls are used to evaluate the behavior of KCOR under settings where the true effect is known to be null, while allowing substantial heterogeneity in baseline risk and selection-induced depletion. Two complementary classes of negative controls are considered: (i) fully synthetic simulations used to assess both $\theta_0$ recovery and null KCOR behavior, and (ii) empirical registry-based constructions that preserve a pseudo-null by repurposing age strata as pseudo-exposures without selective sampling. Together, these controls assess whether KCOR remains stable in the presence of non-proportional hazards arising from selection rather than treatment.

#### S4.2.1 Synthetic negative control: gamma-frailty null

The synthetic negative control (Figure @fig:neg_control_synthetic) is a fully specified simulation designed to induce **strong selection-induced depletion curvature under a true null effect**. Under the revised estimator, this synthetic setting is used not only to ask whether KCOR remains near 1, but also whether the estimator recovers the cohort-specific enrollment-time frailty parameters governing the simulated geometry and aligns quiet windows coherently. Near-1 behavior here is interpreted as a diagnostic implication under the working model, not as stand-alone proof that the estimator is correct.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

Both cohorts share identical per-frailty-group death probabilities; only the mixture weights differ. This induces different cohort-level curvature under the null.

Figure @fig:si_kcor_bias_vs_theta provides a compact summary of KCOR bias as a function of frailty variance under the same synthetic-null grid used in Table @tbl:cox_bias_demo.

![Simulated-null summary: KCOR bias as a function of enrollment-time frailty variance $\theta_0$. Bias is defined as $\mathrm{KCOR}_{\mathrm{asymptote}} - 1$ at the end of follow-up in the synthetic-null grid (no treatment effect), reflecting cumulative deviation under the null rather than instantaneous hazard bias. This figure is a diagnostic summary under the working model, not proof that the model is true. Points show single-run estimates from the grid; no error bars are shown.](figures/fig_si_kcor_bias_vs_theta.png){#fig:si_kcor_bias_vs_theta}

![Synthetic negative control under strong selection (different curvature) but no effect: $\mathrm{KCOR}(t)$ stays near 1 under the synthetic null. Top panel shows cohort hazards with different frailty-mixture weights inducing different curvature. Bottom panel shows $\mathrm{KCOR}(t)$ remaining near 1.0 after normalization, which is the diagnostic behavior expected under the working model rather than proof that all confounding has been removed. Uncertainty bands (95% bootstrap intervals; aggregated cohort--time resampling) are shown.](figures/fig_neg_control_synthetic.png){#fig:neg_control_synthetic}

#### S4.2.2 Empirical negative control: age-shift construction

The empirical negative control (Figures @fig:neg_control_10yr and @fig:neg_control_20yr) repurposes registry cohorts to create a **pseudo-null comparison** while inducing large baseline hazard differences via 10–20 year age shifts. Because these are full-population strata rather than selectively sampled subcohorts, selection-induced depletion is reduced and the fitted $\theta_{0,d}$ values are expected to be small or weakly identified. The goal here is to test end-to-end KCOR behavior, not to use the empirical geometry as proof that all confounding has been removed.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

This construction ensures that dose comparisons are within the same underlying vaccination category, preserving a pseudo-null while inducing 10–20 year age differences.

This contrasts with the synthetic negative control (Section S4.2.1), where strong, deliberately induced frailty heterogeneity requires gamma-frailty normalization to recover the null.

### S4.3 Positive control: injected effect

Positive controls are used to verify that KCOR responds appropriately when a true effect is present. Starting from a negative-control simulation with no treatment effect, a known multiplicative hazard shift is injected into one cohort over a prespecified time window. This construction allows direct assessment of whether the same preprocessing and $\theta_{0,d}$ estimation pipeline detects both the direction and timing of the injected effect while remaining stable outside the effect window.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

The injection multiplies the treatment cohort's baseline hazard by factor $r$ during the effect window, while leaving the control cohort unchanged.

### S4.4 Sensitivity analysis parameters

Sensitivity analyses evaluate the robustness of KCOR estimates to reasonable variation in analysis choices that do not alter the underlying data-generating process. Skip weeks, quiet-window placement, omission of individual quiet windows, and optional NPH settings are perturbed over a prespecified range while holding all other parameters fixed. For the optional NPH module, this robustness menu includes baseline anchor choice, excess-hazard handling, theta propagation scale, and agreement between the pairwise and collapse estimators. These analyses assess whether KCOR behavior is stable to tuning choices that primarily affect identification rather than cohort composition.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

Output grids show KCOR(t) values for each parameter combination.

![Sensitivity analysis summary showing $\mathrm{KCOR}(t)$ values across parameter grid. Heatmaps display $\mathrm{KCOR}(t)$ estimates for different combinations of baseline weeks (rows) and quiet-window start offsets (columns). Across all comparisons, $\mathrm{KCOR}(t)$ varies smoothly and modestly across a wide range of quiet-start offsets and baseline window lengths, with no qualitative changes in sign or magnitude, indicating robustness to reasonable parameter choices. All panels use a unified color scale centered at 1.0 to enable direct visual comparison across dose comparisons.](figures/fig_sensitivity_overview.png){#fig:sensitivity_overview}

### S4.5 Tail-sampling / bimodal selection (adversarial selection geometry)

This adversarial simulation evaluates KCOR under extreme but controlled violations of typical cohort-selection geometry. Two cohorts are constructed to share identical mean frailty while differing sharply in how risk is distributed, using mid-quantile sampling versus a low/high-tail mixture. This setting stress-tests whether the revised $\theta_{0,d}$ estimator and depletion normalization remain effective when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.

- **Mid-sampled cohort**: frailty restricted to central quantiles (e.g., 25th–75th percentile) and renormalized to mean 1.
- **Tail-sampled cohort**: mixture of low and high tails (e.g., 0–15th and 85th–100th percentiles) with mixture weights chosen to yield mean 1.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

Both cohorts share the same baseline hazard $h_0(t)$ and have no treatment effect (negative-control version). Positive-control versions are also generated by applying a known hazard multiplier in a prespecified window. The evaluation includes (i) KCOR drift, (ii) fit quality under the revised estimator, (iii) post-normalization linearity, and (iv) parameter stability under window perturbation.

### S4.6 Joint frailty and treatment-effect simulation

This simulation evaluates KCOR under conditions in which **both selection-induced depletion (frailty heterogeneity)** and a **true treatment effect (harm or benefit)** are present simultaneously. The purpose is to assess whether KCOR can (i) correctly identify and neutralize frailty-driven curvature using aligned quiet windows and (ii) detect a true treatment effect outside that period without confounding the two mechanisms.

This joint simulation combines the selection-induced depletion mechanisms examined in Sections S4.2 and S4.5 with the injected-effect framework of Section S4.3.

Parameter values and scripts for this joint simulation are summarized in Table @tbl:si_sim_params.

#### Design

Two fixed cohorts are generated with identical baseline hazards but differing enrollment-time frailty variance. Individual hazards are multiplicatively scaled by a latent frailty term drawn from a gamma distribution with unit mean and cohort-specific variance. A treatment effect is then injected over a prespecified time window that does not overlap the quiet windows used for frailty identification.

Formally, individual hazards are generated as

$$
h_i(t) = z_i \cdot h_0(t) \cdot r(t).
$$
{#eq:si_individual_hazard_with_effect}

where $z_i$ is individual frailty, $h_0(t)$ is a shared baseline hazard, and $r(t)$ is a time-localized multiplicative treatment effect applied to one cohort only.

## S5. Additional figures and diagnostics

This section provides diagnostic outputs and evaluation criteria for the simulations and control-test specifications defined in Section S4.

### S5.1 Fit diagnostics

For each cohort $d$, the revised fit produces diagnostic outputs including:

- **RMSE in hazard space**: Root mean squared error between observed and model-predicted hazards over the pooled quiet-window fit set.
- **Fitted parameters**: baseline scale parameter and cohort-specific enrollment-time frailty variance $\theta_{0,d}$.
- **Number of fit points**: pooled observations across quiet windows.
- **Offset diagnostics**: number, sign, and stability of reconstructed $\delta_i$ terms when the delta path is used.

When uncertainty bands or bootstrap intervals are reported in this supplement, they are computed using an aggregated-data bootstrap at the cohort--time level (resampling event counts and risk-set sizes within time bins/strata), not by resampling individuals.

Example diagnostic output from the reference implementation:

```
KCOR_FIT,EnrollmentDate=2021_24,YoB=1950,Dose=0,
  k_hat=4.29e-03,theta0_hat=8.02e-01,
  RMSE_h=3.37e-03,n_obs=97,delta_count=3,success=1
```

### S5.2 Residual analysis

Fit residuals should be examined in hazard space. Define residuals:

$$
r_{d}(t)=h_d^{\mathrm{adj}}(t)-\frac{\hat{k}_d e^{\gamma t_{\mathrm{rebased}}}}{1+\hat{\theta}_{0,d}\left(H_{\mathrm{gom},d}(t_{\mathrm{rebased}})+\Delta_d(t)\right)}.
$$

- **Systematic patterns**: Residuals should be approximately random around zero. Systematic structure suggests model inadequacy.
- **Outliers**: Individual weeks with large residuals may indicate data quality issues or external shocks.
- **Autocorrelation**: Strong autocorrelation in residuals suggests the model is missing time-varying structure.

### S5.3 Parameter stability checks

Robustness of fitted parameters is assessed by:

- **Quiet-window perturbation**: Shift the quiet-window start/end by ±4 weeks and re-fit. Stable parameters should vary modestly and preserve the same qualitative interpretation.
- **Window omission**: Drop one quiet window at a time and verify that $\hat{\theta}_{0,d}$ and KCOR remain coherent.
- **Skip-weeks sensitivity**: Vary `SKIP_WEEKS` and verify KCOR trajectories remain qualitatively similar.
- **Gompertz sensitivity**: Check whether modest perturbations of $\gamma$ materially alter interpretation.
- **Optional NPH sensitivity**: In epidemic-wave applications, compare results with and without the module, across plausible anchor and excess-handling choices, across theta propagation scales, and across the pairwise and collapse estimators.

### S5.4 Quiet-window perturbation scan (empirical Czech data)

To assess whether diagnostically valid quiet windows are rare or fragile in applied registry data, we performed a systematic scan of quiet-window placements in the Czech Republic 2021–2024 cohort (enrollment ISO week 2021-24). We re-estimated the revised frailty parameters $(\hat{k}_d,\hat{\theta}_{0,d})$ across a sequence of overlapping quiet windows of fixed duration, holding all other settings fixed.

Specifically, we considered 12-month windows beginning at successive monthly offsets between April 2022 and April 2023 (inclusive). For each window placement, parameters were estimated using the same revised estimator and diagnostic pass/fail criteria used elsewhere in this SI. Results are shown for two birth-decade strata (1940s and 1950s), and for dose groups $d \in \{0,1,2\}$.

Figure @fig:si_quiet_window_theta_scan shows fitted $\hat{\theta}_{0,d}$ as a function of quiet-window midpoint date. The purpose of this scan is technical rather than substantive: it asks whether the estimator remains coherent across nearby quiet-window choices and whether failures are surfaced explicitly. Open markers denote windows failing diagnostics (treated as non-identifiable and not interpreted).

![Quiet-window robustness scan: fitted enrollment-time frailty variance $\hat{\theta}_{0,d}$ vs. quiet-window midpoint date for the Czech 2021–2024 enrollment cohort, using 12-month windows shifted monthly from April 2022 through April 2023. Marker shape denotes birth decade (1940s, 1950s). Filled markers indicate diagnostic pass; open markers indicate diagnostic failure. The figure is intended as a technical stability scan of the estimator rather than as substantive evidence for any one cohort narrative.](figures/fig_si_quiet_window_theta_scan_czech_2021_24.png){#fig:si_quiet_window_theta_scan}

### S5.5 Quiet-window overlay plots

Overlaying prespecified quiet windows on hazard and cumulative-hazard time series plots provides a visual diagnostic of window placement relative to mortality dynamics. The fit windows should:

- Avoid major epidemic waves or external mortality shocks
- Contain sufficient event counts for stable estimation
- Be interpretable after rebasing and, when relevant, after offset alignment

Visual inspection of quiet-window placement relative to mortality dynamics remains an essential diagnostic step, but it should be interpreted jointly with the formal multi-window and offset diagnostics described above.

### S5.6 Robustness to age stratification

This subsection examines $\mathrm{KCOR}(t)$ under narrow age stratification by repeating the same fixed-cohort comparison in three single birth-year cohorts spanning advanced ages (1930, 1940, 1950). Across these strata, the trajectories remain qualitatively stable after depletion normalization, which is consistent with the observed behavior not being driven solely by age aggregation.

![Birth-year cohort 1930: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and evaluated over calendar year 2023. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. X-axis units are weeks since enrollment.](figures/supplement/kcor_realdata_yob1930_enroll2022w26_eval2023.png){#fig:si_yob1930}

![Birth-year cohort 1940: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and evaluated over calendar year 2023. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. X-axis units are weeks since enrollment.](figures/supplement/kcor_realdata_yob1940_enroll2022w26_eval2023.png){#fig:si_yob1940}

![Birth-year cohort 1950: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and evaluated over calendar year 2023. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. X-axis units are weeks since enrollment.](figures/supplement/kcor_realdata_yob1950_enroll2022w26_eval2023.png){#fig:si_yob1950}

**Additional empirical negative-control variant (20-year age shift).**  
For completeness, the more extreme 20-year age-shift negative control referenced in the main text is included:

![Empirical negative control with approximately 20-year age difference between cohorts. Even under extreme composition differences, $\mathrm{KCOR}(t)$ shows little systematic drift, which is consistent with the estimator handling selection-induced curvature in this pseudo-null construction. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). Uncertainty bands (95% bootstrap intervals) are shown. Data source: Czech Republic mortality and vaccination dataset processed into KCOR_CMR aggregated format (negative-control construction; see Supplementary Information, SI). X-axis units are weeks since enrollment.](figures/fig3_neg_control_20yr_age_diff.png){#fig:neg_control_20yr}

## S6. Extended Czech empirical application

### S6.1 Empirical application with diagnostic validation: Czech Republic national registry mortality data

The Czech results do not validate KCOR; they represent an application that satisfies all pre-specified diagnostic criteria. Substantive implications follow only if the identification assumptions hold. Throughout this subsection, observed divergences are interpreted strictly as properties of the estimator under real-world selection, not as intervention effects.

Unless otherwise noted, KCOR curves in the Czech analyses are shown anchored at $t_0 = 4$ weeks for interpretability.

#### S6.1.1 Illustrative empirical context: COVID-19 mortality data

The COVID-19 vaccination period provides a natural empirical regime characterized by strong selection heterogeneity and non-proportional hazards, making it a useful illustration for the KCOR framework. During this period, vaccine uptake was voluntary, rapidly time-varying, and correlated with baseline health status, creating clear examples of selection-induced non-proportional hazards. The Czech Republic national mortality registry data exemplify this regime: voluntary uptake led to asymmetric selection at enrollment and to visibly different cohort hazard geometry over follow-up. Under the revised estimator, those differences are summarized through fitted $\theta_{0,d}$ values, aligned quiet-window behavior, depletion-normalized cumulative trajectories, and, when invoked, optional NPH diagnostics for the shared amplification exponent $\alpha$. The Czech application remains illustrative rather than validating; any NPH output should be interpreted as descriptive and model-calibrated rather than as direct biological evidence. While these examples illustrate KCOR's application, the method is general and applies to any retrospective cohort comparison where selection induces differential depletion dynamics.

#### S6.1.2 Descriptive frailty-normalization behavior in the Czech application

Across examined age strata in the Czech Republic mortality dataset, fitted enrollment-time frailty parameters vary across cohorts, with some estimates close to the weak-identification boundary and others appreciably above zero:

$$
\hat{\theta}_{0,d} \ge 0.
$$
{#eq:si_theta_near_zero}

These values should be interpreted as **descriptive outputs of the revised estimator**, not as validation evidence. They summarize how much depletion curvature the estimator attributes to each cohort at rebased enrollment time under the working model, conditional on the diagnostics for that analysis.

As a consequence, KCOR normalization leaves some cohorts' cumulative hazards nearly unchanged, while substantially increasing the depletion-neutralized baseline cumulative hazard for others. This behavior is consistent with curvature-driven normalization rather than cohort identity. The pattern is visible directly in depletion-neutralized versus observed cumulative hazard plots and is summarized quantitatively in the fitted-parameter logs (see `KCOR_summary.log`).

After frailty normalization, the depletion-neutralized baseline cumulative hazards are approximately linear in event time within diagnostically acceptable quiet windows. Residual deviations from linearity reflect real time-varying risk—such as seasonality or epidemic waves—rather than selection-induced depletion. This linearization is a diagnostic consistency check under the working model; persistent nonlinearity or parameter instability indicates model stress or quiet-window contamination.

Table @tbl:si_diagnostic_gate summarizes these diagnostic checks across age strata.

Table: Diagnostic gate for Czech application: KCOR results reported only where diagnostics pass. {#tbl:si_diagnostic_gate}

| Age band (years) | Quiet window valid | Post-normalization linearity | Parameter stability | KCOR reported |
| ---------------- | ------------------ | ---------------------------- | ------------------- | ------------- |
| 40–49            | Yes                | Yes                          | Yes                 | Yes           |
| 50–59            | Yes                | Yes                          | Yes                 | Yes           |
| 60–69            | Yes                | Yes                          | Yes                 | Yes           |
| 70–79            | Yes                | Yes                          | Yes                 | Yes           |
| 80–89            | Yes                | Yes                          | Yes                 | Yes           |
| 90–99            | Yes                | Yes                          | Yes                 | Yes           |
| All ages         | Yes                | Yes                          | Yes                 | Yes           |

In this application, all examined age strata satisfied the prespecified diagnostic criteria; this indicates internal consistency of the working model in this dataset but does not constitute validation of the estimator. KCOR results are not reported for any age stratum where diagnostics indicate non-identifiability.

**Interpretation:** In this application, some cohorts exhibit larger fitted $\hat{\theta}_{0,d}$ than others, indicating stronger estimated depletion curvature at rebased enrollment time under the working model:

$$
\hat{\theta}_{0,d} > 0.
$$
{#eq:si_theta_positive}

for some cohorts, while others remain close to the weak-identification boundary:

$$
\hat{\theta}_{0,d} \approx 0.
$$
{#eq:si_theta_near_zero_dose2}

Estimated heterogeneity can appear larger at younger ages because baseline hazards are low, so proportional differences across latent risk strata translate into visibly different short-term hazards before depletion compresses the risk distribution. At older ages, higher baseline hazard and stronger ongoing depletion can reduce the apparent dispersion of remaining risk, yielding smaller fitted $\theta_{0,d}$ even if latent heterogeneity is not literally smaller. These descriptive patterns should not be elevated into a validation criterion, and no single dose-ordering pattern is required by the revised estimator. Table @tbl:si_frailty_variance is therefore presented as an empirical summary of fitted parameters rather than as proof that the estimator is correct.

These raw contrasts reflect both selection and depletion effects and are not interpreted causally.

Table: Estimated enrollment-time gamma-frailty variance by age band and vaccination status for Czech cohorts enrolled in 2021_24. {#tbl:si_frailty_variance}

| Age band (years) | Fitted frailty variance (Dose 0) | Fitted frailty variance (Dose 2) |
| ---------------- | -----------------: | -----------------: |
| 40–49            |               16.79 |           $2.66 \times 10^{-6}$ |
| 50–59            |               23.02 |           $1.87 \times 10^{-4}$ |
| 60–69            |               13.13 |           $7.01 \times 10^{-18}$ |
| 70–79            |                6.98 |           $3.46 \times 10^{-17}$ |
| 80–89            |                2.97 |           $2.03 \times 10^{-11}$ |
| 90–99            |                0.80 |           $8.66 \times 10^{-16}$ |
| All ages (full population) |                4.98 |           $1.02 \times 10^{-11}$ |

**Notes:**
- The fitted frailty variance quantifies unobserved frailty heterogeneity and selection-induced depletion within cohorts at rebased enrollment time. Near-zero values indicate weak identifiability or effectively linear cumulative hazards under the working model.
- Each entry reports a single fitted enrollment-time gamma-frailty variance for the specified age band and vaccination status within the 2021_24 enrollment cohort.
- The "All ages (full population)" row corresponds to an independent fit over the full pooled age range, included as a global diagnostic.
- Table @tbl:si_raw_hazards reports raw outcome contrasts for ages 40+ (YOB $\le 1980$) where event counts are stable.

**Diagnostic checks:**
- **Interpretation is conditional on diagnostics:** these values are shown only for strata passing the prespecified diagnostic gate above.
- **Weak-identification boundary:** estimates near zero are treated as low-curvature or weak-identification outputs, not as substantive evidence for or against any cohort story.
- **Age coherence remains descriptive:** variation across age bands is reported descriptively and may reflect the interaction of baseline hazard, follow-up length, and depletion geometry.
- **Stability:** sign reversals, boundary pathologies, or major sensitivity to window perturbation would count against interpretability.
- **Falsifiability:** failure of any one of these checks would constitute evidence against model adequacy for that application.

Table: Ratio of observed cumulative mortality hazards for unvaccinated (Dose 0) versus fully vaccinated (Dose 2) Czech cohorts enrolled in 2021_24. (Note: the all-ages row reflects aggregation effects and is not directly comparable to age-stratified rows.) {#tbl:si_raw_hazards}

| Age band (years) | Dose 0 cumulative hazard | Dose 2 cumulative hazard | Ratio |
| ---------------- | ----------------------: | -----------------------: | ----: |
| 40–49            |               0.005260 |                0.004117 | 1.2776 |
| 50–59            |               0.014969 |                0.009582 | 1.5622 |
| 60–69            |               0.045475 |                0.023136 | 1.9655 |
| 70–79            |               0.123097 |                0.057675 | 2.1343 |
| 80–89            |               0.307169 |                0.167345 | 1.8355 |
| 90–99            |               0.776341 |                0.517284 | 1.5008 |
| All ages (full population) |               0.023160 |                0.073323 | 0.3159 |

This table reports unadjusted cumulative hazards derived directly from the raw data, prior to any frailty normalization or depletion correction, and is shown to illustrate the magnitude and direction of selection-induced curvature addressed by KCOR.

Values reflect raw cumulative outcome differences prior to KCOR normalization and are not interpreted causally due to cohort non-exchangeability. Cumulative hazards were integrated from cohort enrollment through the end of available follow-up for the 2021_24 enrollment window (through week 2024-16), identically for Dose 0 and Dose 2 cohorts.

#### S6.1.3 Illustrative application to national registry mortality data

A brief illustrative application is included to demonstrate end-to-end KCOR behavior on real registry mortality data in a setting that minimizes timing-driven shocks and window-tuning sensitivity. Cohorts were enrolled in ISO week 2022-26, and evaluation was restricted to calendar year 2023, yielding a 26-week post-enrollment buffer before slope estimation and a prespecified assessment period. Frailty parameters were estimated using the revised quiet-window settings and diagnostics described above to minimize wave-related hazard variation. This example is intended to illustrate estimator behavior under real-world selection and heterogeneity and does not support causal inference.

Figure @fig:si_allages shows $\mathrm{KCOR}(t)$ trajectories for dose 2 and dose 3 relative to dose 0 for an all-ages analysis. An all-ages analysis is presented as a high-heterogeneity stress test, since aggregation across age induces substantial baseline hazard and frailty variation.

![All-ages stress test: $\mathrm{KCOR}(t)$ trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and evaluated over calendar year 2023. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior under extreme heterogeneity and does not support causal inference. X-axis units are weeks since enrollment.](figures/kcor_realdata_allages_enroll2022w26_eval2023.png){#fig:si_allages}

## S7. Computational environment and runtime notes

**Environment.** Python 3.11; key dependencies include numpy, scipy, pandas, and lifelines (for Cox-model comparisons), with plotting via matplotlib.

**Compute requirements.** The full simulation grid reproduces in approximately 1 hour 26 minutes on a 20-core CPU with 128 GB RAM; smaller subsets reproduce in minutes.

**Reproduction.** Running `make paper` (or the repository's top-level build command) regenerates all artifacts from a clean checkout.

