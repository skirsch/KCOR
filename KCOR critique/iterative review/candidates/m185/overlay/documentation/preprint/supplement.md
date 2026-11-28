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

This section describes the **observable diagnostics and failure modes** associated with the revised KCOR working assumptions and the corresponding identifiability criteria. Residual structure, unstable fits, or disagreement across quiet windows can reveal a violation. A disturbance that produces a trajectory consistent with the fitted model can pass all of these checks.

The KCOR framework separates **working assumptions**, **empirical diagnostics**, and **identifiability criteria**; these are summarized below in Tables @tbl:si_assumptions–@tbl:si_identifiability.

Table: KCOR working assumptions. {#tbl:si_assumptions}

| Assumption | Description | Role in KCOR |
|---|---|---|
| A1 Cohort stability | Cohorts are fixed at enrollment with no post-enrollment switching or informative censoring. | Ensures cumulative hazards are comparable over follow-up |
| A2 Shared external hazard environment | Cohorts experience the same background hazard over the comparison window, except for explicitly modeled extensions. | Prevents confounding by cohort-specific shocks |
| A3 Enrollment-time frailty | Selection operates through time-invariant latent heterogeneity summarized by cohort-specific enrollment-time frailty variance $\theta_{0,d}$, estimated separately for each cohort. Under minimal aggregated data, $\theta_{0,d}$ is interpreted via curvature in cumulative-hazard space; constant multiplicative effects are not separately identifiable from baseline scale. Multiplying individual baseline hazards by a constant preserves the frailty variance. Multiplying observed cohort hazards by a constant changes the fitted variance (main text §2.4.4). | Defines the parameter targeted by the estimator |
| A4 Adequacy of gamma frailty | Gamma frailty provides a reasonable approximation to observed depletion geometry. | Allows tractable inversion and normalization |
| A5 Adequacy of Gompertz baseline | A fixed Gompertz age slope is a reasonable working approximation over the estimation horizon. | Anchors the structured baseline used to identify $\theta_{0,d}$ |
| A6 Multi-window quiet-period validity | Prespecified quiet windows exist in which depletion can be identified without dominant external shocks. | Permits pooled identification across windows within each cohort |
| A7 Structured offset additivity | When the delta path is used, wave effects are additive in cumulative-hazard space and persist forward after the wave ends. | Permits alignment of quiet windows through $\delta_i$ and $\Delta(t)$ |
| A8 Optional NPH exponent model identifiability | When the optional NPH module is used, excess hazard is reasonably represented by a shared-amplitude frailty-dependent amplification model with a global exponent $\alpha$; $\alpha$ is localized using cross-cohort structure together with **externally supplied VE or other prespecified intervention-effect scale** not learned from the same wave-period contrasts; and the specified survivor law is used for fitting and inversion (main text §2.5; §2.7). | Supports optional NPH correction |

Table: Empirical diagnostics associated with KCOR assumptions. {#tbl:si_diagnostics}

| Diagnostic | Description | Observable signal |
|---|---|---|
| Skip-week sensitivity | Exclude early post-enrollment weeks subject to dynamic selection. | Stable fitted $\theta_{0,d}$ under varying skip weeks |
| Seed-fit quality | Evaluate the nearest-window Gompertz seed fit. | Small residuals and plausible $\hat{k}_d$ / $\hat{\theta}_{0,d}^{(0)}$ |
| Multi-window consistency | Refit after omitting or perturbing individual quiet windows. | Stable $\hat{\theta}_{0,d}$ and similar normalized trajectories |
| Delta plausibility | Check reconstructed offsets and failure flags. | Positive / coherent offsets or transparent fallback |
| Post-normalization linearity | Compare the normalized cumulative hazard with its fitted Gompertz curve. | Agreement with the fitted shape after allowing for the window offset |
| Residual structure | Examine residuals in hazard space. | No systematic time-structure within or across windows |
| KCOR(t) stability | Inspect KCOR trajectories following anchoring. | Stabilization rather than drift, interpreted diagnostically rather than as proof |
| Optional NPH diagnostics | Compare pairwise and collapse estimators, theta propagation scales, excess-handling choices, NPH correction diagnostics, and with/without-module results when relevant. | Estimator agreement, non-flat objectives, and limited dependence on reasonable NPH settings |

Empirical illustration of the "Quiet-window perturbation" diagnostic is provided in Figure @fig:si_quiet_window_theta_scan, which scans monthly-shifted 12-month windows in Czech registry data.

Table: Identifiability criteria governing KCOR interpretation. {#tbl:si_identifiability}

| Criterion | Condition | Consequence if violated |
|---|---|---|
| I1 Diagnostic sufficiency | All required diagnostics pass. | Substantive interpretability not established |
| I2 Information in quiet windows | The observed hazard trajectory distinguishes the baseline rate from the frailty variance with sufficient precision under the specified Gompertz model. | $\theta_{0,d}$ weakly identified |
| I3 Multi-window coherence | Quiet windows remain consistent after alignment. | Interpretation limited |
| I4 Delta applicability | Offset reconstruction is coherent when the delta path is used. | Fall back or treat as non-identifiable |
| I5 Anchoring validity | Post-normalization behavior is stable in the reference quiet window. | Anchoring invalid |
| I6 NPH signal sufficiency | When the optional NPH module is used, cross-cohort differences in depletion geometry and excess hazard are strong enough to localize $\alpha$ jointly with a prespecified **externally supplied VE or other prespecified intervention-effect scale**. | $\alpha$ weakly identified; objectives flatten or estimators disagree |
| I7 Conservative failure rule | Required identification diagnostics are satisfied. | Substantive interpretability not established; flagged outputs may be shown for audit |

When diagnostics indicate non-identifiability, substantive interpretation is withheld; explicitly flagged numerical outputs may be retained for descriptive illustration or audit; this does not invalidate the KCOR estimator itself. In minimal aggregated data, identifiability of $\theta_{0,d}$ from curvature remains conditional on the working model rather than assumption-free: additive or sharply time-varying components can distort curvature, and heterogeneous multiplicative effects across cohorts can perturb depletion paths, with diagnostics intended to surface these failure modes. When the proposed optional NPH localization procedure is used, identifiability of $\alpha$ is likewise conditional: it depends on cross-cohort separation in depletion geometry, stable excess-hazard measurement, and a prespecified **externally supplied VE or other prespecified intervention-effect scale** (main text §2.7.2); the procedure does not jointly identify $\alpha$ and cohort-specific multiplicative effects if that scale is not fixed externally (main text §5.4). The common external intensity is not directly observed.

### S2.1 Optional NPH exponent model: estimator and failure signatures

The optional NPH module extends KCOR only for prespecified periods in which an external hazard may interact with frailty in a non-proportional manner. Under the working model, the instantaneous wave component for cohort $d$, conditional on actual survivors, is represented as

$$
h_{\mathrm{excess},d}(t)=A(t)\,F_d(t;\alpha),
\qquad
F_d(t;\alpha)=E_d[z^{\alpha}\mid t],
$$

where $A(t)$ is an unknown common amplitude and the exponent $\alpha$ is treated as global across cohorts within the analyzed period. When $\alpha=1$, amplification is proportional to baseline risk and, for the same fitted parameters, the wave inversion reduces to the proportional gamma inversion. Enabling the module can still change the fit by excluding marked wave weeks from baseline estimation. The exponent $\alpha$ should be interpreted as a model-calibrated summary of frailty-dependent amplification under the working model, not as a uniquely identified biological or mechanistic constant.

Because $A(t)$ is common across cohorts, cross-cohort ratios of instantaneous wave components eliminate the unknown common amplitude:

$$
\frac{h_{\mathrm{excess},i}(t)}{h_{\mathrm{excess},j}(t)}
=
\frac{F_i(t;\alpha)}{F_j(t;\alpha)}.
$$

**Cumulative correction.** With a supplied nonlinear exponent, fit quiet mortality and invert the observed wave using the same survival law. Solve weekly survival for added exposure with the fitted baseline; when a week falls below that baseline, retain prior wave exposure and solve for a smaller proportional increment. At exit, hold the recovered wave exposure fixed. Ordinary gamma inversion is exact for alpha = 1. See main text §2.7.3 for the equation and limitation of attribution.

**Identification with fixed external scale.** The instantaneous identity does not automatically hold for weekly integrated excess or a no-wave subtraction; the latter compares different survivor populations. The proposed localization procedure requires **externally supplied VE or other prespecified intervention-effect scale**, fixed rather than estimated jointly with $\alpha$ from the wave-period contrasts (main text §2.7.2). Given a fixed scale $x$ relating frailty-neutral excess hazards between cohorts $A$ and $B$,

$$
\frac{\Delta h_A^{\mathrm{obs}}(t)}{\Delta h_B^{\mathrm{obs}}(t)}
=
(1-x)\,\frac{F_A(t;\alpha)}{F_B(t;\alpha)}.
$$

With **externally supplied VE or other prespecified intervention-effect scale** fixed as in main text §2.7.2, the **pairwise objective**, using a measured instantaneous component or a justified approximation, minimizes discrepancies between log excess-hazard ratios and $\log(1-x)+\log F_i(t;\alpha)-\log F_j(t;\alpha)$, in the same spirit as

$$
\sum_{i<j,t}
\left[
\log e_i(t)-\log e_j(t)
-\left(\log F_i(t;\alpha)-\log F_j(t;\alpha)\right)
\right]^2,
$$

with the appropriate $\log(1-x)$ offset absorbed into the target or the residual. The **collapse estimator** minimizes

$$
\operatorname{Var}_d\!\left[
\log e_d(t)-\log F_d(t;\alpha)
\right]
$$

under the same fixed-scale restriction.

Agreement is a consistency check, not independent evidence of identification: at a given time with equal weights, the sum of squared pairwise residual differences equals the number of cohorts times the centered sum of squares. Different weighting or exclusions can change the pooled optimum. Disagreement is diagnostic. The historical weekly approximation has not been validated for identification under the full survivor law. Additional robustness checks vary the excess-hazard handling rule, baseline anchor choice, and theta propagation scale.

Failure of NPH identification manifests through observable signatures: flat objective curves, boundary-seeking estimates, strong dependence on arbitrary settings, or disagreement between the pairwise and collapse estimators. These patterns are treated as evidence of weak signal or model misspecification rather than as valid estimates of $\alpha$.

## S3. Positive controls

### S3.1 Construction of injected effects

The effect window is a simulation construct used solely for positive-control validation and does not represent a real-world intervention period or biological effect window.

During the effect window, the generator multiplies the treated cohort's weekly hazard among survivors by $r$, relative to what that same cohort's hazard would have been without the injection.

$$
h_{\mathrm{treated,injected}}(t) = r \, h_{\mathrm{treated,unperturbed}}(t) \quad \text{for } t_1 \le t < t_2,
$$
{#eq:pos-control-injection}

with $r>1$ for harm and $0<r<1$ for benefit.

The figure shows the historical positive-control comparison. Main-text Table 9 gives the fitted anchored week-80 ratios and intervals. The benefit case is retained as an audit-only stress test because its signed gap interval lies wholly below zero (Section 3.2).

![Historical positive-control comparison. Left panels show harm scenario (r=1.2), right panels show benefit scenario (r=0.8). Top row displays cohort hazard curves with effect window shaded. The lower panels show anchored cumulative hazard ratios reproducible directly from the raw counts. Table 9 gives the fitted endpoints.  X-axis units are weeks since enrollment.](figures/fig_pos_control_injected.png){#fig:pos_control_injected}

## S4. Control-test specifications and simulation parameters

### S4.0 Summary tables for control-test and simulation parameters

Table: Summary of control-test and simulation parameters referenced in Sections S4.2–S4.6. Values shown are representative defaults or grids used to probe the revised estimator; exact scripts and run settings are version-controlled in the repository. {#tbl:si_sim_params}

| Section | Item | Parameter | Value | Notes |
|---|---|---|---|---|
| S4.2.1 | Synthetic validation | Data source / scripts | Repository simulation scripts | Used for $\theta_0$ recovery and null-behavior checks |
| S4.2.1 | Synthetic validation | Frailty heterogeneity grid | Prespecified varying grid | Includes weak- and strong-curvature regimes |
| S4.2.1 | Synthetic validation | Baseline model in estimator | Gompertz with fixed $\gamma$ | Aligns with main-text estimator |
| S4.2.1 | Historical synthetic illustration | Estimator | Four-parameter log-hazard relaxation fit across full follow-up | Five-class mixture; not current gamma-frailty inversion |
| S4.2.1 | Synthetic validation | Time horizon | Multi-year follow-up | Sufficient to probe identifiability |
| S4.2.2 | Empirical negative control | Data source | Czech admin registry data (KCOR_CMR) | Aggregated cohorts |
| S4.2.2 | Empirical negative control | Construction | Age strata remapped to pseudo-doses | Preserves a pseudo-null by design |
| S4.3 | Positive control | Effect multiplier | $r=1.2$ harm; $r=0.8$ benefit | Injected after the same preprocessing pipeline |
| S4.3 | Positive control | Frailty target | Enrollment-time $\theta_{0,d}$ | Estimated separately within each cohort, not fixed in the estimator |
| S4.4 | Sensitivity analysis | Perturbations | Skip weeks, quiet-window placement, omitted windows, anchor choice, excess handling, theta propagation scale, optional NPH settings | Probes robustness of the revised estimator |
| S4.5 | Adversarial selection geometry | Frailty distribution | Tail-mixture versus mid-quantile sampling | Tests robustness to non-gamma geometry |
| S4.6 | Joint frailty + effect (historical) | DGP | Nominal 2 million per cohort; 260 weeks; variances 0.3 and 0.8; effect multipliers 0.8, 1.2 | Earlier single-window cumulative-hazard fitter; 13 scenarios |

### S4.1 Reference implementation and default operational settings

Table: Reference implementation and default operational settings. {#tbl:si_defaults}

| Component | Setting | Default value | Notes |
|---|---|---|---|
| Cohort construction | Cohort indexing | Enrollment period × YearOfBirth group × Dose; plus all-ages cohort (YearOfBirth = -2) | Implementation detail |
| Quiet-period selection | Quiet windows | Prespecified ISO-week intervals spanning follow-up | Used jointly after alignment, not one at a time |
| Early-period stabilization (dynamic HVE) | `SKIP_WEEKS` | 2 | Weeks $t < \mathrm{SKIP\_WEEKS}$ are excluded from accumulation; time is then rebased |
| Gompertz slope | `\gamma` | Fixed implementation value | Same value used throughout the estimator |
| Frailty estimation | Fit method | All-window joint baseline and frailty fit with profiled gaps; nonlinear survivor law for supplied nonlinear exponent | Estimates $(k_d,\theta_{0,d})$ separately within each cohort in the main path |
| Anchoring | `NORMALIZATION_WEEKS` | 4 | Reference window for anchored KCOR plots |
| NPH module | Optional NPH exponent model | Off by default | Enabled only when cross-cohort NPH signal is plausible and diagnostics support identification of $\alpha$ |

### S4.2 Negative controls

Negative controls are used to evaluate the behavior of KCOR under settings where the true effect is known to be null, while allowing substantial heterogeneity in baseline risk and selection-induced depletion. Two complementary classes of negative controls are considered: (i) fully synthetic simulations used to assess both $\theta_0$ recovery and null KCOR behavior, and (ii) empirical registry-based constructions that preserve a pseudo-null by repurposing age strata as pseudo-exposures without selective sampling. Together, these controls assess whether KCOR remains stable in the presence of non-proportional hazards arising from selection rather than treatment.

#### S4.2.1 Synthetic negative controls: gamma-frailty recovery and historical illustration

The historical synthetic negative control (Figure @fig:neg_control_synthetic) illustrates strong depletion curvature under a fixed-frailty null. Its generator (`code/generate_pathological_neg_control_figs.py`) fits separate four-parameter log-hazard relaxation curves over the full follow-up of a five-class frailty mixture and anchors the ratio of the resulting normalized cumulative hazards. It does not fit gamma frailty variance or use the current quiet-window inversion procedure. The separate multiwave recovery example uses expected weekly deaths and fixes the baseline rate at the first observed hazard, which equals the generating rate. It checks recovery of enrollment-time frailty variance and alignment of the quiet windows under those conditions. The historical figure is retained as an illustration of earlier normalization, not as evidence of recovery under the revised estimator. The current-fitter study is in SI S9.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

In the historical illustration, both cohorts share identical per-frailty-group death probabilities; only the mixture weights differ. This induces different cohort-level curvature under the null.

Figure S2 provides a compact summary of observed week-52 KCOR deviation from one as a function of frailty variance under the same synthetic-null grid used in main-text Table 7.

![Simulated-null summary: observed week-52 KCOR deviation from one as a function of enrollment-time frailty variance $\theta_0$. Deviation is defined as week-52 $\mathrm{KCOR} - 1$ in the synthetic-null grid (no treatment effect), reflecting cumulative deviation under the null rather than instantaneous hazard bias. This figure is a diagnostic summary under the working model, not proof that the model is true. Points show single-run estimates from the grid; no error bars are shown.](figures/fig_si_kcor_bias_vs_theta.png){#fig:si_kcor_bias_vs_theta}

![Historical synthetic negative control under strong depletion curvature and no effect at fixed frailty. Top: cohort hazards for distinct five-class frailty-mixture weights. Bottom: anchored cumulative ratio following separate full-follow-up log-hazard relaxation fits. This earlier normalization is not the current gamma-frailty estimator.](figures/fig_neg_control_synthetic.png){#fig:neg_control_synthetic}

#### S4.2.2 Empirical negative control: age-shift construction

The empirical negative control (main-text Figure 6 and SI Figure S9) repurposes registry cohorts to create a **pseudo-null comparison** while inducing large baseline hazard differences via 10–20 year age shifts. Because these are complete strata within the archived September 2025 extract, before the product restriction used for SI S6, rather than selectively sampled subcohorts, their construction introduces no additional selective subsampling; this does not imply limited subsequent frailty depletion or weak identification. The goal here is to test end-to-end KCOR behavior, not to use the empirical geometry as proof that all confounding has been removed.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

This construction compares age groups within the same underlying vaccination category (or all categories pooled). Flatness requires proportional underlying depletion-neutralized hazard trajectories, not merely equal treatment status.

The historical synthetic illustration in Section S4.2.1 deliberately varies frailty-mixture weights to induce different curvature; it used an earlier normalization.

### S4.3 Positive control: injected effect

Positive controls are used to verify that KCOR responds appropriately when a true effect is present. Starting from a negative-control simulation with no treatment effect, a known multiplicative hazard shift is injected into one cohort over a prespecified time window. This construction allows direct assessment of whether the same preprocessing and $\theta_{0,d}$ estimation pipeline detects both the direction and timing of the injected effect while remaining stable outside the effect window.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

The treatment cohort's weekly hazard among survivors follows its unperturbed curve outside the effect window and is multiplied by $r$ within it. The control cohort follows its own unperturbed curve throughout.

### S4.4 Sensitivity analysis parameters

Sensitivity analyses evaluate the robustness of KCOR estimates to reasonable variation in analysis choices that do not alter the underlying data-generating process. Skip weeks, quiet-window placement, omission of individual quiet windows, and optional NPH settings are perturbed over a prespecified range while holding all other parameters fixed. For the optional NPH module, this robustness menu includes baseline anchor choice, excess-hazard handling, theta propagation scale, and agreement between the pairwise and collapse estimators. These analyses assess whether KCOR behavior is stable to tuning choices that primarily affect identification rather than cohort composition.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

The Czech all-ages sweep uses the current ordinary estimator, the supplied non-overlapping all-ages population and the module-off configuration. The seven quiet-window start offsets (-12 to +12 weeks relative to 2023-01, in four-week steps; end fixed at 2023-52) are crossed with reference lengths of two to eight accumulating weeks. Reporting is one year after each enrollment, rounded to the nearest calendar Monday. All 1,323 point estimates across 27 configured comparisons are available; 15 comparisons have grid values on both sides of one (overall range 0.699-1.603). In the three displayed 2021-24 comparisons, shifting the quiet-window start at a fixed reference changes KCOR by at most 0.011, while changing the reference at a fixed quiet window changes it by up to 0.147. These are point sensitivities, not confidence intervals. Full grids, fit statuses and the reproduction script are in `analysis/sensitivity/`.

![Anchored KCOR point sensitivity for the three Czech 2021-24 all-ages dose comparisons on June 13, 2022, using the current ordinary estimator. Rows vary reference length and columns vary the quiet-window start offset. All panels use a shared color scale centered at one. The display is a subset of the full 27-comparison sweep; the cells are point estimates, not uncertainty intervals.](figures/fig_sensitivity_overview.png){#fig:sensitivity_overview}

### S4.5 Tail-sampling / bimodal selection (adversarial selection geometry)

This adversarial simulation evaluates KCOR under extreme but controlled violations of typical cohort-selection geometry. Two cohorts are constructed to share identical mean frailty while differing sharply in how risk is distributed, using mid-quantile sampling versus a low/high-tail mixture. This historical single-quiet-window experiment stress-tests depletion normalization when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.

- **Mid-sampled cohort**: frailty restricted to central quantiles (e.g., 25th–75th percentile) and renormalized to mean 1.
- **Tail-sampled cohort**: mixture of low and high tails (e.g., 0–15th and 85th–100th percentiles) with mixture weights chosen to yield mean 1.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.

Both cohorts share the same baseline hazard $h_0(t)$ and have no treatment effect (negative-control version). Positive-control versions are also generated by applying a known hazard multiplier in a prespecified window. The archived results report KCOR drift, fitted parameters, cumulative-hazard fit error and linearity diagnostics under the earlier fitter. They do not provide replicated coverage or parameter-perturbation stability under the current estimator; the matched current-fitter frailty study is in SI S9.

### S4.6 Joint frailty and treatment-effect simulation

This simulation evaluates KCOR under conditions in which **both selection-induced depletion (frailty heterogeneity)** and a **true treatment effect (harm or benefit)** are present simultaneously. The purpose is to assess whether KCOR can (i) correctly identify and neutralize frailty-driven curvature using aligned quiet windows and (ii) detect a true treatment effect outside that period without confounding the two mechanisms.

This joint simulation combines the selection-induced depletion mechanisms examined in Sections S4.2 and S4.5 with the injected-effect framework of Section S4.3.

The archived S7 experiment used the earlier single-window cumulative-hazard fitter, not the revised all-window estimator. The generator (`test/sim_grid/code/generate_s7_sim.py`) used 260 weeks, nominal 2 million people per cohort, baseline hazard 0.002/week, frailty variances 0.3 and 0.8, two skipped weeks, and quiet weeks 80--140 (zero-based). It used deterministic expected deaths in a 60-bin gamma quantile approximation (quantiles 0.001--0.999). The historical bin masses sum to 0.998 but the initial opening risk is set to the nominal population, leaving 4,000 people per cohort unaccounted for in the first week. That week is within the two-week skip; any future rerun should use consistent initial risk and mixture mass. The 13 scenarios combine step, ramp and bump shapes, multipliers 0.8 and 1.2, and early (10--25) or late (150--190) effect windows, plus a step-harm overlap variant (70--95). Archived trajectories and fit diagnostics are in `test/sim_grid/out/s7_results.xlsx` and `test/sim_grid/out/s7_diagnostics.csv`; both fits reported numerical success in all 13 scenarios. These are historical diagnostic runs, not coverage or current-fitter recovery results.

#### Design

Two fixed cohorts are generated with identical baseline hazards but differing enrollment-time frailty variance. Individual hazards are multiplicatively scaled by a latent frailty term approximating a gamma distribution with unit mean and cohort-specific variance. A treatment effect is then injected over a prespecified time window outside the quiet window in the main grid, or overlapping it in the additional stress variant.

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
- **Offset diagnostics**: number, sign, and stability of reconstructed $\delta_i$ terms when the delta path is used. Report final-fit rather than initialization offsets. With a nonlinear survivor law, the raw gap discrepancy is in observed cumulative-hazard units and the applied increment is in baseline-exposure units.

For real-data Monte Carlo intervals, the resampling unit is one eligible person at enrollment. Each draw retains the complete record and contributes to the counts for its original cohort and follow-up weeks. Ordinary intervals condition on the observed birth-year and dose enrollment sizes; the overlapping age-shift control instead resamples the pooled disjoint components before rebuilding the overlapping groups (main text §2.9.1).

Example diagnostic output from the reference implementation:

```
KCOR_FIT,EnrollmentDate=2021_24,YoB=1950,Dose=0,
  k_hat=4.29e-03,theta0_hat=8.02e-01,
  RMSE_h=3.37e-03,n_obs=97,delta_count=3,success=1
```

### S5.2 Residual analysis

Fit residuals should be examined in hazard space. The integrated weekly prediction is defined in main-text §2.5. Define residuals:

$$
r_{d}(t)=h_d^{\mathrm{adj}}(t)-q_d(t;\hat{k}_d,\hat{\theta}_{0,d},\Delta_d(t)).
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

Specifically, we considered 12-month windows beginning at successive monthly offsets between April 2022 and April 2023 (inclusive). For each placement we jointly fitted baseline rate and frailty using the same survivor law as the main method, with supplied sensitivity alpha=1.18. The quiet fit includes the four early post-skip anchors outside the marked wave. We retained the original enrollment time scale throughout. The numerical acceptance rate counts fits meeting the numerical criteria across all 13 attempted windows per group, including failures in the denominator. Results are shown for two birth-decade strata (1940s and 1950s), and for dose groups $d \in \{0,1,2\}$.

Figure @fig:si_quiet_window_theta_scan shows fitted $\hat{\theta}_{0,d}$ as a function of quiet-window midpoint date. All 78 fits met the numerical acceptance criteria. Of these, 63 triggered the exploratory weekly death-count residual screen (13, 1, and 13 for doses 0, 1, and 2 in the 1940s; 10, 13, and 13 in the 1950s). The screen flags discrepancies without resetting the fitted variance. The estimated frailty variance still depended on window placement. For dose 1 the fitted ranges were 0.00–1.56 in the 1940s and 0.00–0.00 in the 1950s. The windows overlap, so these counts describe sensitivity to placement within the same data.

![Quiet-window robustness scan: fitted enrollment-time frailty variance $\hat{\theta}_{0,d}$ vs. quiet-window midpoint date for the Czech 2021–2024 enrollment cohort, using 12-month windows shifted monthly from April 2022 through April 2023. Rows show birth decade and columns show dose. Each dot gives the fitted frailty variance for one window. All 78 fits were numerically accepted; colors distinguish the 63 with residual-screen flags from the 15 without flags.](figures/fig_si_quiet_window_theta_scan_czech_2021_24.png){#fig:si_quiet_window_theta_scan}

### S5.5 Quiet-window overlay plots

Overlaying prespecified quiet windows on hazard and cumulative-hazard time series plots provides a visual diagnostic of window placement relative to mortality dynamics. The fit windows should:

- Avoid major epidemic waves or external mortality shocks
- Contain sufficient event counts for stable estimation
- Be interpretable after rebasing and, when relevant, after offset alignment

Visual inspection of quiet-window placement relative to mortality dynamics remains an essential diagnostic step, but it should be interpreted jointly with the formal multi-window and offset diagnostics described above.

### S5.6 Robustness to age stratification

This subsection examines $\mathrm{KCOR}(t)$ under narrow age stratification by repeating the same fixed-cohort comparison in three five-year birth bands spanning advanced ages (1930–1934, 1940–1944, 1950–1954). The trajectories and pointwise intervals illustrate how the normalized contrasts vary across age strata under the current estimator; they are descriptive rather than a formal validation of stability.

![Birth band 1930–1934: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and followed through October 7, 2024. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. The horizontal axis shows calendar dates; shaded bands are pointwise 95% person-bootstrap intervals (999 draws). The signed-gap diagnostic was not assessed for these illustrative curves.](figures/supplement/kcor_realdata_yob1930_enroll2022w26_eval2023.png){#fig:si_yob1930}

![Birth band 1940–1944: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and followed through October 7, 2024. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. The horizontal axis shows calendar dates; shaded bands are pointwise 95% person-bootstrap intervals (999 draws). The signed-gap diagnostic was not assessed for these illustrative curves.](figures/supplement/kcor_realdata_yob1940_enroll2022w26_eval2023.png){#fig:si_yob1940}

![Birth band 1950–1954: KCOR(t) trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and followed through October 7, 2024. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior on registry data and does not support causal inference. The horizontal axis shows calendar dates; shaded bands are pointwise 95% person-bootstrap intervals (999 draws). The signed-gap diagnostic was not assessed for these illustrative curves.](figures/supplement/kcor_realdata_yob1950_enroll2022w26_eval2023.png){#fig:si_yob1950}

**Additional empirical negative-control variant (20-year age shift).**  
For completeness, the more extreme 20-year age-shift negative control referenced in the main text is included:

![Empirical negative control with a 20-year age shift: birth years 1950–1959 versus 1930–1939, all vaccination statuses pooled, enrolled in ISO week 2021-24. The all-window joint fitter is used with the optional NPH module inactive. Curves show anchored $\mathrm{KCOR}(t;t_0)$ from the end of the fourth accumulating week after the two-week skip (ISO week 2021-29) through 2024-41. Shading shows pointwise 95% percentile intervals from 999 matched person-bootstrap draws of the disjoint components, refitting the estimator in each draw. The oldest age group triggered the exploratory residual screen in the original fit; 950 of 999 draws flagged at least one contributing fit. All draws were numerically successful and retained. Flatness is expected if the normalized hazard trajectories are proportional. Data source: archived September 2025 Czech registry control, before the product restriction. X-axis: weeks since enrollment.](figures/fig3_neg_control_20yr_age_diff.png){#fig:neg_control_20yr}

## S6. Extended Czech empirical application

### S6.1 Empirical application with diagnostic validation: Czech Republic national registry mortality data

The Czech example illustrates numerical fits and diagnostic outcomes; it does not establish that all pre-specified identification criteria were satisfied. Substantive implications follow only if the identification assumptions hold. Throughout this subsection, observed divergences are interpreted strictly as properties of the estimator under real-world selection, not as intervention effects.

The purpose of this worked example is to illustrate anchored KCOR. Unless otherwise noted, KCOR curves in the Czech analyses are shown anchored at $t_0 = 4$ weeks for interpretability.

#### S6.1.1 Illustrative empirical context: COVID-19 mortality data

The COVID-19 vaccination period provides a natural empirical regime characterized by strong selection heterogeneity and non-proportional hazards, making it a useful illustration for the KCOR framework. During this period, vaccine uptake was voluntary, rapidly time-varying, and correlated with baseline health status, creating clear examples of selection-induced non-proportional hazards. The Czech Republic national mortality registry data exemplify this regime: voluntary uptake led to asymmetric selection at enrollment and to visibly different cohort hazard geometry over follow-up. Under the revised estimator, those differences are summarized through fitted $\theta_{0,d}$ values, aligned quiet-window behavior, depletion-normalized cumulative trajectories, and, when invoked, optional NPH diagnostics for the shared amplification exponent $\alpha$. The Czech application remains illustrative rather than validating; any NPH output should be interpreted as descriptive and model-calibrated rather than as direct biological evidence. While these examples illustrate KCOR's application, the method is general and applies to any retrospective cohort comparison where selection induces differential depletion dynamics.

**Czech population eligibility for the S6 application.** Unlike the archived age-shift controls and Table 8, this application uses the current product-restricted CMR extract. The converter (`code/KCOR_CMR.py`) restricts recorded birth-band starts to 1920 through 2005, retaining unknown birth years, and excludes records with a product code other than Pfizer-BioNTech or Moderna among the first four doses, across the available vaccination history. The all-ages summaries refer to this eligible population. The rule includes post-enrollment products; at enrollment 2021-24, 363,033 otherwise eligible individuals had a disqualifying product in the enrollment week or later and none before enrollment. Of these, 354,604 had no earlier recorded dose among the first four. These counts quantify the potential for retrospective survivor selection, not its effect on KCOR. The alternative enrollment-time eligibility definition has not been evaluated.

#### S6.1.2 Descriptive frailty-normalization behavior in the Czech application

Across examined age strata in the Czech Republic mortality dataset, fitted enrollment-time frailty parameters vary across cohorts, with some estimates close to the weak-identification boundary and others appreciably above zero:

$$
\hat{\theta}_{0,d} \ge 0.
$$
{#eq:si_theta_near_zero}

These values should be interpreted as **descriptive outputs of the revised estimator**, not as validation evidence. They summarize how much depletion curvature the estimator attributes to each cohort at rebased enrollment time under the working model, with diagnostic results shown below.

As a consequence, KCOR normalization leaves some cohorts' cumulative hazards nearly unchanged, while substantially increasing the depletion-neutralized baseline cumulative hazard for others. This behavior is consistent with curvature-driven normalization rather than cohort identity. The pattern is visible directly in depletion-neutralized versus observed cumulative hazard plots and is summarized quantitatively in the fitted-parameter logs (see `KCOR_summary.log`).

The normalized cumulative trajectories are assessed against the fitted Gompertz curve with window offsets (main text §2.6.2). Deviations can reflect baseline misspecification or time-varying risk.

Table @tbl:si_diagnostic_gate reports the numerical fitting status and exploratory residual screen for the six age bands and full aggregate, using the current module-off all-window fitter. All 14 fits were numerically accepted, and six had residual-screen flags. Omitting one configured quiet window at a time while retaining early fitting anchors yielded anchored KCOR point estimates of 1.336–1.555 for ages 40–49 and 1.115–1.134 for all ages (full-window estimates and bootstrap intervals in Table @tbl:si_frailty_variance). These ranges describe point sensitivity, not confidence intervals. Group-specific results are archived in `analysis/s6/window-omission.csv`.

Table: Numerical fitting status and weekly death-count residual flags for the Czech 2021-24 application (ages in calendar year 2024). {#tbl:si_diagnostic_gate}

| Age band (years) | Dose 0 fit status | Dose 2 fit status | Dose 0 residual flag | Dose 2 residual flag |
|---|---|---|---|---|
| 40–49 | OK | OK | No | No |
| 50–59 | OK | OK | No | No |
| 60–69 | OK | OK | No | No |
| 70–79 | OK | OK | No | Yes |
| 80–89 | OK | OK | Yes | Yes |
| 90–99 | OK | OK | No | Yes |
| All ages (eligible population) | OK | OK | Yes | Yes |

Source: `analysis/s6/replay.py` and `analysis/s6/results.json`. The residual screen is exploratory and the fitted parameters are retained. The separate signed first-pass gap diagnostic uses the same 999 person-bootstrap draws as the frailty table: 30 of 42 re-entry intervals are complete and none lies wholly below zero. For ages 40--49 and 50--59, both doses have incomplete gap intervals (respectively 893 and 964, and 996 and 794 valid draws out of 999 for doses 0 and 2); those comparisons remain unassessed, not diagnostic passes. Complete gap estimates, intervals and seed statuses are archived at `analysis/gap_assessment/s6/`. All fitted frailty and KCOR values and intervals are unchanged.

**Interpretation:** In this application, some cohorts exhibit larger fitted $\hat{\theta}_{0,d}$ than others, indicating stronger estimated depletion curvature at rebased enrollment time under the working model:

$$
\hat{\theta}_{0,d} > 0.
$$
{#eq:si_theta_positive}

for some cohorts, while others have fitted variance near the nonnegative parameter boundary:

$$
\hat{\theta}_{0,d} \approx 0.
$$
{#eq:si_theta_near_zero_dose2}

Fitted relative frailty variance may vary across age bands because of differences in entry distributions, sampling uncertainty, or departures from the working model. Under proportional gamma depletion, the absolute survivor variance decreases with the mean, but the relative variance after mean-one rescaling is unchanged; higher baseline exposure alone does not imply a lower relative variance. These descriptive patterns should not be elevated into a validation criterion, and no single dose-ordering pattern is required by the revised estimator. Table @tbl:si_frailty_variance is therefore presented as an empirical summary of fitted parameters rather than as proof that the estimator is correct.

These raw contrasts reflect both selection and depletion effects and are not interpreted causally.

Table: Fitted enrollment-time frailty by age in calendar year 2024 and anchored dose-2/dose-0 KCOR through week 2024-16, with 95% person-bootstrap percentile intervals (999 draws; all-window fit; NPH module off). Frailty intervals describe the raw constrained estimates. {#tbl:si_frailty_variance}

| Age band (years) | Frailty, dose 0 [95%] | Frailty, dose 2 [95%] | Anchored KCOR [95%] |
|---|---:|---:|---:|
| 40–49 | 63.87 [29.60, 96.50] | Poorly resolved [0, 167.90] | 1.507 [0.960, 2.822] |
| 50–59 | 26.82 [18.43, 35.48] | Boundary\* [0.00, 0.00] | 1.228 [0.913, 1.824] |
| 60–69 | 11.47 [9.37, 13.50] | Boundary\* [0.00, 0.00] | 1.295 [1.086, 1.577] |
| 70–79 | 5.16 [4.60, 5.71] | 0.00 [0.00, 0.00] | 1.301 [1.173, 1.441] |
| 80–89 | 1.70 [1.45, 1.94] | 0.00 [0.00, 0.00] | 1.204 [1.108, 1.312] |
| 90–99 | 0.51 [0.39, 0.62] | 0.00 [0.00, 0.00] | 1.012 [0.919, 1.114] |
| All ages | 31.06 [29.38, 32.65] | 0.00 [0.00, 0.28] | 1.136 [1.086, 1.189] |

**Notes:**
- The fitted frailty variance quantifies unobserved frailty heterogeneity and selection-induced depletion within cohorts at rebased enrollment time. Near-zero values indicate little fitted frailty-induced depletion; precision and identification must be assessed separately. At zero variance, the working-model cumulative hazard follows the Gompertz baseline.
- Frailty intervals use the raw constrained optimizer estimates; zeros denote values rounded to two decimals or the lower boundary. \* The dose-2 estimates at ages 50–59 and 60–69 concentrate at zero despite limited local information for small positive variances. A boundary-piled bootstrap is not a calibrated interval for the underlying variance. The 40–49 dose-2 frailty remains poorly resolved. With the numerical ceiling increased to 500, none of the 999 draws reaches the upper-bound flag threshold in either dose group. The upper tail is no longer truncated at 100; the broad raw variance range is a statistical limitation, not evidence that variance is zero. Source: `analysis/gap_assessment/s6/s6-all-window.json` and its matched draws; the previous ceiling and first-window studies are archived separately.
- The "All ages" row fits the supplied eligible-population aggregate counts, including unknown birth years.
- Table @tbl:si_raw_hazards reports raw outcome contrasts for ages 40–99 and the supplied eligible-population total.

**Diagnostic checks:**
- **Interpretation is conditional on diagnostics:** the numerical status and descriptive residual flags are listed above.
- **Near-zero estimates:** these indicate little fitted frailty-induced depletion; their precision and identification must be assessed separately, not inferred from the point estimate alone.
- **Age coherence remains descriptive:** variation across age bands is reported descriptively and may reflect the interaction of baseline hazard, follow-up length, and depletion geometry.
- **Stability:** sign reversals, boundary pathologies, or major sensitivity to window perturbation would count against interpretability.
- **Falsifiability:** failure of any one of these checks would constitute evidence against model adequacy for that application.

Table: Ratio of observed cumulative mortality hazards for dose 0 versus dose 2, Czech cohorts enrolled in 2021-24 (age in calendar year 2024). The eligible-population row includes unknown birth years. {#tbl:si_raw_hazards}

| Age band (years) | Dose 0 cumulative hazard | Dose 2 cumulative hazard | Ratio |
|---|---:|---:|---:|
| 40–49 | 0.005259 | 0.004117 | 1.2776 |
| 50–59 | 0.014968 | 0.009582 | 1.5621 |
| 60–69 | 0.045467 | 0.023134 | 1.9654 |
| 70–79 | 0.123042 | 0.057663 | 2.1338 |
| 80–89 | 0.306825 | 0.167247 | 1.8346 |
| 90–99 | 0.774155 | 0.516352 | 1.4993 |
| All ages (eligible population) | 0.019867 | 0.073304 | 0.2710 |

The cumulative hazards sum the weekly negative log survival probabilities after the two-week stabilization skip (2021-26 through 2024-16 inclusive), before frailty normalization. The age bands use the supplied five-year birth groups combined into ten-year bands. Source: `data/Czech/KCOR_CMR.xlsx`, sheet `2021_24`; replay and results in `analysis/s6/`. The raw contrasts are descriptive.

#### S6.1.3 Illustrative application to national registry mortality data

A brief illustrative application demonstrates end-to-end KCOR behavior on real registry mortality data. Cohorts enrolled in ISO week 2022-26 are followed through 2024-41 (October 7, 2024); the displayed curves begin at the reference week after the two-week stabilization skip. Frailty parameters are estimated using the current all-window joint fitter, with the configured quiet windows within follow-up (2023-19 through 2023-40 and 2024-11 through 2024-14), early fitting anchors, and NPH module off. The figures in S5.6 and S6.1.3 use the same specification and current product-restricted CMR extract. Shading shows pointwise 95% person-bootstrap percentile intervals (999 draws, refitting each draw). All 12 point fits were numerically accepted; residual flags occurred for the 1940-1944 dose-3 cohort and the all-ages dose-0 and dose-3 cohorts. The trajectories and diagnostic results are archived in `analysis/s6-figures/`. This example is intended to illustrate estimator behavior under real-world selection and heterogeneity and does not support causal inference.

Figure @fig:si_allages shows $\mathrm{KCOR}(t)$ trajectories for dose 2 and dose 3 relative to dose 0 for an all-ages analysis. An all-ages analysis is presented as a high-heterogeneity stress test, since aggregation across age induces substantial baseline hazard and frailty variation.

![All-ages stress test: $\mathrm{KCOR}(t)$ trajectories comparing dose 2 and dose 3 to dose 0 for cohorts enrolled in ISO week 2022-26 and followed through October 7, 2024. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). This figure is presented as an illustrative application demonstrating estimator behavior under extreme heterogeneity and does not support causal inference. The horizontal axis shows calendar dates; shaded bands are pointwise 95% person-bootstrap intervals (999 draws). The signed-gap diagnostic was not assessed for these illustrative curves.](figures/kcor_realdata_allages_enroll2022w26_eval2023.png){#fig:si_allages}

## S7. Computational environment and runtime notes

**Environment.** Python 3.11; key dependencies include numpy, scipy, pandas, and lifelines (for Cox-model comparisons), with plotting via matplotlib.

**Compute requirements.** The full simulation grid reproduces in approximately 1 hour 26 minutes on a 20-core CPU with 128 GB RAM; smaller subsets reproduce in minutes.

**Reproduction.** Running `make paper` renders the paper and supplement from the manuscript sources and existing figures. Scientific analyses use their separate runners and settings (see `analysis/README.md`). The S6 table and trajectories, both Table 8 columns, age-shift figures, quiet-window scan and sensitivity grid have been refitted with numerical ceiling 500. The historical S8 draws and study source scripts are archived; `python analysis/survivor_validation/verify_archive.py` reproduces the stored intervals from those draws, not a new simulation under the current estimator.



## S8. Joint-fit and nonlinear survivor-model validation

The main sampled nonlinear-wave experiment (100 independent datasets, 199 resamples) gives mean KCOR 1.00093 [0.99596,1.00591] with the consistent model, versus 0.98982 [0.98493,0.99471] with the gamma working fit and exact inversion; coverage is 90/100 [82.4%,95.1%] for both. In the stress design (50 datasets, 99 resamples), coverage is 48/50 [86.3%,99.5%]; mean with consistent model is 1.04019 [1.01789,1.06249]. In the noiseless labeled controls both original baseline and frailty are recovered. The proportional entry correction changes sampled mean from 1.20386 [1.19425,1.21347] to 0.99774 [0.99321,1.00228]. The actual pre-entry and wave allocation is unknown in registry controls. Earlier validation figures and tables not rerun under the updated estimator illustrate their stated historical configurations.


## S9. Matched frailty-misspecification validation

The five scenarios use 100,000 people per cohort, 120 weeks, a common Gompertz baseline (weekly initial rate 0.002, annual slope 0.085), mean-one frailty at enrollment and variances 1.0 and 0.3. The stabilization skip is two weeks, the reference is the fourth accumulating week, and the quiet window is enrollment weeks 20--80 (zero-based). Gamma, lognormal, two-point and continuous bimodal-mixture frailties are sampled independently across people. For the two-point law, the low support is 0.2 and the high support and weights match the stated moments; the bimodal law is a mixture of two narrow gamma distributions with within-component relative variance 0.02, adjusted to match the same marginal moments. Correlated gamma frailty is shared by clusters of 20 people, with independent clusters. The target anchored endpoint is 1 under the specified fixed-frailty null.

Each scenario uses 100 independent datasets and 199 matched-unit bootstrap replicates of the current fitter. Independent scenarios resample persons; the dependent scenario resamples entire clusters, with a separate person-resampling comparison. Means have 95% Monte Carlo confidence intervals; sampling variance is calculated across independent datasets. Coverage is the proportion of all 100 attempted datasets whose complete percentile interval contains 1, with Wilson uncertainty for coverage. Non-OK numerical-fit and exploratory residual-screen flags are reported separately; neither implies a calibrated identification test.

Table: Matched frailty-misspecification null study at the endpoint. Bias is mean KCOR minus 1. The variance is across independent point estimates, not across time. {#tbl:si_misspecification}

| Frailty / resampling | Mean KCOR [95% MC CI] | Sampling variance | Coverage [95% Wilson CI] | Non-OK fit / 100 | Residual flag / 100 |
|---|---:|---:|---:|---:|---:|
| Gamma / person | 0.995 [0.988, 1.003] | 0.00143 | 98% [93.0, 99.4] | 0 | 4 |
| Lognormal / person | 0.994 [0.986, 1.001] | 0.00141 | 99% [94.6, 99.8] | 0 | 6 |
| Two-point / person | 0.994 [0.986, 1.002] | 0.00165 | 94% [87.5, 97.2] | 0 | 12 |
| Bimodal / person | 1.002 [0.993, 1.011] | 0.00205 | 92% [85.0, 95.9] | 0 | 9 |
| Correlated gamma / cluster | 1.005 [0.997, 1.013] | 0.00155 | 96% [90.2, 98.4] | 0 | 13 |
| Correlated gamma / person | 1.005 [0.997, 1.013] | 0.00155 | 96% [90.2, 98.4] | 0 | 13 |

Full percentile intervals were available for every dataset; no numerical fitting failures occurred. With expected (noiseless) counts, fitted anchored endpoints ranged from 0.99918 to 1.00010; the fitted variance need not equal the true variance for non-gamma laws. For correlated frailty, median interval widths were 0.164 with clusters and 0.165 with persons. The exploratory residual screen is not calibrated for clustered dependence. All Monte Carlo intervals for mean KCOR include 1, as do all Wilson intervals for coverage at 95%. These are design-specific results, not a general coverage guarantee under arbitrary misspecification or dependence. Source: `analysis/misspecification/` (simulation generator, replicate-level results and summary).


\newpage

## S10. Table 8 contributing-fit diagnostics

For each pooled comparison, a diagnostic flag means that at least one positively weighted subgroup fit for either dose triggered that diagnostic. The point flag and number of flagged bootstrap draws out of 999 are reported separately. The proportional analysis was freshly refitted with opening risk sets supplied to the diagnostic routine and the shared seed and pooled ceiling. For the forced-alpha sensitivity, the same routine and survivor-model prediction were evaluated at the archived fitted parameters on the original matched death histories; 300 fresh matched refits under the preceding seed implementation independently reproduced the archived parameters, points and all diagnostic flags exactly. No draw was excluded. The refreshed primary Table 8 point estimates and percentile intervals are unchanged at displayed precision; the forced-alpha intervals remain archived from the preceding seed implementation.

Table: Table 8 diagnostic flags by comparison. Point flag / number of draws (out of 999). {#tbl:si_table8_diagnostics}

| Mode | Enrollment | Dose pair | Residual screen | Raw gap clamp |
|---|---|---|---|---|
| Proportional | 2021_24 | 1/0 | Yes / 999 | No / 0 |
| Proportional | 2021_24 | 2/0 | Yes / 999 | No / 0 |
| Proportional | 2021_24 | 2/1 | Yes / 999 | No / 0 |
| Proportional | 2022_06 | 1/0 | Yes / 999 | Yes / 999 |
| Proportional | 2022_06 | 2/0 | Yes / 999 | Yes / 991 |
| Proportional | 2022_06 | 2/1 | Yes / 999 | Yes / 992 |
| Forced alpha | 2021_24 | 1/0 | Yes / 999 | Yes / 999 |
| Forced alpha | 2021_24 | 2/0 | Yes / 999 | Yes / 999 |
| Forced alpha | 2021_24 | 2/1 | Yes / 999 | Yes / 999 |
| Forced alpha | 2022_06 | 1/0 | Yes / 997 | No / 13 |
| Forced alpha | 2022_06 | 2/0 | Yes / 986 | No / 93 |
| Forced alpha | 2022_06 | 2/1 | No / 986 | No / 90 |

For every comparison, the point flags and draw counts for insufficient signal, unavailable residual screen, upper frailty bound, baseline-rate bound, weak identification, insufficient deaths, failed fit and legacy relative-error screen were zero. The raw gap-clamp indicator records any nonpositive reconstructed offset; it is not the material-deficit assumption diagnostic in Section 2.5. Residual-screen flags are exploratory model-discrepancy indicators, not calibrated global goodness-of-fit tests or criteria for discarding resamples. Source: `analysis/finite_reference/table8-primary/` and archived `analysis/table8_diagnostics/` (full subgroup and comparison flag arrays, category counts, and fresh-refit verification).
