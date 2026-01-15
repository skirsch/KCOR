# KCOR: A depletion-neutralized framework for retrospective cohort comparison under latent frailty

## Manuscript metadata

- **Article type**: Methods / Statistical method
- **Running title**: KCOR under selection-induced cohort bias
- **Author**: Steven T. Kirsch
- **Affiliations**: Independent Researcher, United States
- **Corresponding author**: stk@alum.mit.edu
- **Word count**: 12,100 (excluding Abstract and References)
- **Keywords**: selection bias; frailty model; gamma mixture model; frailty inversion; frailty heterogeneity; selection-induced depletion; non-proportional hazards; cumulative hazard; hazard normalization; cumulative hazards; estimands; gamma frailty; negative controls; observational studies; observational cohort studies

## Abstract

Selection-induced depletion under latent frailty heterogeneity can generate non-proportional hazards and curvature in observed cumulative hazards, biasing standard survival estimands in retrospective cohort studies using registry and administrative data. KCOR is a depletion-neutralized cohort comparison framework based on gamma-frailty normalization. It estimates cohort-specific depletion geometry during prespecified quiet periods and applies an analytic inversion to map observed cumulative hazards into a common comparison scale prior to computing cumulative contrasts. Across simulations spanning frailty heterogeneity and selection strength and across negative and positive controls, Cox proportional hazards regression can exhibit systematic non-null behavior under selection-only regimes. In contrast, KCOR-normalized trajectories remain stable and centered near the null while detecting injected effects. KCOR provides a diagnostic and descriptive framework for comparing fixed cohorts under selection-induced hazard curvature by separating depletion normalization from outcome comparison and improving interpretability of cumulative outcome analyses under minimal-data constraints.

## 1. Introduction

### 1.1 Retrospective cohort comparisons under selection

Randomized controlled trials (RCTs) are the gold standard for causal inference, but are often infeasible, underpowered for rare outcomes, or unavailable for questions that arise after rollout. As a result, observational cohort comparisons are widely used to estimate intervention effects on outcomes such as all-cause mortality.

Although mortality is used throughout this paper as a motivating and concrete example, the method applies more generally to any irreversible event process observed in a fixed cohort, including hospitalization, disease onset, or other terminal or absorbing states. Mortality is emphasized here because it is objectively defined, reliably recorded in many national datasets, and free from outcome-dependent ascertainment biases that complicate other endpoints.

However, when intervention uptake is voluntary, prioritized, or otherwise selective, treated and untreated cohorts are frequently **non-exchangeable** at baseline and evolve differently over follow-up. This problem is not limited to any single intervention class; it arises whenever the same factors that influence treatment uptake also influence outcome risk.

This manuscript is a methods paper. Real-world registry data are used solely to demonstrate estimator behavior, diagnostics, and failure modes under realistic selection-induced non-proportional hazards; no policy conclusions are drawn.

### 1.2 Curvature (shape) is the hard part: non-proportional hazards from frailty depletion

Selection does not merely shift mortality **levels**; it can alter mortality **curvature**—the time-evolution of cohort hazards. Frailty heterogeneity and selection-induced depletion (depletion of susceptibles) naturally induce curvature of the cumulative hazard (reflecting time-varying hazard) even when individual-level hazards are simple functions of time. When selection concentrates high-frailty individuals into one cohort (or preferentially removes them from another), the resulting cohort-level hazard trajectories can be strongly non-proportional.

One convenient way to formalize "curvature" is in cumulative-hazard space: if the cumulative hazard $H(t)$ were perfectly linear in time, then its second derivative would be zero, whereas selection-induced depletion generally produces negative concavity (downward curvature) in observed cumulative hazards during otherwise stable periods.

This violates core assumptions of many standard tools:

- **Cox PH**: assumes hazards differ by a time-invariant multiplicative factor (proportional hazards).
- **IPTW / matching**: can balance measured covariates yet fail to balance unmeasured frailty and the resulting depletion dynamics.
- **Age-standardization**: adjusts levels across age strata but does not remove cohort-specific time-evolving hazard shape.

KCOR is designed for this failure mode: **cohorts whose hazards are not proportional because selection induces different depletion dynamics (curvature).** Approximate linearity of cumulative hazard after adjustment is therefore not assumed, but serves as an internal diagnostic indicating that selection-induced depletion has been successfully removed.

The methodological problem addressed here is general. The COVID-19 period provides a natural empirical regime characterized by strong selection heterogeneity and non-proportional hazards, serving as a useful illustration for the proposed framework. However, KCOR is not specific to COVID, vaccination, or infectious disease. KCOR refers to the method as presented here; earlier internal iterations are not material to the estimand or results and are omitted for clarity.

Two mechanisms often lumped as the 'healthy vaccinee effect' (HVE) are distinguished here:

- **Static HVE:** baseline differences in latent frailty distributions at cohort entry (e.g., vaccinated cohorts are healthier on average). In the KCOR framework, this manifests as differing depletion curvature (different $\theta_d$) and is the primary target of frailty normalization.

- **Dynamic HVE:** short-horizon, time-local selection processes around enrollment that create transient hazard suppression immediately after enrollment (e.g., deferral of vaccination during acute illness, administrative timing, or short-term behavioral/health-seeking changes). Dynamic HVE is operationally addressed by prespecifying a skip/stabilization window (§2.7) and can be evaluated empirically by comparing early-period signatures across related cohorts in multi-dose settings.

### 1.3 Related work (brief positioning)

KCOR builds on the frailty and selection-induced depletion literature in which unobserved heterogeneity induces deceleration of cohort-level hazards over follow-up (a standard working model is gamma frailty) [@vaupel1979]. KCOR’s distinct contribution is not additional hazard flexibility, but a **diagnostics-driven normalization** of selection-induced depletion geometry in cumulative-hazard space prior to defining a cumulative cohort contrast. Related approaches that address non-proportional hazards (time-varying effects, flexible parametric hazards, additive hazards) or time-varying confounding (MSM/IPW/g-methods) target different estimands and typically require richer longitudinal covariates than are available in minimal registry data [@grambsch1994; @andersen1982; @royston2002; @aalen1989; @lin1994; @vanhouwelingen2007; @robins2000; @cole2008]. Additional discussion is provided in the Supplementary Information (SI).

### 1.4 Evidence from the literature: residual confounding despite meticulous matching

Motivating applied studies show that even careful matching and adjustment can leave substantial residual differences in non-COVID mortality and time-varying “healthy vaccinee effect” signatures, consistent with selection and depletion dynamics not captured by measured covariates [@obel2024; @chemaitelly2025].

### 1.5 Contribution of this work

This work makes four primary contributions: (i) it formalizes selection-induced depletion under latent frailty heterogeneity as a source of non-proportional hazards and curvature that can bias common survival estimands; (ii) it defines a diagnostics-first normalization that fits depletion geometry in quiet periods and maps observed cumulative hazards into a depletion-neutralized space; (iii) it validates operating characteristics using synthetic and empirical controls, including a synthetic null under selection-only regimes; and (iv) it separates normalization from comparison by permitting standard post-normalization cumulative estimands.

A central implication is identifiability: in minimal-data retrospective cohorts, interpretability depends on an epidemiologically quiet window and on internal diagnostics that indicate depletion geometry has been estimated and removed, rather than absorbed into a time-varying effect estimate.

Together, these contributions position KCOR not as a replacement for existing survival estimands, but as a prerequisite normalization step that addresses a source of bias arising prior to model fitting in many retrospective cohort studies.

### 1.6 Target estimand and scope (non-causal)

> **Box 1. Target estimand and scope (non-causal).**
>
> - **Primary estimand (KCOR)**: For two fixed enrollment cohorts $A$ and $B$, it is defined as
>   $$
>   \mathrm{KCOR}(t)=\tilde{H}_{0,A}(t)/\tilde{H}_{0,B}(t),
>   $$
>   where $\tilde{H}_{0,d}(t)$ is cohort $d$'s **depletion-neutralized baseline cumulative hazard** obtained by fitting depletion geometry in a prespecified quiet window and applying the gamma-frailty inversion (Methods §2).
> - **Interpretation**: KCOR is a time-indexed **cumulative** contrast on the depletion-neutralized scale. Values above/below 1 indicate greater/less cumulative event accumulation in cohort $A$ than $B$ by time $t$ after depletion normalization. KCOR is not an instantaneous hazard ratio.
> - **What it is not**: KCOR is **not** a causal effect estimator (no ATE/ATT) and does not recover counterfactual outcomes under hypothetical interventions.
> - **When interpretable**: Interpretation is conditional on explicit assumptions (fixed cohorts; shared external hazard environment; adequacy of the working frailty model; existence of an epidemiologically quiet window) **and** on internal diagnostics (quiet-window fit quality; post-normalization linearity within the quiet window; parameter stability to small window perturbations).
> - **If diagnostics indicate non-identifiability**: the analysis is treated as not identified and KCOR is not reported as a “corrected effect”.

### 1.7 Paper organization and supporting information (SI)

The main text introduces the KCOR estimator, provides a canonical demonstration of Cox bias under frailty-driven depletion, and presents two primary validation examples (a negative control and a stress test). Additional validations—including positive controls—along with extended diagnostics, empirical registry-based nulls, and detailed simulation specifications are provided in the Supplementary Information (SI; Sections S2 and S4.2–S4.3; Tables @tbl:si_assumptions–@tbl:si_identifiability).

## 2. Methods

Mortality is used as the primary example throughout this section because it is objectively defined and reliably recorded in many administrative datasets.

Table @tbl:notation defines the notation used throughout the Methods section.

For COVID-19 vaccination analyses, intervention count corresponds to the number of vaccine doses received; more generally, this can index any discrete exposure level.

### 2.1 Conceptual framework and estimand

Retrospective cohort differences can arise from two qualitatively different components:

* **Level differences**: cohort hazards differ by an approximately time-stable multiplicative factor (or, equivalently, cumulative hazards have different slopes but similar shape).
* **Depletion (curvature) differences**: cohort hazards evolve differently over time because cohorts differ in latent heterogeneity and are **selectively depleted** at different rates.

This framework targets the second failure mode. Under latent frailty heterogeneity, high-risk individuals die earlier, so the surviving risk set becomes progressively "healthier." This induces **downward curvature** (deceleration) in cohort hazards and corresponding concavity in cumulative-hazard space, even when individual-level hazards are simple and even under a true null treatment effect. When selection concentrates frailty heterogeneity differently across cohorts, the resulting curvature differences produce strong non-proportional hazards and can drive misleading contrasts for estimands that condition on the evolving risk set.

The strategy is therefore:

1. **Estimate the cohort-specific depletion geometry** (via curvature) during prespecified epidemiologically quiet periods.
2. **Map observed cumulative hazards into a depletion-neutralized space** by inverting that geometry.
3. **Compare cohorts only after normalization** using a prespecified post-adjustment estimand; ratios of depletion-neutralized cumulative hazards (KCOR) are used here.

All analyses are performed using discrete weekly time bins; continuous-time notation is used solely for expositional convenience.

#### 2.1.1 Target estimand

Scope and interpretation are summarized in Box 1 (§1.6); the formal definition used throughout is provided here.

Let $\tilde{H}_{0,d}(t)$ denote the **depletion-neutralized baseline cumulative hazard** for cohort $d$ at event time $t$ since enrollment (Table @tbl:notation). For two cohorts $A$ and $B$, KCOR is defined as

$$
\mathrm{KCOR}(t) = \frac{\tilde{H}_{0,A}(t)}{\tilde{H}_{0,B}(t)}.
$$
{#eq:kcor-estimand}

For visualization, an **anchored KCOR** is sometimes reported to show post-reference divergence:
$$
\mathrm{KCOR}(t; t_0) = \mathrm{KCOR}(t)/\mathrm{KCOR}(t_0),
$$
with prespecified $t_0$ (e.g., 4 weeks).

#### 2.1.2 Identification versus diagnostics

Scope and interpretation are summarized in Box 1 (§1.6). 

Interpretability of a KCOR trajectory is assessed via prespecified diagnostics (Supplementary Information §S2; Tables @tbl:si_assumptions–@tbl:si_identifiability). When diagnostics indicate non-identifiability, the analysis is treated as not identified and results are not reported. Checks include:

* stability of $(\hat{k}_d,\hat{\theta}_d)$ to small quiet-window perturbations,
* approximate linearity of $\tilde{H}_{0,d}(t)$ within the quiet window,
* absence of systematic residual structure in cumulative-hazard space.

Diagnostics corresponding to each assumption are summarized in Supplementary Information §S2 (Tables @tbl:si_assumptions–@tbl:si_identifiability).

#### 2.1.3 KCOR assumptions and diagnostics

These assumptions define when KCOR normalization is interpretable.

The KCOR framework relies on the following assumptions, which are framed diagnostically:

1. **Fixed cohort enrollment.**
   Cohorts are defined at a common enrollment time and followed forward without dynamic entry or rebalancing.

2. **Multiplicative latent frailty.**
   Individual hazards are assumed to be multiplicatively composed of a baseline hazard and an unobserved frailty term, with cohort-specific frailty distributions.

3. **Quiet-window stability.**
   A prespecified epidemiologically quiet period exists during which external shocks to the baseline hazard are minimal, allowing depletion geometry to be estimated from observed cumulative hazards.

4. **Independence across strata.**
   Cohorts or strata are analyzed independently, without interference, spillover, or cross-cohort coupling.

5. **Sufficient event-time resolution.**
   Event timing is observed at a temporal resolution adequate to estimate cumulative hazards over the quiet window.

These assumptions are evaluated empirically using post-normalization diagnostics. Violations are expected to manifest as residual curvature, drift, or instability in adjusted cumulative hazard trajectories.

### 2.2 Cohort construction

KCOR is defined for **fixed cohorts at enrollment**. Required inputs are minimal: enrollment date(s), event date, and optionally birth date (or year-of-birth) for age stratification. Analyses proceed in discrete event time $t$ (e.g., weeks) measured since cohort enrollment.

Cohorts are assigned by intervention state at the start of the enrollment interval. In the primary estimand:

* **No post-enrollment switching** is allowed (individuals remain in their enrollment cohort),
* **No censoring** is applied (other than administrative end of follow-up),
* analyses are performed on the resulting fixed risk sets.

Censoring or reclassification due to cohort transitions (e.g., moving between exposure groups over time) is not permitted, because such transitions alter the frailty composition of the cohort in a time-dependent manner. Allowing transitions would introduce additional, endogenous selection that changes cohort mortality trajectories in unpredictable ways, confounding depletion effects that KCOR is designed to normalize.

This fixed-cohort design is intentional. It avoids immortal-time artifacts and prevents outcome-driven switching rules from creating time-dependent selection that is difficult to diagnose under minimal covariate availability. Extensions that allow switching or censoring are treated as sensitivity analyses (§5.2) because they change the estimand and introduce additional identification requirements.

Conceptual requirements of the KCOR framework are distinguished from operational defaults, which are reported separately for reproducibility (Supplementary Section S4).

Throughout this manuscript the failure event is **all-cause mortality**. KCOR therefore targets cumulative mortality hazards and is not framed as a cause-specific competing-risks analysis.

### 2.3 Hazard estimation and cumulative hazards in discrete time

For each cohort $d$, let $N_d(0)$ denote the number of individuals at enrollment.
Let $d_d(t)$ denote deaths occurring during interval $t$, and let
$$
D_d(t) = \sum_{s \le t} d_d(s)
$$
denote cumulative deaths up to the end of interval $t$.

Define the risk set size at the start of interval $t$ as
$$
N_d(t) = N_d(0) - \sum_{s < t} d_d(s) = N_d(0) - D_d(t-1).
$$
In the primary estimand, individuals do not switch cohorts after enrollment and there is no loss to follow-up; therefore $N_d(t)$ is the risk set used to define all discrete-time hazards and cumulative hazards in this manuscript.

Define the interval mortality ratio
$$
\mathrm{MR}_{d,t} = \frac{d_d(t)}{N_d(t)}.
$$

The discrete-time cohort hazard is computed as

$$
h_{\mathrm{obs},d}(t) = -\ln\!\left(1 - \mathrm{MR}_{d,t}\right) = -\ln\!\left(1 - \frac{d_d(t)}{N_d(t)}\right).
$$
{#eq:hazard-discrete}

This transform is standard: it maps an interval event probability into a continuous-time equivalent hazard under a piecewise-constant hazard assumption. For rare events, $h_{\mathrm{obs},d}(t) \approx \mathrm{MR}_{d,t} = d_d(t)/N_d(t)$, but the log form remains accurate and stable when weekly risks are not negligible.

*All hazard and cumulative-hazard quantities used in KCOR are discrete-time integrated hazard estimators derived from fixed-cohort risk sets; likelihood-based or partial-likelihood formulations are not used for estimation or for the subsequent frailty-based normalization.*

Observed cumulative hazards are accumulated over event time after an optional stabilization skip (§2.7):

$$
H_{\mathrm{obs},d}(t) = \sum_{s \le t} h_d^{\mathrm{eff}}(s),
\qquad \Delta t = 1.
$$
{#eq:cumhazard-observed}

Discrete binning accommodates tied events and aggregated registry releases. Bin width is chosen based on diagnostic stability (e.g., smoothness and sufficient counts per bin) rather than temporal resolution alone.

In addition to the primary implementation above, $\hat H_{\mathrm{obs},d}(t)$ was computed using the Nelson--Aalen estimator $\sum_{s \le t} d_d(s)/N_d(s)$ as a sensitivity check; results were unchanged.

### 2.4 Selection model: gamma frailty and depletion normalization

#### 2.4.1 Individual hazards with multiplicative frailty

Within cohort $d$, individual $i$ is modeled as having hazard

$$
h_{i,d}(t) = z_{i,d}\,h_{0,d}(t),
\qquad
z_{i,d} \sim \mathrm{Gamma}(\mathrm{mean}=1,\ \mathrm{var}=\theta_d).
$$
{#eq:individual-hazard-frailty}

Here $h_{0,d}(t)$ is the cohort's depletion-neutralized baseline hazard and $z_{i,d}$ is a latent multiplicative frailty term. The frailty variance $\theta_d$ governs the strength of depletion-induced curvature: larger $\theta_d$ yields stronger deceleration at the cohort level due to faster early depletion of high-frailty individuals.

Gamma frailty is used because it yields a closed-form link between observed and baseline cumulative hazards via the Laplace transform [@vaupel1979]. In KCOR, gamma frailty is a **working geometric model** for depletion normalization, not a claim of biological truth. Adequacy is evaluated empirically via fit quality, post-normalization linearity, and stability diagnostics.

#### 2.4.2 Gamma-frailty identity and inversion

Let

$$
H_{0,d}(t) = \int_0^t h_{0,d}(s)\,ds
$$
{#eq:baseline-cumhazard}

denote the baseline cumulative hazard. Integrating over gamma frailty yields the gamma-frailty identity

$$
H_{\mathrm{obs},d}(t) = \frac{1}{\theta_d}\,\log\!\left(1 + \theta_d H_{0,d}(t)\right),
$$
{#eq:gamma-frailty-identity}

which can be inverted exactly as

$$
H_{0,d}(t) = \frac{\exp\!\left(\theta_d H_{\mathrm{obs},d}(t)\right) - 1}{\theta_d}.
$$
{#eq:gamma-frailty-inversion}

This inversion is the **normalization operator**: given an estimate $\hat{\theta}_d$, it maps the observed cumulative hazard $H_{\mathrm{obs},d}(t)$ into a depletion-neutralized cumulative hazard scale.

#### 2.4.3 Baseline shape used for frailty identification

To identify $\theta_d$, KCOR fits the gamma-frailty model within prespecified epidemiologically quiet periods. In the reference specification, the baseline hazard is taken to be constant over the fit window:

$$
h_{0,d}(t)=k_d,
\qquad
H_{0,d}(t)=k_d\,t.
$$
{#eq:baseline-shape-default}

This choice intentionally minimizes degrees of freedom: during a quiet window, curvature is forced to be explained by depletion (via $\theta_d$) rather than by introducing time-varying baseline hazard terms. If the observed cumulative hazard is near-linear over the fit window, the model naturally collapses toward $\hat{\theta}_d \approx 0$, signaling weak or absent detectable depletion curvature for that cohort over that window.

#### 2.4.4 Quiet-window validity as the key dataset-specific requirement

Frailty parameters are estimated using only bins whose corresponding calendar weeks lie inside a prespecified quiet window (defined in ISO-week space). The quiet window is prespecified to avoid sharp, cohort-differential hazard perturbations (e.g., epidemic waves or policy shocks) that would confound depletion-geometry estimation. A window is acceptable only if diagnostics indicate (i) good fit in cumulative-hazard space, (ii) post-normalization linearity within the window, and (iii) stability of $(\hat{k}_d,\hat{\theta}_d)$ to small boundary perturbations. If no candidate window passes, diagnostics indicate non-identifiability and the analysis is treated as not identified rather than producing a potentially misleading normalized contrast. All diagnostics are computed over discrete event-time bins (weekly intervals since enrollment) whose corresponding calendar weeks fall within the prespecified quiet window.

#### Quiet-window selection protocol (operational)

Quiet-window selection is prespecified and evaluated using diagnostic criteria summarized in Supplementary Information §S2 (Tables @tbl:si_assumptions–@tbl:si_identifiability).

### 2.5 Estimation during quiet periods (cumulative-hazard least squares)

KCOR estimates $(\hat{k}_d,\hat{\theta}_d)$ independently for each cohort $d$ using only time bins that fall inside a prespecified quiet window in calendar time (see §2.4.4). The quiet window is applied consistently across cohorts within an analysis. Let $\mathcal{T}_d$ denote the set of event-time bins $t$ whose corresponding calendar week lies in the quiet window, with $t$ also satisfying $t \ge \mathrm{SKIP\_WEEKS}$.

Under the default baseline shape, the model-implied observed cumulative hazard is

$$
H_{d}^{\mathrm{model}}(t; k_d, \theta_d) = \frac{1}{\theta_d}\,\log\!\left(1+\theta_d k_d t\right).
$$
{#eq:hobs-model}

Identifiability of $(\hat{k}_d,\hat{\theta}_d)$ comes from curvature in cumulative-hazard space: observed cumulative hazards are nonlinear in event time when $\theta_d>0$. When depletion is weak (or the quiet window is too short to show curvature), the model smoothly collapses to a linear cumulative hazard, since $H_{d}^{\mathrm{model}}(t; k_d, \theta_d) \to k_d t$ as $\theta_d \to 0$. Operationally, near-linear observed cumulative hazards naturally drive the fitted frailty variance toward zero; fit diagnostics such as $n_{\mathrm{obs}}$ and RMSE in $H$-space provide a practical check on whether the selection parameters are being identified from the quiet-window data. In practice, lack of identifiable curvature naturally manifests as fitted frailty variance estimates approaching zero, providing an internal diagnostic for non-identifiability over short or sparse follow-up.

In applied analyses, this behavior is most commonly observed in vaccinated cohorts, whose cumulative hazards during quiet periods are often close to linear. In such cases, the gamma-frailty fit collapses naturally, indicating minimal detectable depletion. This outcome reflects the absence of observable selection-induced curvature rather than a modeling assumption. When residual time-varying risk contaminates a nominally quiet window, fitted frailty variance estimates naturally shrink toward zero, signaling limited identifiability rather than inducing spurious correction.

Parameters are estimated by constrained nonlinear least squares:

$$
(\hat{k}_d,\hat{\theta}_d)
=
\arg\min_{k_d>0,\ \theta_d \ge 0}
\sum_{t \in \mathcal{T}_d}
\left[
H_{\mathrm{obs},d}(t) - H_{d}^{\mathrm{model}}(t; k_d, \theta_d)
\right]^2.
$$
{#eq:nls-objective}

Fitting is performed in cumulative-hazard space rather than maximizing a likelihood because the primary inputs are discrete-time, cohort-aggregated hazards and the objective is stable estimation of selection-induced depletion curvature during quiet periods. Least-squares fitting is used as a numerical estimating equation rather than as a likelihood-based estimator. Least squares on observed cumulative hazards is numerically robust under sparse events, emphasizes shape agreement over the fit window, and yields diagnostics (e.g., RMSE in $H$-space) that directly reflect the quality of the depletion fit. Likelihood-based fitting can be treated as a sensitivity analysis, but is not required for the normalization identity itself.

All analyses use a prespecified reference implementation with fixed operational defaults; full details are provided in Supplementary Section S4.

### 2.6 Normalization (depletion-neutralized cumulative hazards)

After fitting, KCOR computes the depletion-neutralized baseline cumulative hazard for each cohort $d$ by applying the inversion to the full post-enrollment trajectory:

$$
\tilde{H}_{0,d}(t) = \frac{\exp\!\left(\hat{\theta}_d\,H_{\mathrm{obs},d}(t)\right)-1}{\hat{\theta}_d}.
$$
{#eq:normalized-cumhazard}

This normalization maps each cohort into a depletion-neutralized baseline-hazard space in which the contribution of gamma frailty parameters $(\hat{\theta}_d, \hat{k}_d)$ to hazard curvature has been factored out. This normalization defines a common comparison scale in cumulative-hazard space; it is not equivalent to Cox partial-likelihood baseline anchoring, but serves an analogous geometric role for cumulative contrasts. In this space, cumulative hazards are directly comparable across cohorts, and remaining differences reflect real differences in baseline risk rather than selection-induced depletion.
The core identities used in KCOR are given in Equations @eq:hazard-discrete, @eq:nls-objective, @eq:normalized-cumhazard, and @eq:kcor-estimand. Normalization defines a common comparison scale; the scientific estimand is then computed on that scale (Box 1).

#### 2.6.1 Computational considerations

KCOR operates on aggregated event counts in discrete time and cumulative-hazard space. Computational complexity scales linearly with the number of time bins and strata rather than the number of individuals, making the method feasible for very large population registries. In practice, KCOR analyses on national-scale datasets (millions of individuals) are memory-bound rather than CPU-bound and can be implemented efficiently using standard vectorized numerical libraries. No iterative optimization over individual-level records is required.

#### 2.6.2 Internal diagnostics and 'self-check' behavior

KCOR includes internal diagnostics intended to make model stress visible rather than hidden.

1. **Post-normalization linearity in quiet periods.** Within the prespecified quiet window (see §2.4.4), the depletion-neutralized cumulative hazard should be approximately linear in event time after inversion. Systematic residual curvature indicates window contamination (external shocks, secular trends) or misspecified depletion geometry for that cohort.

2. **Fit residual structure in cumulative-hazard space.** Define residuals over the fit set $\mathcal{T}_d$:

$$
r_{d}(t)=H_{\mathrm{obs},d}(t)-H_{d}^{\mathrm{model}}(t;\hat{k}_d,\hat{\theta}_d).
$$
{#eq:si_residuals}

KCOR expects residuals to be small and not systematically time-structured. Strongly patterned residuals indicate that the curvature attributed to depletion is instead being driven by unmodeled time-varying hazards.

3. **Parameter stability to window perturbations.** Under valid quiet-window selection,

$$
(\hat{k}_d,\hat{\theta}_d)
$$

should be stable to small perturbations of the quiet-window boundaries (e.g., ±4 weeks). Large changes in fitted frailty variance under small boundary shifts signal that the fitted curvature is sensitive to transient dynamics rather than stable depletion.

4. **Non-identifiability manifests as:**

$$
\hat{\theta}_d\rightarrow 0.
$$

When the observed cumulative hazard is near-linear (weak curvature) or events are sparse, $\theta$ is weakly identified. In such cases, KCOR should be interpreted primarily as a diagnostic (limited evidence of detectable depletion curvature) rather than a strong correction.

These diagnostics are reported alongside $\mathrm{KCOR}(t)$ curves. The goal is not to assert that a single parametric form is always correct, but to ensure that when the form is incorrect or the window is contaminated, the method signals this explicitly rather than silently producing a misleading 'corrected' estimate. When diagnostics indicate non-identifiability, the depletion-based normalization is inappropriate and KCOR should not be interpreted.

### 2.7 Stabilization (early weeks)

In many applications, the first few post-enrollment intervals can be unstable due to immediate post-enrollment artifacts (e.g., rapid deferral, short-term sorting, administrative effects). KCOR supports a prespecified stabilization rule by excluding early weeks from accumulation and from quiet-window fitting. The skip-weeks parameter is prespecified and evaluated via sensitivity analysis to exclude early enrollment instability rather than to tune estimates.

In discrete time, define an effective hazard for accumulation:

$$
h_d^{\mathrm{eff}}(t)=
\begin{cases}
0, & t < \mathrm{SKIP\_WEEKS} \\
h_{\mathrm{obs},d}(t), & t \ge \mathrm{SKIP\_WEEKS}.
\end{cases}
$$
{#eq:effective-hazard-skip}

Then compute observed cumulative hazards from $h_d^{\mathrm{eff}}(t)$ as in §2.3:

$$
H_{\mathrm{obs},d}(t).
$$

### 2.8 KCOR estimator

With depletion-neutralized cumulative hazards in hand, the primary KCOR trajectory is defined as:

$$
\mathrm{KCOR}(t) = \frac{\tilde{H}_{0,A}(t)}{\tilde{H}_{0,B}(t)}.
$$
{#eq:kcor-estimator}

This ratio is computed after depletion normalization and is interpreted conditional on the stated assumptions and diagnostics (Box 1; §2.1.2).

### 2.9 Uncertainty quantification

Uncertainty is quantified using stratified bootstrap resampling, which propagates uncertainty through the full pipeline (event counts, frailty fitting, inversion, and KCOR computation).

#### 2.9.1 Stratified bootstrap procedure

The stratified bootstrap procedure for KCOR proceeds as follows:

1. **Resample cohort--time counts (aggregated bootstrap).** Within each cohort and stratum (e.g., age group), resample event counts and risk-set sizes within time bins, preserving the original cohort/stratum structure. When individual-level data are available, an equivalent individual-level resampling can be used; in this work, we use the aggregated cohort--time bootstrap.

2. **Re-estimate frailty parameters.** For each bootstrap replicate, re-estimate $(\hat{k}_d,\hat{\theta}_d)$ independently for each cohort $d$ using the resampled data, applying the same quiet-window selection and fitting procedure as in the primary analysis.

3. **Recompute normalized cumulative hazards.** Using the bootstrap-estimated frailty parameters, recompute $\tilde{H}_{0,d}(t)$ for each cohort using Eq. @eq:normalized-cumhazard applied to the resampled observed cumulative hazards.

4. **Recompute KCOR.** Compute $\mathrm{KCOR}(t)$ for each bootstrap replicate as the ratio of the bootstrap-normalized cumulative hazards.

5. **Form percentile intervals.** From the bootstrap distribution of $\mathrm{KCOR}(t)$ values at each time point, form percentile-based confidence intervals (e.g., 2.5th and 97.5th percentiles for 95% intervals).

Bootstrap resampling is performed at the cohort-count level in the aggregated representation (resampling event counts and risk-set sizes within time bins and strata), rather than resampling individual-level records.

Uncertainty intervals reflect event stochasticity and model-fit uncertainty in $(\hat{k}_d,\hat{\theta}_d)$ and are interpreted conditional on the observed risk sets and modeling assumptions.

### 2.10 Algorithm summary and reproducibility checklist

Table @tbl:KCOR_algorithm summarizes the complete KCOR pipeline.

![**KCOR as a two-stage framework.**
**(A)** Fixed-cohort cumulative hazards exhibit curvature due to selection-induced depletion; late-time curvature is used to estimate frailty parameters for normalization.
**(B)** Gamma-frailty normalization yields approximately linearized cumulative hazards that are directly comparable across cohorts; $\mathrm{KCOR}(t)$, defined as the ratio of depletion-neutralized baseline cumulative hazards, is near-flat under the null and deviates only under net hazard differences.
*In the schematic, $\tilde{H}_{0,d}(t)$ denotes the depletion-neutralized baseline cumulative hazard.*
](figures/fig_kcor_workflow.png){#fig:kcor_workflow}


### 2.11 Relationship to Cox proportional hazards

Cox proportional hazards models estimate an instantaneous hazard ratio under the assumption that hazards differ by a time-invariant multiplicative factor. Under selective uptake with latent frailty heterogeneity, this assumption is typically violated, yielding time-varying hazard ratios induced purely by depletion dynamics. This reflects an estimand mismatch: Cox targets a different quantity under depletion than KCOR's cumulative hazard estimand. Cox is behaving correctly for its estimand, but that estimand may not align with the scientific question when selection-induced depletion is present. Accordingly, Cox results are presented here as a diagnostic demonstration of estimand mismatch, not as a competing intervention-effect estimator.

Cox regression estimates a weighted average hazard ratio under non-proportional hazards; KCOR targets a cumulative hazard estimand. Even when Cox models are extended with shared frailty to accommodate heterogeneity, they continue to estimate instantaneous hazard ratios conditional on survival, whereas KCOR estimates cumulative contrasts after explicit depletion normalization.

Conceptually, Cox regression estimates an instantaneous hazard ratio by fitting a hazard model to observed data, whereas KCOR uses a parametric working model only to normalize selection-induced depletion geometry and then computes a cumulative contrast on the depletion-neutralized scale.

#### 2.11.1 Demonstration: Cox bias under frailty heterogeneity with no treatment effect

A controlled synthetic experiment was conducted in which the **true effect is known to be zero by construction**, isolating latent frailty heterogeneity as the sole driver of depletion-induced non-proportional hazards. Cox and KCOR were applied to the same simulated datasets under identical information constraints.

**Data-generating process.**

Two cohorts of equal size were simulated under the same baseline hazard $h_0(t)$ over time (constant or Gompertz). Individual hazards were generated as $z\,h_0(t)$, with frailty
$$
z \sim \text{Gamma}(\theta^{-1}, \theta^{-1}),
$$
with mean 1 and variance $\theta$.

Cohort A was generated with $\theta = 0$ (no frailty heterogeneity), while Cohort B was generated with $\theta > 0$. **No treatment or intervention effect was applied**: conditional on frailty, the two cohorts have identical hazards at all times. Thus, the true hazard ratio between cohorts is exactly 1 for all $t$.

Simulations were repeated over a grid of frailty variances $\theta \in \{0, 0.5, 1, 2, 5, 10, 20\}$.

**Cox analysis.**

For each simulated dataset, a standard Cox proportional hazards model was fitted using partial likelihood (statsmodels `PHReg`), with cohort membership as the sole covariate (no time-varying covariates or interactions). The resulting hazard ratio estimates and confidence intervals therefore reflect **only differences induced by frailty-driven depletion**, not any treatment effect.

**KCOR analysis.**

The same simulated datasets were analyzed using KCOR. Observed cumulative hazards were estimated nonparametrically using the Nelson–Aalen estimator, then normalized using Eq. @eq:normalized-cumhazard with frailty parameters fitted in the prespecified quiet window prior to computing $\mathrm{KCOR}(t)$. Although the data-generating process specifies individual hazards, Nelson–Aalen is used to mirror the information available in observational registry studies rather than exploiting simulator-only knowledge. Post-normalization slope and asymptotic $\mathrm{KCOR}(t)$ values were examined to assess departure from the null.

**Expected behavior under the null.**

Because the data-generating process includes **no treatment effect**, any valid estimator should return a null result. In this setting:

* **Cox regression** is expected to produce apparent non-null hazard ratios as $\theta$ increases, reflecting differential selection-induced depletion and violation of proportional hazards induced by frailty heterogeneity.
* **KCOR** is expected to remain centered near unity with negligible post-normalization slope across all $\theta$, consistent with correct null behavior after depletion normalization.

**Summary of findings.**

Across increasing values of $\theta$, Cox regression produced progressively larger apparent deviations from a hazard ratio of 1. The direction and magnitude of the apparent effect depended on the follow-up horizon and degree of frailty heterogeneity. In contrast, $\mathrm{KCOR}(t)$ trajectories remained stable and centered near unity, with post-normalization slopes approximately zero across all simulated conditions.

These results demonstrate that **frailty heterogeneity alone is sufficient to induce spurious hazard ratios in Cox regression**, while KCOR correctly returns a null result under the same conditions.

Table @tbl:cox_bias_demo reports numerical summaries of the Cox-vs-KCOR behavior across the frailty grid.

Additional Cox HR results from the same synthetic-null grid are shown in Figure @fig:cox_bias_hr.

A compact summary of KCOR bias as a function of frailty variance $\theta$ is provided in the Supplementary Information (Figure @fig:si_kcor_bias_vs_theta).

![Cox regression produces spurious non-null hazard ratios under a *synthetic null* as frailty heterogeneity increases. Hazard ratios (with 95% confidence intervals) from Cox proportional hazards regression comparing cohort B to cohort A in simulations where the true treatment effect is identically zero and cohorts differ only in frailty variance ($\theta$). Deviations from HR=1 arise solely from frailty-driven depletion and associated non-proportional hazards.](figures/fig_cox_bias_hr_vs_theta.png){#fig:cox_bias_hr}

![$\mathrm{KCOR}(t)$ remains null under a synthetic null across increasing frailty heterogeneity. $\mathrm{KCOR}(t)$ asymptotes remain near 1 across $\theta$ in the same simulations, consistent with correct null behavior after depletion normalization. Uncertainty bands (95% bootstrap intervals) are shown but are narrow due to large sample sizes.](figures/fig_cox_bias_kcor_vs_theta.png){#fig:cox_bias_kcor}

**Interpretation.**

This controlled synthetic null shows that Cox proportional hazards regression can report highly statistically significant non-null hazard ratios even when the true effect is identically zero, purely due to frailty-driven depletion and induced non-proportional hazards. KCOR remains near unity under the same conditions because depletion normalization precedes comparison.

### 2.12 Worked example (descriptive)

A brief worked example is included to illustrate the KCOR workflow end-to-end. This example is descriptive and intended solely to demonstrate the mechanics of cohort construction, hazard estimation, frailty fitting, depletion normalization, and KCOR computation.

The example proceeds from aggregated cohort counts through cumulative-hazard estimation, quiet-window frailty fitting, gamma inversion, and $\mathrm{KCOR}(t)$ construction, accompanied by diagnostic plots assessing post-normalization linearity and parameter stability.

### 2.13 Reproducibility and computational implementation

All figures, tables, and simulations can be reproduced from the accompanying code repository. The manuscript is built from `documentation/preprint/paper.md` using the root `Makefile` paper target `make paper-full`.

Additional environment and runtime details are provided in the Supplementary Information (SI); code and archival links are provided in Code/Data Availability.


## 3. Results

This section is the core validation claim of KCOR:

- **Negative controls (null under selection):** under a true null effect, KCOR remains approximately flat at 1 even when selection induces large curvature differences.
- **Positive controls (detect injected effects):** when known harm/benefit is injected into otherwise-null data, KCOR reliably detects it.
- **Failure signaling (diagnostics):** when key assumptions are violated or the working model is stressed, KCOR’s diagnostics degrade (e.g., poor quiet-window fit, post-normalization nonlinearity, parameter instability), and the analysis is treated as not identified rather than reported as a stable contrast.

Throughout, curvature in cumulative hazard plots reflects selection-induced depletion, while linearity after normalization indicates successful removal of that curvature.

In vaccinated–unvaccinated comparisons, large early differences in $\mathrm{KCOR}(t)$ may reflect baseline risk selection rather than intervention effects; in such cases, $\mathrm{KCOR}(t; t_0)$ is emphasized to report deviations relative to an early post-enrollment reference while preserving time-varying divergence.

### 3.1 Negative controls: null under selection-induced curvature
#### 3.1.2 Empirical negative control using national registry data (Czech Republic)

This application is presented solely to illustrate KCOR's diagnostic behavior on real registry data; it uses an age-shift construction (pseudo-cohorts) that is a negative control by design rather than an observational treatment-effect analysis.

The repository includes a pragmatic negative control construction that repurposes a real dataset by comparing "like with like" while inducing large composition differences (e.g., age band shifts). In this construction, age strata are remapped into pseudo-doses so that comparisons are, by construction, within the same underlying category; the expected differential effect is near zero, but the baseline hazards differ strongly.

These age-shift negative controls deliberately induce extreme baseline mortality differences (10–20 year age gaps) while preserving a true null effect by construction, since all vaccination states are compared symmetrically. The near-flat $\mathrm{KCOR}(t)$ trajectories are consistent with the estimator normalizing selection-induced depletion curvature without introducing spurious time trends or cumulative drift.

For the empirical age-shift negative control (Figure @fig:neg_control_10yr), aggregated weekly cohort summaries derived from the Czech Republic administrative mortality and vaccination dataset are used and exported in KCOR_CMR format.

Notably, KCOR estimates frailty parameters independently for each cohort without knowledge of exposure status; the observed asymmetry in depletion correction arises entirely from differences in hazard curvature rather than from any vaccination-specific assumptions.

Figure @fig:neg_control_10yr provides a representative illustration; additional age-shift variants are provided in the Supplementary Information (SI).

![Empirical negative control with approximately 10-year age difference between cohorts. Despite large baseline mortality differences, $\mathrm{KCOR}(t)$ remains near-flat at 1 over follow-up, consistent with a true null effect. Curves are shown as anchored $\mathrm{KCOR}(t; t_0)$, i.e., $\mathrm{KCOR}(t)/\mathrm{KCOR}(t_0)$, which removes pre-existing cumulative differences and displays post-anchor divergence only. KCOR curves are anchored at $t_0 = 4$ weeks (i.e., plotted as $\mathrm{KCOR}(t; t_0)$). Uncertainty bands (95% bootstrap intervals) are shown. Data source: Czech Republic mortality and vaccination dataset processed into KCOR_CMR aggregated format (negative-control construction; see Supplementary Information, SI).](figures/fig2_neg_control_10yr_age_diff.png){#fig:neg_control_10yr}

Table @tbl:neg_control_summary provides numeric summaries.

<!--
NOTE: The empirical negative control is built from real-world data where non-proportional external hazards (e.g., epidemic waves) can create small deviations from an idealized null.
The key validation claim is that KCOR does not produce spurious *drift* under large composition differences; curves remain near-flat, and injected effects (positive controls) are detectable.
-->

### 3.2 Positive controls: detect injected harm/benefit

Positive controls (injected harm/benefit) are provided in Supplementary Section S3. They verify that under a known injected effect, KCOR deviates in the expected direction and with magnitude consistent with the injection (up to discretization and sampling noise).

In positive-control simulations with injected multiplicative hazard shifts, KCOR reliably detects both harm and benefit, with estimated $\mathrm{KCOR}(t)$ trajectories tracking the imposed effects; full results are shown in Supplementary Figure S1.

### 3.3 Stress test: robustness to frailty misspecification

#### 3.3.1 Frailty misspecification robustness

To assess robustness to departures from the gamma frailty assumption, simulations were conducted under alternative frailty distributions while maintaining the same selection-induced depletion geometry. Simulations were performed for:

- **Gamma** (baseline reference)
- **Lognormal** frailty
- **Two-point mixture** (discrete frailty)
- **Bimodal** frailty distributions
- **Correlated frailty** (within-subgroup correlation)

For each frailty specification, bias (deviation from true cumulative hazard ratio), variance (trajectory stability), coverage (proportion of simulations where uncertainty intervals contain the true value), and non-identifiability rate (proportion of simulations where quiet-window diagnostics indicated non-identifiability) are reported.

Under frailty misspecification, KCOR can degrade gracefully by attenuating toward unity or by not meeting diagnostic criteria, rather than producing spurious large effects. When the alternative frailty distribution produces similar depletion geometry to gamma frailty, KCOR normalization remains approximately valid, with bias remaining small and diagnostics indicating successful identification. When the alternative frailty structure produces substantially different depletion geometry, KCOR diagnostics (poor cumulative-hazard fit, residual autocorrelation, parameter instability) correctly signal that the gamma-frailty approximation is inadequate, and $\mathrm{KCOR}(t)$ trajectories either remain near-unity (reflecting attenuation) or are not computed when diagnostic thresholds are not met.
Additional validation results—including full simulation grids, quiet-window robustness catalogs, dynamic-selection checks, and extended comparator analyses—are provided in the Supplementary Information (SI).

Additional derivations, simulation studies, robustness analyses, and implementation details are provided in the Supplementary Information.

## 4. Discussion

### 4.1 Limits of attribution and non-identifiability

KCOR does not uniquely identify the biological, behavioral, or clinical mechanisms responsible for observed hazard heterogeneity. In particular, curvature in the cumulative hazard may arise from multiple sources, including selection on latent frailty, behavior change, seasonality, treatment effects, reporting artifacts, or their combination. Depletion of susceptibles is therefore used as a parsimonious working model whose adequacy is evaluated through diagnostics and negative controls, rather than assumed as a substantive truth. KCOR's estimand is whether a cumulative outcome contrast persists after removal of curvature consistent with selection-induced depletion, not attribution of that curvature to a specific mechanism.

### 4.2 What KCOR estimates

*Table @tbl:positioning clarifies that KCOR differs from non-proportional hazards methods not in flexibility, but in estimand and direction of inference.* KCOR operates at a specific but critical layer of the retrospective inference stack: it both neutralizes selection-induced depletion dynamics and defines how the resulting depletion-neutralized baseline cumulative hazards must be compared.

Once cohorts are mapped into depletion-neutralized baseline cumulative hazard space, $\mathrm{KCOR}(t)$ answers whether one cohort accumulated higher or lower cumulative event risk than another by time $t$, conditional on the stated assumptions and diagnostics (Box 1). For intuition, $\mathrm{KCOR}(t)=1.2$ indicates that, after depletion normalization, cohort $A$ has accumulated approximately 20% greater cumulative hazard than cohort $B$ by time $t$. Stabilization of $\mathrm{KCOR}(t)$ in quiet windows is a falsification check: failure to flatten indicates residual curvature or loss of identifiability, not a substantive cumulative effect.

*Many commonly used survival estimands—such as hazard ratios, cumulative hazard differences, or restricted mean survival time—are not intrinsically invalid. Their failure in retrospective cohort studies arises when they are applied to unadjusted data exhibiting selection-induced depletion. KCOR does not replace these estimands; instead, it provides a normalization step that restores comparability. After depletion normalization, such estimands may be meaningfully computed, with the choice driven by interpretability rather than by identifiability constraints imposed by selection bias.*

The frailty term is used as a geometric working model for selection-induced depletion; it is not interpreted mechanistically.

KCOR is a **cumulative** comparison of depletion-neutralized cumulative hazards; it does not estimate instantaneous hazard ratios. It is designed for settings where selection induces non-proportional hazards such that conventional proportional-hazards estimators can be difficult to interpret. A controlled synthetic null experiment (Section 2.11.1) shows that Cox regression can return statistically significant non-null hazard ratios solely from frailty-induced depletion—even when the true treatment effect is identically zero—reflecting an estimand mismatch where Cox targets a different quantity under depletion than KCOR's cumulative estimand. Cox is behaving correctly for its estimand, but that estimand may not align with the scientific question when selection-induced depletion is present. KCOR remains centered near unity with negligible post-normalization slope under the same conditions.

Under the working assumptions that:

1. selection-induced depletion dynamics can be estimated during quiet periods using a gamma-frailty mixture model, and
2. the fitted selection parameters can be used to invert observed cumulative hazards into depletion-neutralized baseline cumulative hazards,

then the remaining differences between cohorts are interpretable, **conditional on the stated selection model and quiet-window validity**, as differences in baseline hazard level (on a cumulative scale), summarized by $\mathrm{KCOR}(t)$.

A useful way to view KCOR is as an intermediate layer between purely descriptive hazard summaries and fully identified effect estimators. KCOR is descriptive in that it summarizes cohort differences in a cumulative-hazard scale under explicit normalization of depletion geometry; it is inferential in that it provides falsifiable diagnostics and control-test behavior that constrain when the normalized contrast is interpretable. Under minimal-data constraints, explicitly normalizing a dominant bias geometry and reporting when identifiability is not supported can be more reliable than insisting on point-identification of an intervention effect.

The observation that frailty correction is negligible for vaccinated cohorts but substantial for the unvaccinated cohort is not incidental. It reflects the asymmetric action of healthy-vaccinee selection, which concentrates lower-frailty individuals into vaccinated cohorts at enrollment while leaving the unvaccinated cohort heterogeneous. KCOR explicitly detects and removes this asymmetry by mapping cohorts into a depletion-neutralized comparison space rather than assuming proportional hazards.

Because the normalization targets selection-induced depletion curvature, KCOR results alone do not justify claims about net lives saved or lost by a particular intervention. This manuscript focuses on method definition, diagnostics, and operating characteristics; interpretation should be read in light of the non-identifiability considerations described above.

Although cumulative hazards and survival functions are in one-to-one correspondence, KCOR operates in cumulative-hazard space because curvature induced by frailty depletion is additive and more readily diagnosed there. While survival-based summaries such as restricted mean survival time may be derived from depletion-neutralized baseline cumulative hazards, KCOR's primary estimand remains cumulative by construction.

### 4.3 Relationship to negative control methods

Negative control outcomes/tests are widely used to *detect* confounding. KCOR's objective is different: it is an estimator intended to *normalize away a specific confounding structure*—selection-induced depletion dynamics—prior to comparison. Negative and positive controls are nevertheless central to validating the estimator's behavior.

This asymmetry helps explain why standard observational analyses often report large apparent mortality benefits during periods lacking a plausible mechanism: vaccinated cohorts are already selection-filtered, while unvaccinated hazards are suppressed by ongoing frailty depletion. Unadjusted comparisons therefore systematically understate unvaccinated baseline risk and exaggerate apparent benefit.

### 4.4 Practical guidelines for implementation

This subsection summarizes common operational practices for applying KCOR in retrospective cohort studies and for assessing when resulting contrasts are interpretable.

Reporting commonly includes:

- Enrollment definition and justification
- Risk set definitions and event-time binning
- Quiet-window definition and justification
- Baseline-shape choice (default constant baseline over the fit window) and fit diagnostics
- Skip/stabilization rule and robustness to nearby values
- Predefined negative/positive controls used for validation
- Sensitivity analysis plan and results

KCOR should therefore be applied and reported as a complete pipeline—from cohort freezing, through depletion normalization, to cumulative comparison and diagnostics—rather than as a standalone adjustment step. Scope and interpretation are summarized once in Box 1 (§1.6).

## 5. Limitations

This section summarizes the principal limitations of the KCOR framework, emphasizing conditions under which interpretation is restricted rather than situations in which the estimator fails. These limitations are diagnostic and design-related, reflecting the framework’s intentionally conservative scope.

KCOR is intentionally diagnostic rather than test-based: it does not attempt to formally test properties such as quiet-window validity or frailty distributional form, but instead enforces conservative interpretability gates when prespecified empirical diagnostics fail. KCOR is not a causal effect estimator and does not identify counterfactual outcomes under hypothetical interventions.

- **Model dependence**: Normalization relies on the adequacy of the gamma-frailty model and the baseline-shape assumption during the quiet window.
- **Relation to existing non-PH methods**: KCOR is complementary to time-varying Cox, flexible parametric, additive hazards, and MSM approaches; these methods address different estimands and identification strategies, whereas KCOR targets depletion-geometry normalization under minimal-data constraints (see §1.3.1).
- **$\theta$ estimation is data-derived**: KCOR does not impose $\theta = 0$ for any cohort. The frequent observation that fitted frailty variance estimates collapse toward zero for vaccinated cohorts is a result of the frailty fit and should not be interpreted as an assumption of homogeneity.
- **Sparse events**: When event counts are small, hazard estimation and parameter fitting can be unstable.
- **Contamination of quiet periods**: External shocks (e.g., epidemic waves) overlapping the quiet window can bias selection-parameter estimation.
- **Applicability to other outcomes**: Although this paper focuses on all-cause mortality, KCOR is applicable to other irreversible outcomes provided that event timing and risk sets are well defined. Application to cause-specific mortality requires careful consideration of competing risks and interpretation of cumulative hazards within cause-restricted populations. Extension to non-fatal outcomes such as hospitalization is conceptually straightforward but may require additional attention to outcome definitions, censoring mechanisms, and recurrent events. These considerations affect interpretation rather than the core KCOR framework.
- **Non-gamma frailty**: The KCOR framework assumes that selection acts approximately multiplicatively through a time-invariant frailty distribution, for which the gamma family provides a convenient and empirically testable approximation. In settings where depletion dynamics are driven by more complex mechanisms—such as time-varying frailty variance, interacting risk factors, or shared frailty correlations within subgroups—the curvature structure exploited by KCOR may be misspecified. In such cases, KCOR diagnostics (e.g., poor curvature fit or unstable fitted frailty variance estimates) serve as indicators of model inadequacy rather than targets for parameter tuning. Extending the framework to accommodate dynamic or correlated frailty structures would require explicit model generalization rather than modification of KCOR normalization steps and is left to future work. Empirically, KCOR's validity depends on curvature removal rather than the specific parametric form; alternative frailty distributions that generate similar depletion geometry would yield equivalent normalization.

### 5.1 Failure modes and diagnostics

KCOR is designed to normalize selection-induced depletion curvature under its stated model and windowing assumptions. Reviewers and readers should expect the method to degrade when those assumptions are violated. Common failure modes include:

- **Mis-specified quiet window**: If the quiet window overlaps major external shocks (epidemic waves, policy changes, reporting artifacts), the fitted parameters may absorb non-selection dynamics, biasing normalization.
- **External time-varying hazards masquerading as frailty depletion**: Strong secular trends, seasonality, or outcome-definition changes can introduce curvature that is not well captured by gamma-frailty depletion alone. For example, COVID-19 waves disproportionately increase mortality among frail individuals; if one cohort has higher baseline frailty, such a wave can preferentially deplete that cohort, producing the appearance of a benefit in the lower-frailty cohort that is actually due to differential frailty-specific mortality from the external hazard rather than from the intervention under study.
- **Extremely sparse cohorts**: When events are rare, observed cumulative hazards become noisy and $(\hat{k}_d,\hat{\theta}_d)$ can be weakly identified, often manifesting as unstable fitted frailty variance estimates or wide uncertainty.
- **Non-frailty-driven curvature**: Administrative censoring, cohort-definition drift, changes in risk-set construction, or differential loss can induce curvature unrelated to latent frailty.

Practical diagnostics include:

- **Quiet-window overlays** on hazard/cumulative-hazard plots to confirm the fit window is epidemiologically stable.
- **Fit residuals in $H$-space** (RMSE, residual plots) and stability of fitted parameters under small perturbations of the quiet-window bounds.
- **Sensitivity analyses** over plausible quiet windows and skip-weeks values.
- **Prespecified negative controls**: $\mathrm{KCOR}(t)$ curves should remain near-flat at 1 under control constructions designed to induce composition differences without true effects.

In practice, prespecified negative controls—such as the age-shift controls presented in §3.1.2—provide a direct empirical check that KCOR does not generate artifactual cumulative effects under strong selection-induced curvature.

### 5.2 Conservativeness and edge-case detection limits

Because KCOR compares fixed enrollment cohorts, subsequent uptake of the intervention among initially unexposed individuals (or additional dosing among exposed cohorts) introduces treatment crossover over time. Such crossover attenuates between-cohort contrasts and biases $\mathrm{KCOR}(t)$ toward unity, making the estimator conservative with respect to detecting sustained net benefit or harm. Analyses should therefore restrict follow-up to periods before substantial crossover or stratify by dosing state when the data permit.

Because KCOR defines explicit diagnostic failure modes—instability, dose reversals, age incoherence, or absence of asymptotic convergence—the absence of such failures in the Czech 2021_24 Dose 0 versus Dose 2 cohorts provides stronger validation than goodness-of-fit alone.

**Conservativeness under overlap.**  
When treatment effects overlap temporally with the quiet window used for frailty estimation, $\mathrm{KCOR}(t)$ does not attribute the resulting curvature to treatment nor amplify it into a spurious cumulative effect. Instead, overlap manifests as degraded quiet-window fit, reduced post-normalization linearity, and instability of estimated frailty parameters, all of which are explicitly surfaced by KCOR's diagnostics. In these regimes, $\mathrm{KCOR}(t)$ trajectories tend to attenuate toward unity rather than diverge, reflecting loss of identifiability rather than false detection. This behavior is illustrated in the S7 overlap variant, where treatment and selection are deliberately confounded in time: $\mathrm{KCOR}(t)$ does not recover a clean effect signal, and diagnostic criteria correctly indicate that the assumptions required for interpretable normalization are violated. As a result, KCOR is conservative under temporal overlap—preferring diagnostic failure and attenuation over over-interpretation—rather than producing misleading treatment effects when separability is not supported by the data. This design choice reflects an intentional bias toward false negatives rather than false positives in ambiguous regimes. See §2.1.1 and Supplementary Section S7 for the corresponding identifiability assumptions and stress tests.

KCOR analyses commonly exclude an initial post-enrollment window to exclude dynamic Healthy Vaccinee Effect artifacts. If an intervention induces an acute mortality effect concentrated entirely within this skipped window, that transient signal will not be captured by the primary analysis. This limitation is addressed by reporting sensitivity analyses with reduced or zero skip-weeks and/or by separately evaluating a prespecified acute-risk window.

In degenerate scenarios where an intervention induces a purely proportional level-shift in hazard that remains constant over time and does not alter depletion-driven curvature, KCOR's curvature-based contrast may have limited ability to distinguish such effects from residual baseline level differences under minimal-data constraints. Such cases are pathological in the sense that they produce no detectable depletion signature; in practice, KCOR diagnostics and control tests help identify when curvature-based inference is not informative.

Simulation results in §3.4 illustrate that when key assumptions are violated—such as non-gamma frailty geometry, contamination of the quiet window by external shocks, or extreme event sparsity—frailty normalization may become weakly identified. In such regimes, KCOR's diagnostics, including poor cumulative-hazard fit and reduced post-normalization linearity, explicitly signal that curvature-based inference is unreliable without model generalization or revised window selection.

Increasing model complexity within the Cox regression framework—via random effects, cohort-specific frailty, or information-criterion–based selection—does not resolve this limitation, because these models continue to target instantaneous hazard ratios conditional on survival rather than cumulative counterfactual outcomes. Model-selection criteria applied within the Cox regression family favor specifications that improve likelihood fit of instantaneous hazards, but such criteria do not validate cumulative counterfactual interpretation under selection-induced non-proportional hazards.

### 5.3 Data requirements and external validation

In finite samples, KCOR precision is driven primarily by the number of events observed over follow-up. In simulation (selection-only null), cohorts of approximately 5,000 per arm yielded stable KCOR estimates with narrow uncertainty, whereas smaller cohorts exhibited appreciable Monte Carlo variability and occasional spurious deviations. Reporting event counts and conducting a simple cohort-size sensitivity check are recommended when applying KCOR to sparse outcomes.

**External validation across interventions.** A natural next step is to apply KCOR to other vaccines and interventions where large-scale individual-level event timing data are available. Many RCTs are underpowered for all-cause mortality and typically do not provide record-level timing needed for KCOR-style hazard-space normalization, while large observational studies often publish only aggregated effect estimates. Where sufficiently detailed time-to-event data exist (registries, integrated health systems, or open individual-level datasets), cross-intervention comparisons can help characterize how often selection-induced depletion dominates observed hazard curvature and how frequently post-normalization trajectories remain stable under negative controls.

## 6. Conclusion

KCOR addresses selection-induced hazard curvature in retrospective cohort comparisons by explicitly modeling and inverting frailty-driven depletion prior to comparison. Across synthetic and empirical controls, KCOR remains near-null under selection without effect and reliably detects injected effects when present. Rather than presuming identifiability, KCOR enforces its assumptions diagnostically, flagging violations through degraded fit, instability, or residual curvature instead of absorbing them into model-dependent estimates.

\newpage

## Declarations

### Ethics approval and consent to participate

This study used only simulated data and publicly available, aggregated registry summaries that contain no individual-level or identifiable information; as such, it did not constitute human subjects research and was exempt from institutional review board oversight. The primary validation results use synthetic data. Empirical negative-control figures (Figures @fig:neg_control_10yr and @fig:neg_control_20yr) use aggregated cohort summaries derived from Czech Republic administrative data; no record-level data are shared in this manuscript.[@sanca2024]

### Consent for publication

Not applicable.

### Data availability

This study analyzes aggregated cohort-level summaries derived from administrative health records. The underlying individual-level data from the Czech Republic are record-level administrative data collected and maintained by the National Health Information Portal and were not accessed directly by the author. Access to the underlying record-level data is subject to the data provider’s governance, approval, and disclosure-control policies.

All synthetic validation datasets used for method development and evaluation (including negative and positive control simulations), along with their generation scripts, are publicly available in the project repository. Sensitivity analysis outputs and example datasets in KCOR_CMR format are included to support full computational reproducibility. A formal specification of the KCOR data formats, including schema definitions and disclosure-control semantics, is provided in `documentation/specs/KCOR_file_format.md`.

The complete KCOR reference implementation, simulation code, and manuscript build instructions are available at https://github.com/skirsch/KCOR. A citable archival release of the software is available via Zenodo (DOI: 10.5281/zenodo.18050329).

### Use of artificial intelligence tools

The KCOR method and estimand were developed by the author without the use of artificial intelligence (AI) tools. Generative AI tools, including OpenAI’s ChatGPT and Cursor Composer 1, were used during manuscript preparation to assist with drafting and editing text, mathematical typesetting, refactoring code, and implementing simulation studies described in this manuscript.

Simulation designs were either specified by the author or proposed during iterative discussion and subsequently reviewed and approved by the author prior to implementation. AI assistance was used to draft code for approved simulations, which the author reviewed, tested, and validated. Additional large language models (including Gemini, DeepSeek, and Claude) were used to provide feedback on manuscript wording and methodological exposition in a role analogous to informal peer review.

All scientific decisions, methodological choices, analyses, interpretations, and judgments regarding which suggestions to accept or reject were made solely by the author, who reviewed and understands all content and takes full responsibility for the manuscript.

### Competing interests

The author is a board member of the Vaccine Safety Research Foundation.

### Funding

This research received no external funding.

### Authors' contributions

Steven T. Kirsch conceived the method, wrote the code, performed the analysis, and wrote the manuscript.

### Acknowledgements

The author thanks James Lyons-Weiler. Dr. Clare Craig, and Paul Fischer for helpful discussions and methodological feedback during the development of this work. All errors remain the author’s responsibility.

\newpage

## References

::: {#refs}

:::

\newpage

## Tables

Table: Summary of two large matched observational studies showing residual confounding / HVE despite meticulous matching. {#tbl:HVE_motivation}

| Study | Design | Matching/adjustment | Key control finding | Implication for methods |
|---|---|---|---|---|
| Obel et al. (Denmark) [@obel2024] | Nationwide registry cohorts (60–90y) | 1:1 match on age/sex + covariate adjustment; negative control outcomes | Vaccinated had higher rates of multiple negative control outcomes, but substantially lower mortality after unrelated diagnoses | Strong evidence of confounding in observational VE estimates; "negative control methods indicate… substantial confounding" |
| Chemaitelly et al. (Qatar) [@chemaitelly2025] | Matched national cohorts (primary series and booster) | Exact 1:1 matching on demographics + coexisting conditions + prior infection; Cox models | Strong early reduction in non-COVID mortality (HVE), with time-varying reversal later | Even meticulous matching leaves time-varying residual differences consistent with selection/frailty depletion |


Table: Comparison of Cox proportional hazards, Cox with frailty, and KCOR across key methodological dimensions. {#tbl:cox_vs_kcor}

| Feature                             | Cox PH       | Cox + frailty     | KCOR                    |
| ----------------------------------- | ------------ | ----------------- | ----------------------- |
| Primary estimand                    | Hazard ratio | Hazard ratio      | Cumulative hazard ratio |
| Conditions on survival              | Yes          | Yes               | No                      |
| Assumes PH                          | Yes          | Yes (conditional) | No                      |
| Frailty role                        | None         | Nuisance          | Object of inference     |
| Uses partial likelihood             | Yes          | Yes               | No                      |
| Handles selection-induced curvature | No           | Partial           | Yes (targeted)          |
| Output interpretable under non-PH   | No           | No                | Yes (cumulative)        |

Note: KCOR is reported here as a cumulative hazard ratio for comparability; alternative post-normalization estimands are admissible within the framework.


Table: Positioning KCOR relative to non-proportional hazards methods. {#tbl:positioning}

| Method class                           | Primary target                  | What is modeled        | Handles selection-induced depletion? | Typical output           | Failure under latent frailty      |
| -------------------------------------- | ------------------------------- | ---------------------- | ------------------------------------ | ------------------------ | --------------------------------- |
| Cox PH                                 | Instantaneous hazard            | Linear predictor       | No                                   | HR                       | Non-PH from depletion → biased HR |
| Time-varying Cox                       | Instantaneous hazard            | Time-varying $\beta(t)$      | No                                   | HR(t)                    | Fits depletion as signal          |
| Flexible parametric survival (splines) | Survival / hazard shape         | Baseline hazard        | No                                   | Smooth hazard / survival | Absorbs depletion curvature       |
| Additive hazards (Aalen)               | Hazard differences              | Additive hazard        | No                                   | $\Delta h(t)$            | Still conditional on survival     |
| RMST                                   | Mean survival                   | Survival curve         | No                                   | RMST                     | Inherits depletion bias           |
| Frailty regression                     | Heterogeneity- adjusted HR       | Random effects         | Partial                              | HR                       | Frailty treated as nuisance       |
| **KCOR (this work)**                   | **Cumulative outcome contrast** | **Depletion geometry** | **Yes (targeted)**                   | **$\mathrm{KCOR}(t)$**              | Diagnostics flag failure          |


Table: Notation used throughout the Methods section. {#tbl:notation}

| Symbol | Definition |
|--------|------------|
| $d$ | Cohort index |
| $A,B$ | Indices of the two cohorts compared in a KCOR contrast |
| $t$ | Event time since enrollment (discrete bins, e.g., weeks) |
| $h_{\mathrm{obs},d}(t)$ | Discrete-time cohort hazard (conditional on $N_d(t)$) |
| $H_{\mathrm{obs},d}(t)$ | Observed cumulative hazard (after skip/stabilization) |
| $h_{0,d}(t)$ | Baseline hazard for cohort $d$ under the depletion-neutralized model |
| $H_{0,d}(t)$ | Baseline cumulative hazard for cohort $d$ under the depletion-neutralized model |
| $\tilde{H}_{0,d}(t)$ | Depletion-neutralized baseline cumulative hazard |
| $\theta_d$ | Frailty variance (selection strength) for cohort $d$; governs curvature in the observed cumulative hazard |
| $\hat{\theta}_d$ | Estimated frailty variance from quiet-window fitting |
| $k_d$ | Baseline hazard level for cohort $d$ under the default baseline shape |
| $\hat{k}_d$ | Estimated baseline hazard level from quiet-window fitting |
| $t_0$ | Anchor time for baseline normalization (prespecified) |
| $\mathrm{KCOR}(t; t_0)$ | Anchored KCOR: $\mathrm{KCOR}(t)/\mathrm{KCOR}(t_0)$ |

Table: Step-by-step KCOR algorithm (high-level), with recommended prespecification and diagnostics. {#tbl:KCOR_algorithm}

| Step | Operation | Output | Prespecify? | Diagnostics |
|---|---|---|---|---|
| 1 | Choose enrollment date and define fixed cohorts | Cohort labels | Yes | Verify cohort sizes/risk sets |
| 2 | Compute discrete-time hazards (observed hazards) | Hazard curves | Yes (binning/transform) | Check for zeros/sparsity |
| 3 | Apply stabilization skip and accumulate observed cumulative hazards | Observed cumulative hazards | Yes (skip rule) | Plot observed cumulative hazards |
| 4 | Select quiet-window bins in calendar ISO-week space | Fit points $\mathcal{T}_d$ | Yes | Overlay quiet window on hazard plots |
| 5 | Fit $(\hat{k}_d,\hat{\theta}_d)$ via cumulative-hazard least squares | Fitted parameters | Yes | RMSE, residuals, fit stability |
| 6 | Normalize: invert gamma-frailty identity to depletion-neutralized cumulative hazards | Depletion-neutralized cumulative hazards | Yes | Compare pre/post shapes; sanity checks |
| 7 | Cumulate and ratio: compute $\mathrm{KCOR}(t)$ | $\mathrm{KCOR}(t)$ curve | Yes (horizon) | Flat under negative controls |
| 8 | Uncertainty | CI / intervals | Yes | Coverage on positive controls |


Table: Cox vs KCOR under a synthetic null with increasing frailty heterogeneity. Two cohorts are simulated with identical baseline hazards and no treatment effect *(null by construction)*; cohorts differ only in gamma frailty variance ($\theta$). Despite the true hazard ratio being 1 by construction, Cox regression produces increasingly non-null hazard ratios as $\theta$ increases, reflecting depletion-induced non-proportional hazards. $\mathrm{KCOR}(t)$ remains centered near unity with negligible post-normalization slope across $\theta$ values. (Exact values depend on simulation seed and follow-up horizon.) {#tbl:cox_bias_demo}

| $\theta$ | Cox HR | 95% CI | Cox p-value | KCOR asymptote | KCOR post-norm slope |
| --------: | -----: | -----: | ----------: | -------------: | -------------------: |
|                         0.0 |  0.988 | [0.969, 1.008] | 0.234 | 0.988 | $7.6 \times 10^{-4}$ |
|                         0.5 |  0.965 | [0.946, 0.985] | $4.9 \times 10^{-4}$ | 0.990 | $-3.8 \times 10^{-5}$ |
|                         1.0 |  0.944 | [0.926, 0.963] | $1.7 \times 10^{-8}$ | 0.992 | $-3.0 \times 10^{-4}$ |
|                         2.0 |  0.902 | [0.884, 0.921] | $2.4 \times 10^{-23}$ | 0.991 | $3.7 \times 10^{-4}$ |
|                         5.0 |  0.804 | [0.787, 0.820] | $1.5 \times 10^{-93}$ | 0.993 | $-5.3 \times 10^{-4}$ |
|                        10.0 |  0.701 | [0.686, 0.717] | $<10^{-200}$ | 1.020 | $3.2 \times 10^{-4}$ |
|                        20.0 |  0.551 | [0.539, 0.564] | $<10^{-300}$ | 1.024 | $-1.6 \times 10^{-4}$ |


Table: Example end-of-window $\mathrm{KCOR}(t)$ values from the empirical negative control (pooled/ASMR summaries), showing near-null behavior under large composition differences. (Source: `test/negative_control/out/KCOR_summary.log`) {#tbl:neg_control_summary}

| Enrollment | Dose comparison | KCOR (pooled/ASMR) | 95% CI |
|---|---|---:|---|
| 2021_24 | 1 vs 0 | 1.0097 | [0.992, 1.027] |
| 2021_24 | 2 vs 0 | 1.0213 | [1.000, 1.043] |
| 2021_24 | 2 vs 1 | 1.0115 | [0.991, 1.033] |
| 2022_06 | 1 vs 0 | 0.9858 | [0.970, 1.002] |
| 2022_06 | 2 vs 0 | 1.0756 | [1.055, 1.097] |
| 2022_06 | 2 vs 1 | 1.0911 | [1.070, 1.112] |

Table: Positive control results comparing injected hazard multipliers to detected KCOR deviations. Both scenarios show KCOR deviating from 1.0 in the expected direction, validating that the estimator can detect true effects. {#tbl:pos_control_summary}

| Scenario | Effect window | Hazard multiplier $r$ | Expected direction | Observed $\mathrm{KCOR}(t)$ at week 80 |
|---|---|---:|---|---:|
| Benefit | week 20–80 | 0.8 | < 1 | 0.825 |
| Harm | week 20–80 | 1.2 | > 1 | 1.107 |



Table: Comparison of Cox regression, shared frailty Cox models, and KCOR under selection-only and joint frailty + treatment effect scenarios. Results are from S7 simulation (joint frailty + treatment) and gamma-frailty null scenario (selection-only). Standard Cox regression produces non-null hazard ratios under selection-only conditions due to depletion dynamics. Shared frailty Cox models partially mitigate this bias but still exhibit residual non-null behavior. KCOR remains near-null under selection-only conditions and correctly detects treatment effects when temporal separability holds. {#tbl:joint_frailty_comparison}

| Scenario | True effect (r) | Cox HR | Shared frailty Cox HR | KCOR drift/year | Cox indicates null? | Frailty-Cox indicates null? | KCOR indicates null? |
|----------|-----------------|--------|----------------------|-----------------|---------------------|---------------------------|---------------------|
| Gamma-frailty null | 1.0 (null) | 0.87 | 0.94 | < 0.5% | No (HR $\neq 1$) | No (HR $\neq 1$) | Yes (flat) |
| S7 harm (r=1.2) | 1.2 | 1.18 | 1.19 | +1.8% | No (detects effect) | No (detects effect) | No (detects effect) |
| S7 benefit (r=0.8) | 0.8 | 0.83 | 0.82 | -2.1% | No (detects effect) | No (detects effect) | No (detects effect) |


Table: Simulation comparison of KCOR and alternative estimands under selection-induced non-proportional hazards. Results are summarized across simulation scenarios (null scenarios: gamma-frailty null, non-gamma frailty, contamination, sparse events; effect scenarios: injected hazard increase/decrease). KCOR remains stable under selection-only regimes, while RMST inherits depletion bias and time-varying Cox captures non-proportional hazards without normalizing selection geometry. All methods were applied to identical simulation outputs. {#tbl:comparison_estimands}

| Method | Target estimand | Deviation from null (selection-only scenarios) | Variance/instability | Interpretability notes |
|--------|----------------|--------------------------------|---------------------|------------------------|
| **KCOR** | Cumulative hazard ratio (depletion-normalized) | Near zero (median KCOR $\approx 1.0$) | Low (stable trajectory) | Stable under selection-induced depletion; normalization precedes comparison |
| **RMST** | Restricted mean survival time | Non-zero (depends on depletion strength) | Moderate (depends on depletion strength) | Summarizes survival differences that may reflect depletion rather than treatment effect; does not normalize selection geometry |
| **Cox** | Time-varying hazard ratio | Non-zero under frailty heterogeneity | Moderate (HR instability across time windows) | Improves fit to non-proportional hazards but does not normalize selection geometry; inherits depletion structure |


Table: Bootstrap coverage for KCOR uncertainty intervals. Coverage is evaluated across simulation scenarios using stratified bootstrap resampling. Nominal 95% confidence intervals are compared to empirical coverage (proportion of simulations where the true value lies within the interval). {#tbl:bootstrap_coverage}

| Scenario | Nominal coverage | Empirical coverage | Notes |
|----------|-----------------|-------------------|-------|
| Gamma-frailty null | 95% | 94.2% | Coverage evaluated under selection-only conditions |
| Injected effect (harm) | 95% | 93.8% | Coverage evaluated under known treatment effect |
| Injected effect (benefit) | 95% | 93.5% | Coverage evaluated under known treatment effect |
| Non-gamma frailty | 95% | 89.3% | Coverage under frailty misspecification |
| Sparse events | 95% | 87.6% | Coverage under reduced event counts |
