Below is **paste-ready replacement text** for **§§2.1–2.4**. It’s written to (i) be tighter, (ii) avoid over-claiming causality, and (iii) cleanly separate **estimand / identification / diagnostics**.


---

## 2.1 Conceptual framework and estimand

Retrospective cohort differences can arise from two qualitatively different components:

* **Level differences**: cohort hazards differ by an approximately time-stable multiplicative factor (or, equivalently, cumulative hazards have different slopes but similar shape).
* **Depletion (curvature) differences**: cohort hazards evolve differently over time because cohorts differ in latent heterogeneity and are **selectively depleted** at different rates.

KCOR targets the second failure mode. Under latent frailty heterogeneity, high-risk individuals die earlier, so the surviving risk set becomes progressively “healthier.” This induces **downward curvature** (deceleration) in cohort hazards and corresponding concavity in cumulative-hazard space, even when individual-level hazards are simple and even under a true null treatment effect. When selection concentrates frailty heterogeneity differently across cohorts, the resulting curvature differences produce strong non-proportional hazards and can drive misleading contrasts for estimands that condition on the evolving risk set.

KCOR’s strategy is therefore:

1. **Estimate the cohort-specific depletion geometry** (via curvature) during prespecified epidemiologically quiet periods.
2. **Map observed cumulative hazards into a depletion-neutralized space** by inverting that geometry.
3. **Compare cohorts only after normalization**, using a cumulative-hazard ratio defined on the depletion-neutralized scale.

### 2.1.1 Target estimand

Let (\tilde{H}_{0,d}(t)) denote the **depletion-neutralized baseline cumulative hazard** for cohort (d) at event time (t) since enrollment (Table @tbl:notation). For two cohorts (A) and (B), KCOR is defined as

[
\mathrm{KCOR}(t) ;=; \frac{\tilde{H}_{0,A}(t)}{\tilde{H}_{0,B}(t)}.
]
{#eq:kcor-estimand}

KCOR((t)) is a **cumulative outcome contrast** after removal of curvature attributed to selection-induced depletion under the working frailty model. The estimand is defined regardless of whether it has a causal interpretation.

### 2.1.2 Identification versus diagnostics

KCOR is presented here as a **normalization-and-comparison framework**, not as a general causal estimator under unmeasured confounding. A causal interpretation of (\mathrm{KCOR}(t)) requires additional substantive conditions (e.g., that the quiet-window curvature is dominated by selection-induced depletion rather than cohort-differential external shocks). Because these conditions are inherently dataset- and design-dependent, KCOR emphasizes **diagnostic enforcement**: when assumptions required for interpretable normalization are not supported, the method should signal this through degraded fit, residual structure, or instability to window perturbations rather than silently producing a “corrected” contrast.

Operationally, interpretability of a KCOR trajectory is assessed via prespecified checks (Appendix D), including:

* stability of ((\hat{k}_d,\hat{\theta}_d)) to small quiet-window perturbations,
* approximate linearity of (\tilde{H}_{0,d}(t)) within the quiet window,
* absence of systematic residual structure in cumulative-hazard space.

---

## 2.2 Cohort construction

KCOR is defined for **fixed cohorts at enrollment**. Required inputs are minimal: enrollment date(s), event date, and optionally birth date (or year-of-birth) for age stratification. Analyses proceed in discrete event time (t) (e.g., weeks) measured since cohort enrollment.

Cohorts are assigned by intervention state at the start of the enrollment interval. In the primary estimand:

* **No post-enrollment switching** is allowed (individuals remain in their enrollment cohort),
* **No censoring** is applied (other than administrative end of follow-up),
* analyses are performed on the resulting fixed risk sets.

This fixed-cohort design is intentional. It avoids immortal-time artifacts and prevents outcome-driven switching rules from creating time-dependent selection that is difficult to diagnose under minimal covariate availability. Extensions that allow switching or censoring are treated as sensitivity analyses (§5.2) because they change the estimand and introduce additional identification requirements.

Throughout this manuscript the failure event is **all-cause mortality**. KCOR therefore targets cumulative mortality hazards and is not framed as a cause-specific competing-risks analysis.

---

## 2.3 Hazard estimation and cumulative hazards in discrete time

Let (D_d(t)) denote events during interval (t) in cohort (d), and (N_d(t)) the number at risk at the start of interval (t). We compute the discrete-time cohort hazard as

[
h_{\mathrm{obs},d}(t) ;=; -\ln!\left(1 - \frac{D_d(t)}{N_d(t)}\right).
]
{#eq:hazard-discrete}

This transform is standard: it maps an interval event probability into a continuous-time equivalent hazard under a piecewise-constant hazard assumption. For rare events, (h_{\mathrm{obs},d}(t) \approx D_d(t)/N_d(t)), but the log form remains accurate and stable when weekly risks are not negligible.

Observed cumulative hazards are accumulated over event time after an optional stabilization skip (§2.7):

[
H_{\mathrm{obs},d}(t) ;=; \sum_{s \le t} h_d^{\mathrm{eff}}(s),
\qquad \Delta t = 1.
]
{#eq:cumhazard-observed}

Discrete binning accommodates tied events and aggregated registry releases. Bin width is chosen based on diagnostic stability (e.g., smoothness and sufficient counts per bin) rather than temporal resolution alone.

---

## 2.4 Selection model: gamma frailty and depletion normalization

### 2.4.1 Individual hazards with multiplicative frailty

Within cohort (d), individual (i) is modeled as having hazard

[
h_{i,d}(t) ;=; z_{i,d},h_{0,d}(t),
\qquad
z_{i,d} \sim \mathrm{Gamma}(\mathrm{mean}=1,\ \mathrm{var}=\theta_d).
]
{#eq:individual-hazard-frailty}

Here (h_{0,d}(t)) is the cohort’s depletion-neutralized baseline hazard and (z_{i,d}) is a latent multiplicative frailty term. The frailty variance (\theta_d) governs the strength of depletion-induced curvature: larger (\theta_d) yields stronger deceleration at the cohort level due to faster early depletion of high-frailty individuals.

Gamma frailty is used because it yields a closed-form link between observed and baseline cumulative hazards via the Laplace transform [@vaupel1979]. In KCOR, gamma frailty is a **working geometric model** for depletion normalization, not a claim of biological truth. Adequacy is evaluated empirically via fit quality, post-normalization linearity, and stability diagnostics.

### 2.4.2 Gamma-frailty identity and inversion

Let

[
H_{0,d}(t) = \int_0^t h_{0,d}(s),ds
]
{#eq:baseline-cumhazard}

denote the baseline cumulative hazard. Integrating over gamma frailty yields the gamma-frailty identity

[
H_{\mathrm{obs},d}(t) ;=; \frac{1}{\theta_d},\log!\left(1 + \theta_d,H_{0,d}(t)\right),
]
{#eq:gamma-frailty-identity}

which can be inverted exactly as

[
H_{0,d}(t) ;=; \frac{\exp!\left(\theta_d,H_{\mathrm{obs},d}(t)\right) - 1}{\theta_d}.
]
{#eq:gamma-frailty-inversion}

This inversion is the **normalization operator**: given an estimate (\hat{\theta}_d), it maps the observed cumulative hazard (H_{\mathrm{obs},d}(t)) into a depletion-neutralized cumulative hazard scale.

### 2.4.3 Baseline shape used for frailty identification

To identify (\theta_d), KCOR fits the gamma-frailty model within prespecified epidemiologically quiet periods. In the reference specification, the baseline hazard is taken to be constant over the fit window:

[
h_{0,d}(t)=k_d,
\qquad
H_{0,d}(t)=k_d,t.
]
{#eq:baseline-shape-default}

This choice intentionally minimizes degrees of freedom: during a quiet window, curvature is forced to be explained by depletion (via (\theta_d)) rather than by introducing time-varying baseline hazard terms. If the observed cumulative hazard is near-linear over the fit window, the model naturally collapses toward (\hat{\theta}_d \approx 0), signaling weak or absent detectable depletion curvature for that cohort over that window.

### 2.4.4 Quiet-window validity as the key dataset-specific requirement

Frailty parameters are estimated using only bins whose corresponding calendar weeks lie inside a prespecified quiet window (defined in ISO-week space). A window is acceptable only if diagnostics indicate (i) good fit in cumulative-hazard space, (ii) post-normalization linearity within the window, and (iii) stability of ((\hat{k}_d,\hat{\theta}_d)) to small boundary perturbations. If no candidate window passes, KCOR is treated as not identified for that analysis rather than producing a potentially misleading normalized contrast.

---

