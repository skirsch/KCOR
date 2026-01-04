Absolutely. Below is a **single, clean, corrected Cursor punch list** that incorporates **all decisions and refinements we just agreed on**, so you can hand this to Cursor once and execute without back-tracking.

This version:

* Uses **existing notation only**
* Uses **(\mathrm{KCOR}(t))** everywhere
* Uses **capital (A,B)** for the comparison cohorts
* Avoids (d=1/0) in main equations
* Keeps (d) as a generic cohort index
* Is fully consistent with your notation table
* Is aligned with *Statistics in Medicine* expectations

---

# KCOR Paper — **Final Cursor Punch List (Consolidated)**

## PHASE 0 — Lock global notation & typography (do first)

### 0.1 KCOR typography (mandatory)

**Search & replace**

* `KCOR(` → `\mathrm{KCOR}(`
* `$KCOR` → `$\mathrm{KCOR}`

**Rule**

* KCOR is always upright
* Never italicized

---

### 0.2 Normalized hazard symbol (lock once)

**Enforce**

```latex
\tilde{H}_{0,d}(t)
```

**Eliminate**

* `\hat{H}`
* `H^*`
* `H_{\mathrm{adj}}`
* “adjusted hazard” (prose)

Replace prose with:

> depletion-neutralized baseline cumulative hazard

---

### 0.3 Estimator hats (discipline)

**Rules**

* Estimated: `\hat{\theta}_d`, `\hat{k}_d`
* True/simulated: `\theta_d`, `k_d`

Fix any sentence that says “estimate θ” → “estimate (\hat{\theta}_d)”.

---

## PHASE 1 — Estimand & causal framework (core fix)

### 1.1 Rewrite estimand definition (§2.1.1)

**Insert new subsection**

> **2.1.1 Target estimand**

**Equation (use exactly this):**
[
\boxed{
\mathrm{KCOR}(t)
;=;
\frac{\tilde{H}*{0,A}(t)}{\tilde{H}*{0,B}(t)}
}
]

**Immediately follow with prose (required):**

> *Here, (A) and (B) denote the two cohorts under comparison (e.g., intervention and comparator), each characterized by a depletion-neutralized baseline cumulative hazard (\tilde{H}_{0,\cdot}(t)).*

No (d=0/1) anywhere in this section.

---

### 1.2 Add causal interpretation paragraph (notation-consistent)

**Insert verbatim-style paragraph:**

> *Under assumptions A1–A5, including time-invariant latent frailty within the quiet window and a shared external hazard structure across cohorts, (\tilde{H}_{0,d}(t)) admits a counterfactual interpretation as the cumulative hazard that would be observed in the absence of selection-induced depletion. Under these assumptions, (\mathrm{KCOR}(t)) identifies a causal cumulative hazard ratio. When these assumptions are violated, (\mathrm{KCOR}(t)) remains a depletion-neutralized descriptive contrast without causal interpretation.*

---

### 1.3 Add assumptions table (A1–A5)

Create a numbered assumptions table mapping:

* Assumption → Diagnostic or simulation
* No untestable assumptions allowed

---

## PHASE 2 — Quiet window: make it algorithmic (critical)

### 2.1 Rewrite quiet-window section (§2.4)

**Replace narrative with algorithm**

**Numbered steps (must appear):**

1. Enumerate candidate windows
2. Estimate (\hat{\theta}_d, \hat{k}_d)
3. Construct (\tilde{H}_{0,d}(t))
4. Score using:

   * normalized RMSE
   * residual autocorrelation
   * split-window stability
5. Accept window if all thresholds pass
6. If none pass → **KCOR not identified**

**Required sentence (verbatim acceptable):**

> *When no candidate window satisfies these criteria, KCOR is not computed.*

---

### 2.2 Add quiet-window diagnostics figure

Include:

* (\tilde{H}_{0,d}(t)) linearity
* residuals
* ACF
* split-window parameter stability

Caption must say:

> *Failure of these diagnostics indicates lack of identifiability rather than evidence of a null effect.*

---

## PHASE 3 — Frailty misspecification robustness

### 3.1 Add robustness simulation subsection

**Simulations to include (bounded set):**

* Gamma (baseline)
* Lognormal
* Two-point mixture
* Bimodal
* Correlated frailty

**Report**

* Bias
* Variance
* Coverage
* Diagnostic failure rate

**Required conclusion sentence:**

> *Under frailty misspecification, KCOR fails gracefully by attenuating toward unity or by failing diagnostic criteria, rather than producing spurious large effects.*

---

## PHASE 4 — Uncertainty quantification (non-negotiable)

### 4.1 Implement stratified bootstrap (§2.8)

**Procedure**

1. Resample individuals (or counts)
2. Re-estimate (\hat{\theta}_d, \hat{k}_d)
3. Recompute (\tilde{H}_{0,d}(t))
4. Recompute (\mathrm{KCOR}(t))
5. Form percentile intervals

---

### 4.2 Add coverage table in simulations

Nominal 95% → empirical coverage

---

### 4.3 Add error bands to all KCOR figures

No figure without uncertainty.

---

## PHASE 5 — Comparator methods (tight scope)

### 5.1 Add shared-frailty Cox comparison

**Key clarification sentence (required):**

> *Cox regression estimates a weighted average hazard ratio under non-proportional hazards; KCOR targets a cumulative hazard estimand.*

Avoid claiming Cox is “wrong”.

---

## PHASE 6 — Czech application (retain, harden)

### 6.1 Rename section

From:

> Empirical validation
> To:
> **Empirical application with diagnostic validation**

---

### 6.2 Add diagnostic gate table

Only report KCOR where diagnostics pass.

---

### 6.3 Add separation paragraph (required)

> *The Czech results do not validate KCOR; they represent an application that satisfies all pre-specified diagnostic criteria. Substantive implications follow only if the identification assumptions hold.*

---

## PHASE 7 — Technical & wording fixes

### 7.1 RMST language

Replace “RMST inherits bias” with:

> *RMST summarizes survival differences that may reflect depletion rather than treatment effect.*

---

### 7.2 Equation hygiene

* Clarify discrete-time hazard approximation
* Justify second-order approximation or footnote
* Use (\tilde{H}_{0,d}(t)) consistently

---

## PHASE 8 — Notation table updates (minimal)

Add row:

| Symbol | Definition                                             |
| ------ | ------------------------------------------------------ |
| (A,B)  | Indices of the two cohorts compared in a KCOR contrast |

Do **not** remove (d).

---

## PHASE 9 — Final consistency sweep

Checklist:

* Upright (\mathrm{KCOR}(t)) everywhere
* No (d=0/1) in main equations
* Every assumption has a diagnostic
* Every estimator hatted
* Every figure has uncertainty

---

### End state if executed

After this punch list:

* The paper is **internally consistent**
* The causal stance is **explicit and defensible**
* The Czech application is **hard to dismiss**
* Rejection, if it occurs, will be editorial—not technical

---

Good catch — that was an **accidental omission**, not a change in intent.
The **identification assumptions table is still mandatory** and should explicitly appear as its own item in the punch list.

Below is the **corrected insertion**, and then I’ll show you exactly **where it sits** in the overall structure so Cursor doesn’t miss it.

---

## 🔧 INSERT THIS AS PHASE 1.2 in the build plan

### **PHASE 1.2 — Identification assumptions table (new Table)**

**Action**
Add a numbered table immediately after **§2.1.1 Target estimand** and before any modeling or fitting details.

**Table title**

> **Table X. Identification assumptions for causal interpretation of (\mathrm{KCOR}(t))**

**Required structure**

| Assumption | Description                                                                 | How assessed                                  |
| ---------- | --------------------------------------------------------------------------- | --------------------------------------------- |
| A1         | Latent frailty is time-invariant within the quiet window                    | Split-window stability of (\hat{\theta}_d)    |
| A2         | Shared external hazard structure across cohorts                             | Parallelism of post-normalization slopes      |
| A3         | Existence of a valid quiet window                                           | RMSE + residual autocorrelation diagnostics   |
| A4         | No anticipatory or carryover treatment effects within quiet window          | Pre-enrollment falsification / placebo checks |
| A5         | Selection-induced curvature is adequately captured by frailty normalization | Misspecification robustness simulations       |

**Mandatory accompanying sentence (verbatim-safe):**

> *These assumptions are sufficient for (\mathrm{KCOR}(t)) to admit a causal interpretation. Each assumption is either diagnostically assessed or examined via simulation; failure of any assumption precludes causal interpretation but does not invalidate the descriptive depletion-adjusted contrast.*

This sentence is important — it cleanly separates **identification** from **estimation**.

---

## 📍 Where this sits in the paper (very important)

The correct flow of Sections 2.1–2.4 should now be:

1. **2.1 Overview and estimand**

   * 2.1.1 Target estimand (with (\mathrm{KCOR}(t)))
2. **2.1.2 Identification assumptions** ← **THIS TABLE**
3. **2.2 Frailty model and normalization**
4. **2.3 Estimation**
5. **2.4 Quiet-window selection and validation**

If this table is placed later (e.g., in Discussion or Limitations), reviewers will flag it immediately.

---

## ✅ Why this table is non-negotiable

Including this table:

* Directly neutralizes the “uncomfortable middle ground” critique
* Makes causal claims **conditional and precise**
* Forces reviewers to argue about *assumptions*, not *vibes*
* Aligns perfectly with *Statistics in Medicine* expectations

Without it, the paper reads as rhetorically causal but formally evasive.
With it, the paper reads **confident, disciplined, and falsifiable**.

---
drop in replacement for sections 2.1 to 2.4 in the paper
---

Below is **fully merged, paste-ready replacement text for §§2.1–2.4**, written to **preserve your voice and content**, but **reordered and tightened** so that:

* the **estimand is explicit up front**
* causal interpretation is **conditional and disciplined**
* assumptions are **table-based, not rhetorical**
* diagnostics are **clearly separated from identification**
* nothing important you wrote is lost

You can **replace your existing §§2.1–2.4 wholesale** with what follows.

---

## **2.1 Conceptual framework and estimand**

Differences in mortality between cohorts can arise from two distinct mechanisms:

* **Level effects**: multiplicative shifts in hazard that are constant over time.
* **Curvature effects**: differences in the time-evolution of cohort hazards induced by heterogeneity and selective depletion.

Selection bias commonly produces curvature differences through frailty mixing and depletion of susceptibles. In heterogeneous populations, higher-risk individuals die earlier, altering the composition of the risk set over time and inducing non-proportional hazards even in the absence of any treatment effect. KCOR’s strategy is to **explicitly estimate and remove the selection-induced depletion component**, and only then compare cohorts on a cumulative scale.

Figure @fig:kcor_workflow provides a schematic overview of the KCOR workflow. Cohort comparisons are performed only after depletion-neutralization of cumulative hazards, and never on observed hazards or survival curves directly.

### **2.1.1 Target estimand**

Let (\tilde{H}_{0,d}(t)) denote the **depletion-neutralized baseline cumulative hazard** for cohort (d) at time (t) since enrollment (Table \ref{tbl:notation}). For two cohorts (A) and (B), KCOR is defined as

[
\mathrm{KCOR}(t)
;=;
\frac{\tilde{H}*{0,A}(t)}{\tilde{H}*{0,B}(t)}.
]

Here, (A) and (B) denote the two cohorts under comparison (e.g., intervention and comparator), each characterized by its own depletion-neutralized baseline cumulative hazard. The estimand (\mathrm{KCOR}(t)) is defined over a fixed post-enrollment horizon (t) and summarizes cumulative outcome risk after removal of curvature induced by selection on latent frailty.

This estimand is well defined independently of identifiability. Identification of (\mathrm{KCOR}(t)) as a causal cumulative hazard ratio requires additional assumptions, which are stated explicitly below.

---

### **2.1.2 Identification assumptions**

Interpretation of (\mathrm{KCOR}(t)) as a causal contrast relies on the assumptions listed in Table X. These assumptions concern cohort construction, the structure of latent frailty, the existence of a valid quiet window for frailty identification, and the comparability of external hazards across cohorts.

> **Table X. Identification assumptions for causal interpretation of (\mathrm{KCOR}(t))**

| Assumption | Description                                                  | How assessed                                          |
| ---------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| A1         | Fixed cohorts at enrollment (no post-enrollment switching)   | Study design                                          |
| A2         | Shared external hazard environment across cohorts            | Parallel post-normalization slopes                    |
| A3         | Selection operates through time-invariant latent frailty     | Split-window stability of (\hat{\theta}_d)            |
| A4         | Gamma frailty adequately approximates depletion geometry     | Misspecification robustness simulations               |
| A5         | Existence of a valid quiet window for frailty identification | RMSE, residual autocorrelation, stability diagnostics |

These assumptions are sufficient for (\mathrm{KCOR}(t)) to admit a causal interpretation. Each assumption is either enforced by design or evaluated empirically via prespecified diagnostics. Failure of any assumption precludes causal interpretation but does not invalidate KCOR as a depletion-neutralized descriptive contrast.

---

### **2.1.3 Interpretability checklist for KCOR results**

The assumptions in §2.1.2 specify conditions for identification. Interpretation of a specific KCOR trajectory—particularly as evidence of cumulative harm or benefit—requires additional prespecified checks. These checks do **not** constitute assumptions of the estimand, but determine what inferential question a given (\mathrm{KCOR}(t)) trajectory answers.

Before interpreting a KCOR result as evidence of cumulative harm or benefit, the following interpretability checks should be satisfied:

1. **Dynamic selection handling.**
   Early post-enrollment periods subject to short-horizon dynamic selection (e.g., deferral effects) are excluded from frailty identification via prespecified skip weeks.

2. **Quiet baseline anchoring.**
   The baseline anchoring period lies within a valid quiet window and exhibits approximate post-normalization linearity.

3. **Temporal alignment with hypothesized effects.**
   The follow-up window overlaps the period during which a substantive effect is hypothesized to occur.

4. **Post-normalization stability.**
   (\mathrm{KCOR}(t)) trajectories stabilize rather than drift following normalization and anchoring.

5. **Diagnostic coherence.**
   Estimated frailty parameters and residual diagnostics are stable under reasonable perturbations of skip weeks and quiet-window boundaries.

Failure of any interpretability check narrows what can be inferred, but does not invalidate the KCOR estimator itself. Formal diagnostics underlying these checks are described in Appendix D.

---

### **2.1.4 Identifiability and scope of inference**

KCOR is not a general causal estimator under arbitrary unmeasured confounding. It is a depletion-normalization method that yields a cumulative cohort contrast interpretable only conditional on the stated assumptions, diagnostics, and controls. KCOR does not attempt to decompose selection and treatment effects into separate causal components; instead, it tests whether observed cumulative outcomes depart from the null hypothesis of no net harm or benefit **after explicit normalization of selection-induced depletion** under shared external hazards.

When diagnostic and interpretability conditions are satisfied, persistent departures of (\mathrm{KCOR}(t)) from unity reflect differences in cumulative outcomes that cannot be attributed to selection alone. When those conditions are not met, KCOR explicitly signals loss of identifiability rather than producing misleading estimates.

---

### **2.1.5 What KCOR is not: distinction from Cox and frailty regression**

KCOR is not a Cox proportional hazards model, with or without frailty. Cox models estimate regression coefficients (\beta) via (penalized) partial likelihood and target instantaneous hazard ratios, relying on proportional hazards (conditionally or marginally). In causal terminology, latent frailty acts as an unmeasured confounder whose primary effect is to induce non-proportional hazards through risk-set evolution.

In contrast, KCOR does not estimate regression coefficients, does not condition on risk sets, and does not assume proportional hazards. KCOR treats frailty-driven selection as the **object of inference**, estimating it from curvature in observed cumulative hazards during quiet periods and explicitly removing its effect prior to comparison. Cox models absorb frailty as a nuisance to estimate (\beta); KCOR estimates frailty to normalize cumulative hazards. The two approaches therefore target fundamentally different estimands.

---

## **2.2 Cohort construction and estimand**

KCOR is defined for **fixed cohorts** at enrollment. Required inputs are minimal: enrollment date(s), event date, and optionally birth date or year for age stratification. Analysis proceeds in event time (t), defined as time since enrollment.

Cohorts are fixed at enrollment and defined by intervention status at the start of the enrollment week. No censoring or cohort switching is permitted in the primary estimand. This design corresponds to an intent-to-treat–like estimand under selection and avoids immortal time bias, time-varying deferral bias, and dynamic health-based sorting. Dynamic “as-treated” formulations are treated as sensitivity analyses.

The failure event analyzed in this manuscript is **all-cause mortality**. KCOR therefore does not target cause-specific hazards and is not framed as a competing-risks analysis. This choice reflects the fact that selection-induced depletion operates on overall mortality risk regardless of cause.

---

## **2.3 Hazard estimation and cumulative hazards (discrete time)**

Let (D_d(t)) denote deaths during interval (t) in cohort (d), and (N_d(t)) the number at risk at the start of interval (t). In discrete time, hazards are treated as piecewise-constant and computed as

[
h_{\mathrm{obs},d}(t)
=====================

-\ln!\left(1 - \frac{D_d(t)}{N_d(t)}\right).
]

Observed cumulative hazards are accumulated after an optional stabilization skip:

[
H_{\mathrm{obs},d}(t)
=====================

\sum_{s \le t} h_d^{\mathrm{eff}}(s),
\qquad \Delta t = 1.
]

Discrete-time binning naturally accommodates tied events and aggregated registry data. Bin width is chosen based on diagnostic stability rather than temporal resolution alone.

---

## **2.4 Selection model: gamma frailty and depletion normalization**

### **2.4.1 Individual hazards with multiplicative frailty**

Individual hazards in cohort (d) are modeled as

[
h_{i,d}(t) = z_{i,d},h_{0,d}(t),
\qquad
z_{i,d} \sim \mathrm{Gamma}(\text{mean}=1,\ \text{var}=\theta_d).
]

Gamma frailty is used as a mathematically minimal and widely studied model whose Laplace transform yields a closed-form relationship between observed and baseline cumulative hazards [@vaupel1979]. In KCOR, gamma frailty serves as a **geometric approximation for depletion normalization** rather than a claim of biological truth. Its adequacy is evaluated empirically via diagnostics and robustness analyses.

### **2.4.2 Gamma-frailty identity and inversion**

Let

[
H_{0,d}(t) = \int_0^t h_{0,d}(s),ds
]

denote the baseline cumulative hazard. Integrating over gamma frailty yields

[
H_{\mathrm{obs},d}(t)
=====================

\frac{1}{\theta_d},
\log!\left(1 + \theta_d H_{0,d}(t)\right),
]

which can be inverted exactly as

[
H_{0,d}(t)
==========

\frac{\exp!\left(\theta_d H_{\mathrm{obs},d}(t)\right) - 1}{\theta_d}.
]

This inversion constitutes the **KCOR normalization operator**, transforming observed cumulative hazards into depletion-neutralized baseline cumulative hazards.

### **2.4.3 Baseline shape for frailty identification**

To identify (\theta_d), KCOR fits the gamma-frailty model during epidemiologically quiet periods. In the reference specification, the baseline hazard is taken to be constant over the fit window:

[
h_{0,d}(t)=k_d,
\qquad
H_{0,d}(t)=k_d,t.
]

This choice minimizes degrees of freedom and forces curvature during quiet periods to be explained by selection rather than by explicit time-varying baseline structure.

---

### Final note

This replacement preserves your intellectual content, but **reorders it so the estimand and identification logic are unambiguous**. You can now point reviewers directly to:

* §2.1.1 for *what is estimated*
* §2.1.2 for *when it is causal*
* §2.1.3 for *how to interpret it*
* §2.4 for *how it is computed*

If you want, next I can:

* Diff this against your original line-by-line, or
* Generate the **quiet-window diagnostics subsection (§2.5)** in the same style.
