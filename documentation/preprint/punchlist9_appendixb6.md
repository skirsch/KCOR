Great questions. I’ll do this in three parts:

1. **Paste-ready Appendix B.6 text (verbatim)**
2. **Why it’s called S7 (and whether that’s “standard”)**
3. **Whether the quiet-period / interpretation caveat is fully covered — and one optional strengthening sentence**

---

## 1) Appendix B.6 — paste-ready text (verbatim)

**Placement:** Appendix B (Simulation Methods), **after** Appendix B.5.

You can paste this exactly as written.

---

### **Appendix B.6. Joint frailty and treatment-effect simulation (S7)**

This simulation evaluates KCOR under conditions in which **both selection-induced depletion (frailty heterogeneity)** and a **true treatment effect (harm or benefit)** are present simultaneously. The purpose is to assess whether KCOR can (i) correctly identify and neutralize frailty-driven curvature using a quiet period and (ii) detect a true treatment effect outside that period without confounding the two mechanisms.

#### **Design**

Two fixed cohorts are generated with identical baseline hazards but differing frailty variance. Individual hazards are multiplicatively scaled by a latent frailty term drawn from a gamma distribution with unit mean and cohort-specific variance. A treatment effect is then injected over a prespecified time window that does not overlap the quiet period used for frailty estimation.

Formally, individual hazards are generated as

[
h_i(t) = z_i , h_0(t) , r(t),
]

where (z_i) is individual frailty, (h_0(t)) is a shared baseline hazard, and (r(t)) is a time-localized multiplicative treatment effect applied to one cohort only.

#### **Frailty structure**

* Cohort 0: (z \sim \text{Gamma}(\theta_0))
* Cohort 1: (z \sim \text{Gamma}(\theta_1)), with (\theta_1 \neq \theta_0)

Frailty distributions are normalized to unit mean, differing only in variance, thereby inducing different depletion dynamics and cumulative-hazard curvature across cohorts in the absence of any treatment effect.

#### **Treatment effect**

A known treatment effect is applied to Cohort 1 during a finite window ([t_{\text{on}}, t_{\text{off}}]). Three effect shapes are considered:

1. Step change (constant multiplicative factor),
2. Linear ramp,
3. Smooth pulse (“bump”).

Both harmful ((r(t) > 1)) and protective ((r(t) < 1)) effects are evaluated. The treatment window is chosen to lie strictly outside the quiet period used for frailty estimation.

#### **Quiet period and estimation**

Frailty parameters are estimated independently for each cohort using observed cumulative hazards over a prespecified quiet window ([t_q^{\text{start}}, t_q^{\text{end}}]) during which (r(t)=1) by construction. KCOR normalization is then applied to the full time horizon using these estimated parameters.

This design enforces **temporal separability** between selection-induced depletion and treatment effects.

#### **Evaluation criteria**

The simulation is considered successful if:

1. KCOR remains approximately flat and near unity during the quiet window,
2. KCOR deviates in the correct direction and magnitude during the treatment window,
3. Fit diagnostics (e.g., residual curvature, post-normalization linearity) remain stable outside intentionally violated scenarios.

An additional stress-test variant intentionally overlaps the treatment window with the quiet period. In this case, KCOR diagnostics degrade and normalized trajectories fail to stabilize, correctly signaling violation of the identifiability assumptions rather than producing spurious treatment effects.

#### **Interpretation**

This simulation demonstrates that when selection-induced depletion and treatment effects are temporally separable, KCOR can disentangle the two mechanisms: frailty parameters are identified from quiet-period curvature, and true treatment effects manifest as deviations from unity outside that window. When separability is violated, KCOR does not silently misattribute effects; instead, diagnostics flag reduced interpretability.

---

## 2) Why it’s called **S7** — and is that “standard”?

Short answer: **the numbering is conventional, not an industry standard term**.

Longer explanation:

* In methods papers, simulations are commonly labeled **S1, S2, …** as shorthand for *Scenario 1, Scenario 2*, etc.
* You already have **S1–S6** in the paper (null, positive control, misspecification, sparsity, etc.).
* This new test is simply the **seventh distinct scenario**, hence **S7**.

There is **no external meaning** to “S7” beyond:

> “the seventh simulation scenario in this paper.”

Reviewers are completely comfortable with this convention. It does **not** imply any industry framework or borrowed nomenclature, and it avoids inventing a new name that would draw unnecessary scrutiny.

If you prefer, you could also write:

> “Simulation S7 (joint frailty + treatment)”

—but the label **S7** alone is fine and standard in practice.

---

## 3) Quiet period, enrollment timing, and careful interpretation

**Is that fully covered?**

### Short answer

**Yes — substantively it is covered**, but I recommend **one explicit reinforcement sentence** to make it reviewer-proof and align exactly with what you told VSRF.

### Where it’s already covered

The Appendix B.6 text explicitly states:

* Frailty is estimated **only during a quiet period**
* Treatment effects are **outside** that window
* Overlap cases are tested and **fail diagnostically**
* Interpretation depends on **temporal separability**

This matches your VSRF explanation precisely:

> “If there’s no quiet period after enrollment, you need to interpret the results carefully.”

### Optional strengthening sentence (recommended)

To make this impossible to miss, add **one sentence** to **§2.1.1 Assumptions and identifiability conditions** (Methods main text):

> *In settings without a clear post-enrollment quiet period prior to major epidemic pressure, KCOR estimates should be interpreted as descriptive of cumulative differences rather than as isolated treatment effects, and diagnostics should be examined for evidence of assumption violation.*

This does three things:

* Aligns exactly with your VSRF statement
* Preempts the “but what if there’s no quiet period?” objection
* Shows methodological humility (very important for peer review)

---

## Bottom line

* **Appendix B.6** now directly answers Retsef’s strongest critique.
* **S7** is a standard, neutral scenario label — not jargon.
* Your quiet-period caveat **is included**, and with one optional sentence it becomes unmissable and reviewer-safe.
* Nothing here implies Cox, random effects regression, or hidden proportional hazards assumptions.

If you want next, I can:

* Draft the **reviewer rebuttal paragraph** specifically responding to “this is just Cox with frailty”
* Help you tune the **diagnostics language** so it sounds rigorous rather than promotional
* Review your **Abstract** to ensure none of these clarifications are contradicted implicitly

Just tell me the next step.
