# Notation and Estimand Discipline Guidelines

## 1. Cumulative hazard notation

- Use a **single, stable symbol set** for cumulative hazards and do not reuse symbols without qualification.
- Treat the following as **three distinct objects**, each with its **own notation**:
  - Observed cumulative hazard
  - Baseline (depletion-free) cumulative hazard
  - Depletion-neutralized cumulative hazard
- Keep the **cohort index explicit everywhere**.
- Use **roman subscripts** for descriptors and **reserve numeric subscripts** for the baseline index.

Canonical examples:

$$
H_{\mathrm{obs},d}(t),\qquad H_{0,d}(t),\qquad \tilde{H}_{0,d}(t)
$$

---

## 2. Index placement and observational status

- **Avoid superscripts** for observational status.
- Do **not switch between subscript placements**.
- Always keep the **cohort index attached to the same symbol in the same position**.
- Do not let “context” carry the meaning of an index.

Disallowed example:

$$
H_d^{\mathrm{obs}}(t)
$$

---

## 3. Parameters vs. estimates

- **Distinguish sharply** between parameters and estimates.
- Use conceptual or population-level quantities **without hats**.
- Use **hatted symbols only** for fitted values returned by the quiet-window fit.
- Do **not alternate** between hatted and unhatted forms when referring to the same fitted quantity.
- Do **not drop the cohort subscript** on any parameter or estimate.

Conceptual quantities:

$$
\theta_d,\qquad k_d
$$

Fitted (quiet-window) estimates:

$$
\hat{\theta}_d,\qquad \hat{k}_d
$$

---

## 4. Gamma-frailty identity

- Use **one canonical form** for the gamma-frailty identity and **one canonical form** for its inversion throughout the manuscript.
- This applies to the main text, appendices, and figure captions.
- Do **not introduce algebraically equivalent but visually different variants**, as typographic variation causes readers to infer changes in meaning even when none exist.

Canonical mixture identity:

$$
H_{\mathrm{obs},d}(t)
=
\frac{1}{\theta_d}
\log\!\left(1+\theta_d\,H_{0,d}(t)\right)
$$

Canonical inversion:

$$
H_{0,d}(t)
=
\frac{\exp\!\left(\theta_d\,H_{\mathrm{obs},d}(t)\right)-1}{\theta_d}
$$

---

## 5. Quiet-window baseline assumptions

- If the quiet-window baseline is assumed constant:
  - State the assumption **directly** in the baseline cumulative hazard.
  - Do **not introduce auxiliary baseline-shape symbols** unless they are actually used later.
- When the baseline is constant over the fit window:
  - Keep the baseline cumulative hazard in its **simplest canonical form**.
  - Keep the cohort index attached.

Canonical constant-baseline form:

$$
H_{0,d}(t)=k_d\,t
$$

---

## 6. Estimand vs. framework naming

- Treat the **estimand** as an explicit time-indexed function.
- Reserve the **unqualified method name** (“KCOR”) for the framework.
- Use $KCOR(t)$ when referring to:
  - a curve,
  - a plotted quantity, or
  - a reported value at any time $t$.
- Do **not use “KCOR” interchangeably** to mean both the method and the time-indexed estimand.

Canonical estimand definition:

$$
KCOR(t)
=
\frac{\tilde{H}_{0,A}(t)}{\tilde{H}_{0,B}(t)}
$$

---

## 7. Notational isomorphism across the manuscript

- Make the **Methods section, figures, and appendices notationally isomorphic**.

- If a plot shows **observed cumulative hazards**, label axes and legends with:

$$
H_{\mathrm{obs},d}(t)
$$

- If a plot shows **depletion-neutralized quantities**, label axes and legends with:

$$
\tilde{H}_{0,d}(t)
$$

- Do **not label axes** with:

$$
H_0
$$

unless:
- the plotted object is explicitly defined as such in the main text, and
- the meaning remains identical throughout the manuscript.

- Do **not introduce alternate indexing conventions or shorthand** in appendices.
- Every symbol used outside the Methods section must match the **canonical definitions exactly**.
