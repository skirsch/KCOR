Perfect — here is a **Cursor-ready punchlist** that you can hand over verbatim.
It specifies **where**, **what**, and **how much**, and includes **drop-in text** so Cursor doesn’t invent or over-elaborate.

---

## Cursor punchlist — Add “Data requirements & feasibility” section (Baral feedback)

### Goal

Add a short, explicit subsection that clarifies **what data KCOR requires, what it does not require, and which study designs are feasible**, without changing scope or claims.

---

### 1. Location (DO THIS FIRST)

**Insert a new subsection at the end of the Methods section**, after the estimator and diagnostics are fully defined and **before Results**.

**Exact placement instruction:**

> Insert a new subsection immediately after Section 2.13 (Reproducibility and Computational Implementation) and before Section 3 (Results).

This positions the section as **applicability / feasibility**, not limitations.

---

### 2. Section title

Use this exact title (do not invent alternatives):

```markdown
### 2.14 Data requirements and feasible study designs
```

---

### 3. Insert the following text verbatim

Cursor: **paste exactly**, no rewriting unless needed for formatting consistency.

```markdown
### 2.14 Data requirements and feasible study designs

KCOR is designed for settings where outcomes are ascertained repeatedly over time but individual-level covariates, visit schedules, or counterfactual exposure histories are unavailable or unreliable. This section clarifies the data structures required for valid application, as well as designs for which the method is not appropriate.

**Minimum data requirements.** KCOR requires (i) a well-defined cohort entry or enrollment rule, (ii) repeated outcome ascertainment over calendar or follow-up time (which may be passive), (iii) sufficient event counts to estimate cumulative hazards with reasonable stability, and (iv) the presence of at least one diagnostically identifiable “quiet” window during which baseline risk is approximately stable. Individual-level covariates, treatment assignment models, or visit-based follow-up schedules are not required.

**What KCOR does not require.** Unlike causal estimators, interrupted time-series methods, synthetic controls, or digital twin approaches, KCOR does not require exchangeability assumptions, individual-level confounder measurement, continuous exposure tracking, or explicit modeling of counterfactual untreated trajectories. The method operates entirely on aggregated cumulative hazard information and is therefore compatible with registries and administrative systems where outcomes are captured far more frequently than exposures or visits.

**Well-suited data sources and designs.** KCOR is particularly well matched to vital statistics–linked cohorts, administrative or registry-based studies with passive outcome capture, program enrollments with irregular follow-up but reliable endpoints, and population-based datasets where cohorts are defined by eligibility, vulnerability, or sociodemographic characteristics rather than repeated clinical encounters.

**Designs where KCOR is not appropriate.** KCOR is not suitable for settings with extremely sparse events, rapidly shifting baseline hazards with no diagnostically quiet interval, highly fluid cohorts with frequent switching between exposure groups, or outcome ascertainment that is strictly conditional on clinic visits. In such cases, the required identifiability diagnostics will fail and results should not be reported.
```

---

### 4. Guardrails (important — tell Cursor what *not* to do)

Add this instruction explicitly:

> Do not introduce causal language, counterfactual claims, or VE interpretation in this section. Do not reference specific vaccines or datasets. Do not add figures or tables.

