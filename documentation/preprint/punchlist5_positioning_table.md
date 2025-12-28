# Positioning KCOR Among Retrospective Methods (Corrected)

## Purpose
This table clarifies the **inferential role of KCOR** and corrects common misinterpretations that frame it as a standalone frailty-normalization or diagnostic-only method. KCOR is a **complete comparison system**: normalization, comparison, and diagnostics are inseparable.

---

## Table: Inferential Role of KCOR Relative to Established Methods

| Method family | Primary estimand (typical output) | Handles selection-induced depletion curvature? | What it requires | Primary failure mode |
|---|---|---:|---|---|
| Kaplan–Meier / Cox PH | Instantaneous HR; survival differences under PH | No | Exchangeability; proportional hazards; adequate covariates | Non-PH from latent selection yields misleading HRs |
| Cox with frailty term | HR with random-effect heterogeneity | Partial | Correct frailty form; PH-centric interpretation | Depletion geometry can remain; HR interpretation unstable |
| Matching / IPTW / MSM | Model-based contrasts (ATE/ATT) | Indirect (via measured proxies only) | Correct models; rich covariates; positivity | Latent frailty and depletion persist unaddressed |
| Negative control methods | Bias detection (diagnostic) | No | Valid negative controls | Detects bias but does not remove it |
| **KCOR (this work)** | **KCOR(t): ratio of depletion-neutralized cumulative hazards** | **Yes (targeted)** | **DOB/DOD/DOI; valid quiet window; identifiable curvature; diagnostics** | **If depletion model or quiet window fails, diagnostics flag nonlinearity or instability** |

---

## Caption (for manuscript)

**Table X. Positioning KCOR among retrospective methods (corrected).**  
Most retrospective approaches either compare cohorts under proportional-hazards assumptions, balance measured confounding, or diagnose bias without removing it. KCOR occupies a distinct role: it **neutralizes selection-induced depletion dynamics** via gamma-frailty inversion and then **extracts the cohort contrast using a cumulative hazard ratio (KCOR)**. Normalization alone does not yield an interpretable signal; the KCOR ratio is the estimand that answers whether one cohort experienced higher or lower cumulative event risk than another under the stated assumptions.

---

## Recommended placement in manuscript

Insert this table in **Section 1.5 (Contribution of this work)**, immediately after the opening paragraph and before the bullet list of contributions.

**Rationale:** This placement ensures that reviewers understand KCOR as a *complete comparison system* before encountering mathematical details, preventing misclassification as a preprocessing or diagnostic-only method.

---

## Optional cross-reference sentence

> The inferential role of KCOR relative to established retrospective methods is summarized in Table X, emphasizing that KCOR integrates depletion normalization with cumulative comparison and diagnostics into a single system.

