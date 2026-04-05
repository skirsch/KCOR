Here's a mock JASA referee report:

---

# Peer Review: "KCOR: A depletion-neutralized framework for retrospective cohort comparison under latent frailty"

**Journal:** Journal of the American Statistical Association (Theory and Methods)
**Recommendation:** Minor Revision

---

## Summary

The author proposes KCOR, a framework for comparing retrospective cohorts under latent frailty heterogeneity. The core observation — that selection-induced depletion of susceptibles generates non-proportional hazards and curvature in cumulative hazard trajectories that standard Cox regression cannot adequately handle — is correct, well-documented in the frailty literature, and practically important. The proposed estimator fits a gamma-frailty working model with a Gompertz baseline during epidemiologically quiet periods, inverts the frailty-induced curvature analytically, and computes a ratio of depletion-normalized cumulative hazards. An optional module extends the framework to epidemic-wave settings via a non-proportional hazard exponent parameter α.

The manuscript is notably self-aware about its own limitations. It explicitly disclaims causal interpretation, enforces diagnostic gating before reporting results, and treats non-identification of α in the Czech application as a successful diagnostic outcome rather than a method failure. This intellectual honesty is refreshing and distinguishes KCOR from methods papers that oversell their estimands. The validation stack is comprehensive: θ₀ recovery simulations, negative controls (both synthetic and empirical), positive controls with injected effects, stress tests under frailty misspecification, and explicit failure signaling. No mathematical errors were identified.

The primary concern is framing rather than methodology: the paper simultaneously targets a general statistical methods audience and a specific COVID-19 vaccine mortality application, and the resulting tension weakens both contributions without invalidating either. Resolving this framing decision is the main revision needed.

---

## Major Concerns

**1. Commit to a lane: general methods paper or motivated application paper.**

The manuscript occupies an uncomfortable middle ground. The abstract and introduction frame KCOR as a general tool for "any irreversible event process," yet the entire empirical apparatus — Czech registry data, epidemic wave modules, COVID-specific quiet window selection — is drawn from a single application domain. JASA Theory and Methods readers will ask: does this generalize, and where is the evidence?

Two defensible paths exist. Path A: frame this as a general survival analysis methods contribution, reduce the COVID-specific apparatus to one illustrative application, and add at least one empirical example from a different domain (e.g., cardiovascular mortality, cancer registries, or occupational cohort studies where frailty depletion is known to operate). Path B: frame this explicitly as a methods paper motivated by and validated on COVID-era registry data, restrict scope claims accordingly, and own the application rather than hedging it. Either path is publishable. The current straddling is not.

**2. The identification argument needs a formal statement, not just an informal one.**

Section 2.4.4 provides an informal identification statement but explicitly declines to claim a formal nonparametric identification theorem. This is honest, but for a JASA Theory and Methods submission it is insufficient. The paper should either: (a) state and prove a formal identification theorem under the stated working model assumptions, even if those assumptions are strong; or (b) clearly reposition the paper as a methods contribution without formal identification guarantees, with the informal statement explicitly labeled as such and the implications for interpretation spelled out. The current framing — gesturing toward identification while disclaiming formal claims — leaves the theoretical contribution underspecified.

**3. Comparison to shared-frailty Cox is asserted but not demonstrated.**

Table 2 and Section 2.11 argue that shared-frailty Cox models differ fundamentally from KCOR in estimand and fail to remove depletion-induced selection effects from marginal cumulative comparisons. Table 9 provides some simulation evidence. However the simulation is limited: it shows that shared-frailty Cox HR deviates from 1 under a selection-only null (0.94 vs. true 1.0), while KCOR remains flat. This is the right test but the conditions are narrow. JASA readers will ask: under what conditions does shared-frailty Cox fail more severely than KCOR? Under what conditions does it fail less? What is the operating characteristic comparison across the frailty variance grid shown in Table 6? A more complete simulation comparison — matching the frailty grid already present in the paper — would substantially strengthen the contribution.

---

## Moderate Concerns

**4. Bootstrap coverage below nominal under exactly the conditions that matter.**

Table 11 reports empirical coverage of 87.6% under sparse events and 89.3% under non-gamma frailty, both well below the nominal 95%. The paper acknowledges this and correctly notes these regimes coincide with degraded diagnostics. However the explanation — that sub-nominal coverage occurs when assumptions are violated and diagnostics flag failure — is circular as a defense: it says the method is well-calibrated when it works and poorly calibrated when it doesn't. A stronger response would either: (a) provide a theoretical bound on coverage degradation as a function of frailty misspecification, or (b) propose a diagnostic-adjusted interval that achieves closer to nominal coverage in the regimes where the method is used. The current treatment is honest but incomplete.

**5. The Gompertz fixed-slope assumption deserves more than sensitivity analysis.**

The fixed universal γ assumption — that the rate of exponential age-related hazard increase is identical across all cohort definitions, enrollment periods, and age strata — is strong. Figure S5 shows the estimator is empirically robust to quiet-window placement, and the paper references sensitivity analyses over γ. But the sensitivity results are not shown in the main text, and more importantly, no theoretical analysis characterizes how KCOR(t) behaves when γ is misspecified. Is misspecification in γ absorbed into θ̂₀,d, and if so, in which direction? Does it attenuate the normalized contrast or amplify it? A brief analytical treatment of this misspecification pathway — even in a supplementary section — would materially strengthen the paper.

**6. The empirical negative control conflates two validation goals.**

The age-shift negative control in §3.1.2 is presented as validating KCOR's ability to handle composition differences under a pseudo-null. But the paper itself notes that these pseudo-cohorts are full-population strata rather than selectively sampled subcohorts, so fitted θ̂₀,d values are expected to be small or weakly identified. This means the negative control is testing KCOR behavior when frailty depletion is minimal — precisely the regime where the normalization does little work and near-null KCOR(t) is unsurprising. A more informative negative control would induce strong frailty heterogeneity (large θ₀) while preserving a true null, to demonstrate that KCOR removes genuine depletion-induced curvature rather than simply passing through a near-flat trajectory.

---

## Minor Concerns

- The optional NPH module is presented as part of the KCOR architecture despite being non-identified in the primary application and requiring externally supplied VE that is not generally available. Consider whether this module belongs in the main paper or in a companion methods note, particularly given the paper's length.

- Several notation conventions are introduced and then re-encountered under slightly different names (effective hazard, adjusted hazard, preprocessed hazard). A single unified notation table introduced early and used consistently would improve readability considerably. Table 4 is helpful but incomplete relative to what appears in the methods.

- The reproducibility setup — GitHub repository, Zenodo DOI, documented build pipeline — is exemplary and should be highlighted more prominently in the data availability statement.

- Figure 5 (empirical negative control) uses an Excel-style pivot chart that is not of publication quality. This should be replaced with a matplotlib or R-generated figure consistent with the other figures in the paper.

- The claim in §2.11 citing Deeks (2002) that covariate adjustment can exacerbate bias in non-randomized analyses is a selective citation from a context (meta-epidemiology of non-randomized intervention studies) that differs materially from registry-based cohort comparison. The citation is not wrong but the framing is stronger than the evidence supports.

---

## What This Review Does Not Find

For the avoidance of doubt: this review identified no mathematical errors in the KCOR estimator, no errors in the gamma-frailty inversion, no errors in the delta-iteration procedure, and no errors in the bootstrap uncertainty quantification. The core claim — that curvature-based identification of θ₀,d under a Gompertz working model, followed by gamma-frailty inversion, removes selection-induced depletion bias from marginal cumulative hazard comparisons — is not challenged here. The simulation evidence supporting this claim (Tables 6, 9, 10; Figures 3, 4, S1–S3) is credible and appropriately hedged.

---

## Summary Recommendation

This paper addresses a real and underappreciated problem with a distinctive approach. The diagnostic-first philosophy — refusing to report results when identification fails — is methodologically honest and practically important. The core estimator is sound, the validation stack is comprehensive, and the limitations are disclosed forthrightly. The revision needed is primarily one of framing and scope commitment rather than methodology.

Specifically, the author should: (1) choose and commit to a framing lane (general methods or motivated application); (2) provide a formal identification theorem or explicitly reposition the theoretical contribution; (3) extend the shared-frailty Cox comparison across the frailty variance grid; (4) address the bootstrap coverage gap more substantively; and (5) replace the Excel-style figure with a publication-quality plot.

I look forward to reviewing a revised version.

---

*Disclosure: The reviewer has no conflict of interest with respect to this manuscript or its author.*

## fixes
Apply the following targeted revisions to address reviewer feedback. Do NOT expand scope beyond these changes. Keep tone restrained and consistent with current manuscript.

----------------------------------------
1. COMMIT TO GENERAL METHODS LANE
----------------------------------------

Edit Abstract:

- In first paragraph, replace any COVID-specific framing with:

  “We propose KCOR, a framework for retrospective cohort comparison with irreversible outcomes under latent frailty heterogeneity.”

- Add sentence later in abstract:

  “We illustrate the method using national COVID-era registry data as a worked example.”

Edit Introduction (first 2–3 paragraphs):

- Add sentence:

  “While motivated by epidemic-era registry data, the method applies to general retrospective cohort comparisons where selection-induced depletion of susceptibles generates curvature in cumulative hazard trajectories.”

----------------------------------------
2. ADD IDENTIFICATION PROPOSITION BOX
----------------------------------------

Location: immediately after §2.4.4 (Identifiability discussion)

Insert boxed statement:

  **Proposition (Identification under curvature).**  
  Under a Gompertz baseline hazard and gamma frailty with variance parameter θ₀,d, the mapping from observed cumulative hazard H_obs,d(t) over a non-degenerate interval to θ₀,d is injective provided H_obs,d(t) exhibits non-zero curvature. Consequently, θ₀,d is identifiable from quiet-window data up to sampling variability.

Add immediately after box:

  “This is a parametric identification result under the working model. KCOR does not claim nonparametric identification; instead, identifiability is evaluated empirically via curvature diagnostics.”

----------------------------------------
3. EXPAND COX COMPARISON (HIGH PRIORITY)
----------------------------------------

In Results or Simulation section:

- Add new figure:

  X-axis: θ₀ (frailty variance grid from Table 6)  
  Y-axis: bias (estimated / true)

  Plot three curves:
    - Cox HR
    - Shared-frailty Cox
    - KCOR

Caption:

  “Estimator bias as a function of frailty variance. Cox-based estimators exhibit increasing bias under selection-only regimes, while KCOR remains approximately unbiased across the grid.”

- Add sentence in §2.11 or Results:

  “Bias in Cox-based estimators increases monotonically with frailty variance, while KCOR remains near-unbiased across the same parameter range.”

----------------------------------------
4. BOOTSTRAP CLARIFICATION
----------------------------------------

Location: §2.9 (Bootstrap / uncertainty)

Add paragraph:

  “The bootstrap is used as a variance-propagation device for the aggregated process rather than a universally calibrated inferential procedure. Sub-nominal coverage arises in regimes where model assumptions are violated or curvature is insufficient for identification. In these regimes, KCOR diagnostics explicitly flag non-identifiability and results are not interpreted. Reported intervals are therefore conditional on passing identifiability diagnostics.”

----------------------------------------
5. GOMPERTZ MISSPECIFICATION PARAGRAPH
----------------------------------------

Location: §2.4.3 or sensitivity section

Add:

  “Misspecification of the Gompertz slope γ primarily rescales the inferred frailty parameter θ̂₀,d. Because KCOR(t) is constructed from ratios of depletion-normalized cumulative hazards, first-order effects of γ misspecification tend to cancel unless misspecification differs systematically across cohorts. Sensitivity analyses varying γ confirm that KCOR(t) is primarily driven by curvature structure rather than the exact baseline parameterization.”

----------------------------------------
6. NEGATIVE CONTROL CLARIFICATION
----------------------------------------

Location: §3.1.2

Add sentence:

  “This negative control evaluates end-to-end pipeline behavior under model-consistent conditions with minimal depletion. Robustness to strong depletion and model misspecification is assessed separately in simulation stress tests.”

----------------------------------------
7. NPH MODULE REPOSITIONING
----------------------------------------

Location: §2.7 intro or §5 discussion

Add:

  “The optional NPH module extends the framework to settings with wave-induced hazard amplification but is not required for core KCOR estimation. When α is not identified, the module is inactive and does not affect reported results.”

----------------------------------------
8. CONFUNDING / NON-CAUSAL CLARIFICATION
----------------------------------------

Location: Discussion §4.1

Add paragraph:

  “KCOR addresses selection-induced depletion that distorts marginal comparisons but does not remove general confounding. It should be interpreted as a diagnostic normalization step. Differences observed after normalization may reflect causal effects, residual confounding, or other mechanisms and require additional evidence for causal interpretation.”

----------------------------------------
9. GLOBAL PIPELINE SENTENCE (IMPORTANT)
----------------------------------------

Location: end of §2.1 or start of §2.7.1

Add:

  “KCOR operates by first estimating depletion geometry (θ) from quiet windows, then optionally adjusting wave-period amplification (α), and finally comparing cumulative outcomes on the normalized scale.”

----------------------------------------
10. CONSISTENCY CHECK
----------------------------------------

Search entire document:

- Remove any statement that NPH correction is applied before inversion
- Ensure all references say:
  “applied after gamma-frailty inversion in cumulative-hazard space”

----------------------------------------
11. MINOR FIGURE FIX
----------------------------------------

Replace Excel-style figure (Figure 5) with matplotlib/R equivalent consistent with other figures.

----------------------------------------
END
