# KCOR Simulation Grid & Diagnostics — Build Specification

This document is the sole build instruction file for Cursor for additions to `paper.md` KCOR paper so it will be acceptable to "Statistics in Medicine."

Cursor must implement all manuscript edits, simulations, diagnostics, figures, and repository outputs described here. No additional instruction files are to be created. All details not explicitly specified should be reasonably inferred to match the KCOR methodology described in the manuscript.

---

## Objective

Strengthen the KCOR methods paper for submission to *Statistics in Medicine* by adding a compact, prespecified simulation grid that demonstrates:

1. Correct null behavior under selection-induced curvature  
2. Detection of injected effects  
3. Graceful failure with explicit diagnostics under misspecification and adverse regimes  

All additions must remain methodological, label-blind, and non-causal.

---

## Part I — Manuscript Modifications (paper.md)

### I.1 New subsection under Section 3

Insert a new subsection after §3.3 and before §4 Discussion.

Heading:
### 3.4 Simulation grid: operating characteristics and failure-mode diagnostics

Body text to insert verbatim:

We further evaluate KCOR using a compact simulation grid designed to (i) confirm near-null behavior under selection-induced curvature, (ii) confirm detection of injected effects, and (iii) characterize failure modes and diagnostics under model misspecification and adverse data regimes. Each scenario generates cohort-level weekly counts in KCOR_CMR format. KCOR is then fit using the same prespecified quiet-window procedure as in the empirical analyses, and we report both KCOR(t) trajectories and diagnostic summaries, including cumulative-hazard fit error and post-normalization linearity. The scenarios isolate specific stresses, including non-gamma frailty, contamination of the quiet window by an external shock, and sparse events. Code to reproduce all simulations and figures is included in the repository.

Insert the following two figure references immediately after the paragraph above:

![Simulation grid overview: KCOR(t) trajectories across prespecified scenarios, including gamma-frailty null with strong selection, injected harm and benefit, non-gamma frailty, quiet-window contamination, and sparse-event regimes. Under true null, KCOR remains near-flat at 1; injected effects are detected in the expected direction; adverse regimes are accompanied by degraded diagnostics and reduced interpretability.](figures/fig_sim_grid_overview.png){#fig:sim_grid_overview}

![Simulation diagnostics across scenarios: (i) cumulative-hazard fit RMSE over the quiet window, (ii) fitted frailty variance estimates, and (iii) a post-normalization linearity metric for normalized cumulative hazards. Diagnostics identify regimes in which frailty normalization is well identified versus weakly identified.](figures/fig_sim_grid_diagnostics.png){#fig:sim_grid_diagnostics}

---

### I.2 Limitations section (recommended)

Append the following paragraph to §5 Limitations:

Simulation results in §3.4 illustrate that when key assumptions are violated—such as non-gamma frailty geometry, contamination of the quiet window by external shocks, or extreme event sparsity—frailty normalization may become weakly identified. In such regimes, KCOR’s diagnostics, including poor cumulative-hazard fit and reduced post-normalization linearity, explicitly signal that curvature-based inference is unreliable without model generalization or revised window selection.

---

## Part II — Simulation Framework

### II.1 General requirements

- Discrete time in weeks  
- Time horizon approximately 120 weeks  
- Cohort-level KCOR_CMR-style summaries  
- Same quiet-window fitting procedure as the manuscript  
- Deterministic random seeds  
- No access to exposure labels during fitting  

---

### II.2 Prespecified simulation scenarios

Implement the following scenarios exactly:

1. Gamma-frailty null (strong selection)  
   Different frailty variances across cohorts, no effect.  
   Expected behavior: KCOR approximately equal to 1.

2. Injected harm  
   Gamma frailty with a temporary multiplicative hazard increase applied to one cohort.  
   Expected behavior: KCOR greater than 1 during and after the effect window.

3. Injected benefit  
   Gamma frailty with a temporary multiplicative hazard decrease applied to one cohort.  
   Expected behavior: KCOR less than 1.

4. Non-gamma frailty null  
   Frailty drawn from a non-gamma distribution (e.g., lognormal) with no effect.  
   Expected behavior: degraded fit and diagnostics.

5. Quiet-window contamination  
   Shared external hazard shock overlapping the quiet window, no cohort-specific effect.  
   Expected behavior: diagnostics indicate poor identifiability.

6. Sparse-events regime  
   Small cohorts and/or low baseline hazards with no effect.  
   Expected behavior: noisy estimates and weak identifiability.

An optional proportional-hazards level-shift scenario may be added if useful.

---

## Part III — Diagnostics

For each scenario and cohort, compute and store:

- Quiet-window cumulative-hazard RMSE (observed versus fitted)  
- Estimated frailty variance (θ̂)  
- Post-normalization linearity metric (e.g., R² from linear fit to normalized cumulative hazard)  

Diagnostics must be explicitly used to identify failure modes.

---

## Part IV — Figures

Generate the following publication-ready figures:

1. figures/fig_sim_grid_overview.png  
   - KCOR(t) versus time  
   - Clear separation by scenario  
   - Reference line at KCOR = 1  
   - Effect windows marked where applicable  

2. figures/fig_sim_grid_diagnostics.png  
   - Scenario-level summaries of RMSE, θ̂, and linearity metrics  

Figures must be legible, consistently scaled, and suitable for journal publication.

---

## Part V — Repository Outputs

- Create a simulation directory (e.g., test/sim_grid/)  
- Implement reproducible scripts to generate simulations, fit KCOR, compute diagnostics, and render figures  
- Copy final figures to figures/ using the exact filenames referenced above  

---

## Part VI — Acceptance Criteria

This build is complete when:

1. paper.md includes §3.4 and compiles cleanly  
2. Both simulation figures exist and render correctly  
3. Null scenarios yield KCOR approximately equal to 1  
4. Injected-effect scenarios show directional deviations  
5. Misspecification and sparse regimes show degraded diagnostics  
6. All added language remains methodological and non-causal  

End of specification.
