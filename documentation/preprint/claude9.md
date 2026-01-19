I've reviewed your KCOR manuscript. Overall, this is a **well-structured and technically rigorous methods paper** that makes a clear contribution. Here's my assessment:

## Strengths

1. **Clear problem definition**: You've articulated well how selection-induced depletion creates non-proportional hazards that bias standard estimands

2. **Solid methodological foundation**: The gamma-frailty normalization approach is mathematically sound and the inversion formula (Eq. 7) provides an explicit correction mechanism

3. **Comprehensive validation**: 
   - Synthetic null showing Cox bias under pure frailty heterogeneity (Figures 2-3, Table 6)
   - Multiple negative controls (synthetic and empirical age-shift)
   - Positive controls with injected effects
   - Extensive sensitivity analyses in SI

4. **Honest scoping**: You're appropriately clear about what KCOR is and isn't (Box 2), avoiding overclaiming causality

5. **Diagnostic-driven approach**: The emphasis on falsifiable diagnostics rather than assuming identifiability is methodologically sound

## Areas for Refinement

### 1. **Length Management** (Critical for Stat in Med)
At ~10,900 words (excluding abstract/refs), you're likely over the journal's typical limit. Consider:
- Moving more technical details to SI (e.g., §2.4.3-2.4.4 baseline shape justification)
- Condensing the Cox comparison (§2.11) - the point is made well but takes substantial space
- Streamlining §4.2 ("What KCOR estimates") which somewhat repeats earlier framing

### 2. **Positioning vs. Existing Methods** (§1.3)
You mention this is "brief" - reviewers will want more detail on:
- How KCOR compares to **time-varying coefficient Cox models** (which also handle non-PH)
- Relationship to **marginal structural models** (MSMs) that target cumulative effects
- Connection to **flexible parametric survival models** (Royston-Parmar splines)

Table 3 helps but could be expanded with specific references and clearer distinctions.

### 3. **Identifiability Assumptions** (§2.1.3, Box 2)
The quiet-window assumption is doing a lot of work. Consider adding:
- More explicit discussion of **what validates a quiet window** (beyond diagnostics)
- How to handle settings where **no adequate quiet window exists**
- Whether multiple short windows could be pooled

### 4. **Real-Data Example Framing**
The Czech vaccination data example is described as "solely to demonstrate estimator behavior" but:
- Table S7-S8 show striking patterns (e.g., θ̂ ≈ 0 for Dose 2 vs θ̂ = 23 for Dose 0 age 50-59)
- You're making substantive claims about "selective uptake" and "healthy vaccinee effect"

**Recommendation**: Either lean into this as a substantive application (with appropriate caveats) OR use simpler synthetic examples throughout and move Czech data entirely to SI as optional validation.

### 5. **Technical Clarifications**

**Equation 2**: You use the transform h = -ln(1 - MR) but then say "for rare events h ≈ MR". Should clarify you're using the exact transform throughout (not the approximation).

**Bootstrap procedure** (§2.9.1): "Resample cohort-time counts" - are you resampling (d_d(t), N_d(t)) pairs with replacement? More detail would help reproducibility.

**Anchoring** (multiple places): You introduce KCOR(t; t₀) but sometimes report unanchored KCOR(t). Be more consistent about when/why anchoring is used.

### 6. **Figure Quality**
- Figure 1 (schematic): Very helpful conceptually
- Figures 2-3: Clear demonstration of Cox vs KCOR under null
- Figure 4 & S8 (negative controls): These are screen captures from what looks like an internal dashboard rather than publication-quality figures. These need to be regenerated as proper plots.

### 7. **Discussion Points to Strengthen**

**When KCOR fails gracefully** (§5.1): You note that violations lead to θ̂→0 rather than spurious effects. This is important - consider highlighting this "conservative failure mode" more prominently.

**Comparison to cause-specific hazards**: You state KCOR targets all-cause mortality. How would extension to cause-specific mortality work? Brief comment would be useful.

**Multiple testing**: With multiple dose comparisons, age strata, and enrollment windows, how should readers think about multiplicity?

## Specific Editorial Suggestions

### Abstract
- "KCOR provides a diagnostic and descriptive framework" - consider "KCOR provides a normalization and comparison framework" (more active)
- Consider adding one sentence on when KCOR should NOT be used

### Introduction
- §1.2 could be shortened - the conceptual point about curvature is made multiple times
- §1.4 literature evidence: Obel and Chemaitelly citations are good but need brief context on study design

### Methods
- §2.2 "Cohort construction": The no-switching rule is important but buried. Consider a highlighted box.
- §2.10 Algorithm summary: Table 5 is excellent - consider moving earlier as a roadmap

### Results
- §3.1.2: "pragmatic negative control construction" - be more explicit this is a pseudo-null by design
- §3.2: Positive controls are in SI only - consider showing at least one in main text

### Discussion
- §4.4 "Practical guidelines" reads like methods - consider moving some to Methods
- §6 Conclusion is appropriately modest but very short - could expand slightly

## Minor Issues

1. **Notation**: You use both H̃₀,d and H_obs,d - the tilde notation is clear but consider defining in a notation table earlier (Table 4 comes late)

2. **References**: Reference 12 (Czech data source) is cited as "Accessed December 20, 2025" - should be 2024

3. **SI organization**: Very thorough but ~20 pages. Consider moving some simulation details to a separate technical appendix or online repository

4. **Code availability**: You mention GitHub and Zenodo - excellent. Ensure the DOI is finalized before submission.

## Bottom Line

This is **publishable work** that makes a genuine methodological contribution. The core idea is sound, validation is thorough, and framing is honest. 

**Priority revisions for Statistics in Medicine**:
1. Cut 1,000-1,500 words (target ~9,500)
2. Expand positioning vs. time-varying Cox and MSMs
3. Regenerate Figure 4 and S8 as publication-quality plots
4. Clarify Czech data role (demonstrative vs. substantive)
5. Add brief multiplicity/multiple testing comment

**Estimated revision effort**: 2-3 weeks for a solid revision addressing these points.

Would you like me to help with any specific section - for example, drafting expanded text for the methods comparison, or suggesting specific cuts to reach word count?
