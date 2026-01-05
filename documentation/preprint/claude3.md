I'll review this paper systematically to assess its readiness for submission to *Statistics in Medicine*.

## Overall Assessment

This is a substantial methodological paper (12,100 words) that introduces KCOR, a novel framework for retrospective cohort comparisons under selection-induced non-proportional hazards. The work is **nearly ready** for submission but would benefit from several revisions.

## Strengths

1. **Clear methodological contribution**: The paper addresses a real gap - handling selection-induced depletion in retrospective cohorts with minimal data
2. **Comprehensive validation**: Excellent use of negative/positive controls and simulation studies
3. **Transparent about limitations**: Honest discussion of what KCOR does and doesn't do
4. **Reproducible**: Code and data availability clearly specified
5. **Strong mathematical foundation**: Gamma-frailty framework is well-established

## Issues Requiring Attention Before Submission

### Major Issues

1. **Framing and Positioning** (Critical)
   - The COVID-19/vaccine focus dominates the paper despite claims of generality
   - *Statistics in Medicine* reviewers will want clearer demonstration of broader applicability
   - Consider leading with a non-COVID example or making the general framework more prominent
   - The extensive COVID-specific discussion may trigger reflexive skepticism

2. **Comparison with Existing Methods** (Important)
   - Tables 2 and 3 are good starts, but need more depth
   - Missing formal comparison with:
     - Joint frailty models (which also handle selection)
     - G-computation approaches
     - Doubly robust methods
   - Need simulation comparisons showing when KCOR outperforms alternatives

3. **Estimand Clarity** (Important)
   - The paper wavers between "diagnostic tool" and "estimator"
   - Section 1.6 and 4.1 need tighter integration
   - Be crystal clear: what parameter does KCOR estimate under what assumptions?
   - The "depletion-neutralized baseline cumulative hazard ratio" needs more intuitive interpretation

4. **Identifiability** (Critical for *Stat Med*)
   - The quiet window assumption (A5) does heavy lifting
   - Need more formal treatment of when/why this identifies causal effects vs. just removes bias
   - The connection between "depletion-neutralized" and "causal" needs clearer explication
   - Reviewers will push hard on: "How do we know the quiet window removes ALL confounding?"

### Moderate Issues

5. **Structure and Length**
   - At 12,100 words + extensive appendices, this may exceed journal limits
   - Consider moving some material to supplementary materials:
     - The extensive COVID literature review (§1.4)
     - Some simulation details
     - The Czech application (it's not really validated yet as stated)
   
6. **Results Presentation**
   - Table 9 is mentioned but says "[coverage value]" - needs actual results
   - Table 10 is incomplete
   - Simulation results in §3.4 reference figures but don't provide enough numeric summaries
   - Need a clear "operating characteristics" table showing bias, MSE, coverage across scenarios

7. **The Czech Application**
   - You explicitly state this is "illustrative" and "does not support causal inference"
   - But Tables 11-13 and Figures 14-17 occupy significant space
   - Either:
     - (A) Move entirely to supplement as "worked example"
     - (B) Properly analyze with causal claims (separate paper, as you indicate)
     - (C) Remove entirely and use only synthetic examples
   - Current middle ground undermines the "methods paper" framing

8. **Writing Issues**
   - Some repetition (e.g., the "normalization is necessary but not sufficient" point appears multiple times)
   - The distinction between static and dynamic HVE, while important, interrupts flow
   - Some notation inconsistencies (e.g., $\tilde{H}_0$ vs $H_0$)

### Minor Issues

9. **References**
   - Missing some key frailty modeling papers (Hougaard, Wienke)
   - The ADEMP framework is mentioned but not cited
   - Some URLs in references may need updating

10. **Figures**
    - Figure 1 is helpful but could be clearer
    - Figures 2-3: The synthetic null demonstration is excellent
    - Figures 9-10: Good simulation diagnostics
    - Some figures in Appendix C could move to supplement

11. **Technical Details**
    - Equation (11): The specific hazard transform seems arbitrary - justify or simplify
    - The "skip weeks" approach (§2.7) needs better justification
    - Why constrained NLS instead of MLE? (You explain, but could be stronger)

## Specific Recommendations

### For *Statistics in Medicine* Specifically

1. **Add a Methods Summary Box** early in the paper:
   - What problem does KCOR solve?
   - When should it be used vs. alternatives?
   - What assumptions are required?
   - What diagnostics signal failure?

2. **Strengthen the simulation study**:
   - More direct comparisons with competing methods
   - Operating characteristics table (bias, RMSE, coverage) across DGPs
   - Show convergence properties

3. **Address causal inference more carefully**:
   - Either commit to causal interpretation with clear DAG + assumptions
   - Or explicitly position as "bias reduction" tool requiring additional work for causality
   - Current hedge may frustrate reviewers

4. **Simplify the narrative**:
   - Make it about "selection-induced depletion" not "COVID vaccines"
   - Lead with motivating example that's less politically fraught
   - Save COVID application for follow-up paper

### Before Submission Checklist

- [ ] Reduce length to journal limits (check *Stat Med* guidelines)
- [ ] Complete all tables (especially Table 9, 10)
- [ ] Add numerical results for all simulations mentioned
- [ ] Clarify estimand and identifiability conditions
- [ ] Add direct comparison with joint frailty models
- [ ] Decide on Czech application: supplement, remove, or full analysis
- [ ] Add methods summary box
- [ ] Check all references are current and complete
- [ ] Ensure all figures are publication-quality
- [ ] Proofread for repetition and tighten prose

## Bottom Line

**Not quite ready, but close.** The core methodology is sound and the validation is thorough. However, the framing needs work to avoid reviewer skepticism about the COVID focus, and the estimand/identifiability needs clearer exposition. With 2-3 weeks of focused revision addressing the major issues above, this could be a strong submission.

The biggest risk is that reviewers dismiss it as "COVID vaccine methods paper" rather than seeing the general contribution. Frame it as solving the "selection + depletion + non-PH + minimal data" problem, with COVID as one application domain among others.

Would you like me to elaborate on any of these points or help draft specific revisions?
