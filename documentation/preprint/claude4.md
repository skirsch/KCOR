I'll conduct a thorough review of your KCOR methodology paper for submission readiness to Statistics in Medicine.

## Overall Assessment

This is a **substantially mature manuscript** with strong methodological content, but it requires **moderate revisions** before submission. The core contribution is clear and novel, but several areas need refinement for a methods journal.

## Major Strengths

1. **Clear methodological contribution**: KCOR addresses a genuine gap in survival analysis under selection-induced depletion
2. **Comprehensive validation**: Excellent use of negative/positive controls and simulation studies
3. **Transparent about limitations**: Honest discussion of assumptions and failure modes
4. **Reproducible**: Code availability and detailed specifications
5. **Strong mathematical foundation**: Rigorous derivations in appendices

## Critical Issues Requiring Revision

### 1. **Framing and Scope** (Major)

**Problem**: The paper oscillates between being a general methods paper and a COVID-vaccine-specific analysis. For Statistics in Medicine, you need to decide:

- **Option A**: Pure methods paper with COVID as *one illustration*
- **Option B**: Methods paper motivated by vaccine studies but validated generally

**Current issue**: Sections 1.4.1-1.4.2 (Denmark/Qatar studies) and extensive COVID discussion suggest vaccine-specific focus, but you claim generality.

**Recommendation**: 
- Move vaccine-specific motivation to a brief paragraph
- Lead with the general problem: "Selection-induced non-proportional hazards in retrospective cohorts"
- Use COVID as *one* of multiple illustrations (add 1-2 non-COVID examples even if synthetic)

### 2. **Estimand Definition** (Major)

**Problem**: §2.1.1 defines KCOR formally, but the interpretation remains ambiguous:

```
KCOR(t) = H̃₀,A(t) / H̃₀,B(t)
```

**Questions a reviewer will ask**:
- What does this ratio *mean* scientifically?
- How does it relate to traditional estimands (HR, survival differences)?
- When is KCOR(t) ≈ 1 meaningful vs. when KCOR(t) = 1.2 is meaningful?

**Recommendation**:
- Add a "Target Estimand and Interpretation" subsection early in Methods
- Provide a plain-language interpretation (e.g., "KCOR(t) = 1.2 means that after removing selection-induced depletion, cohort A accumulated 20% higher baseline hazard than cohort B by time t")
- Clarify relationship to cumulative incidence

### 3. **Assumption 5 Operationalization** (Major)

**Problem**: A5 (quiet window validity) is your **most critical** assumption but lacks algorithmic specification.

**Current state**: "Prespecified quiet period... where external shocks are minimal"

**What's missing**:
- **How do you select the quiet window prospectively?**
- What constitutes "minimal" shocks quantitatively?
- Can multiple candidate windows be tested?

**Recommendation**:
Add to §2.4.4 or new §2.4.5:
```
Quiet Window Selection Protocol:
1. Visual inspection of calendar-time hazard overlays
2. Exclude periods with >X% week-over-week hazard changes
3. Require minimum window length of Y weeks
4. Pre-register window before analysis (or sensitivity test multiple windows)
```

### 4. **Cox Comparison Interpretation** (Moderate)

**Problem**: §2.11 and Table 6 present Cox as producing "spurious" results under frailty.

**Issue**: This fram framing may antagonize reviewers. Cox PH estimates a **different estimand** (average HR) that can be non-null even when the null is true for *your* estimand.

**Recommendation**:
- Reframe as "estimand mismatch" rather than "Cox failure"
- Acknowledge Cox correctly estimates *its* target (marginal HR) but that target conflates depletion with treatment
- Position KCOR as targeting a *different* (cumulative, depletion-neutralized) contrast

### 5. **Uncertainty Quantification** (Moderate)

**Problem**: §2.9 describes bootstrap but:
- Doesn't justify *why* bootstrap is preferred over delta method
- Doesn't discuss coverage properties
- Table 11 shows empirical coverage < 95% in some scenarios

**Recommendation**:
- Add justification: "Bootstrap preferred because [captures estimation uncertainty in frailty parameters / avoids normal approximation assumptions]"
- Discuss why coverage is <95% in non-gamma frailty scenarios (model misspecification)
- Consider reporting both bootstrap and asymptotic intervals as sensitivity

## Moderate Issues

### 6. **Length and Organization**

**Current**: 12,100 words + extensive appendices

**Problem**: Dense for a methods paper. Statistics in Medicine readers expect:
- Core method: ~8,000 words
- Essential validation: Integrated
- Extended material: Supplementary

**Recommendation**:
- Move §1.3.1 (related work beyond PH) to Supplement
- Move most of Appendix B (control specifications) to Supplement  
- Move Appendix C.1 (Czech application) to Supplement or separate applied paper
- Keep main text focused on: Problem → Method → Core validation → Diagnostics

### 7. **Figures**

**Current**: 17 figures, many are illustrative applications

**Issues**:
- Figures 14-17 (Czech data) should be supplement or separate paper
- Figure 1 (conceptual schematic) is good but could be more specific
- Figure 2-3 (synthetic null) are excellent, keep prominent

**Recommendation**:
Main text figures (aim for ~8-10):
- Figure 1: Enhanced schematic
- Figures 2-3: Synthetic null (Cox vs KCOR)
- Figure 4: Negative control
- Figure 7: Positive control
- Figure 8: Sensitivity analysis
- Figures 9-10: Simulation grid + diagnostics
- Figures 11-12: S7 validation

Move Czech illustrations to supplement.

### 8. **Notation and Accessibility**

**Problem**: Heavy notation may limit accessibility

**Recommendation**:
- Add Table 4 (notation) *earlier* (currently on page 9)
- Consider a "Quick Reference Box" summarizing key equations
- Add more intuitive explanations alongside formal definitions

## Minor Issues

### 9. **Title**

Current: "KCOR: A Depletion-Neutralized Cohort Comparison Framework..."

**Too long** for Statistics in Medicine (typically <15 words)

**Suggestion**:
"KCOR: Gamma-Frailty Normalization for Retrospective Cohort Comparisons Under Selection-Induced Non-Proportional Hazards"

Or simpler:
"Depletion-Normalized Survival Comparisons Using Gamma-Frailty Inversion"

### 10. **Abstract**

- Currently 250 words, good length
- Consider adding one sentence on computational implementation
- Clarify what "minimal registry data" means (just dates)

### 11. **Code Availability**

- Good: GitHub + Zenodo
- Add: "Code was reviewed by [independent party / co-author]" if applicable
- Consider: Submitting to CRAN or similar for long-term accessibility

## Specific Section Comments

### Introduction
- **1.1**: Excellent, concise
- **1.2-1.3**: Good motivation, but could be 20% shorter
- **1.4**: Too vaccine-specific for a general methods paper
- **1.6**: Target estimand section is good but needs expansion (see issue #2)

### Methods
- **2.1-2.4**: Strong, clear mathematical development
- **2.5**: Equation (10) - justify least squares vs. MLE more explicitly
- **2.7**: Skip-weeks handling is ad-hoc; needs more principled discussion
- **2.11**: Reframe Cox comparison (see issue #4)

### Validation
- **3.1-3.2**: Excellent control-test design
- **3.3**: Good sensitivity analyses
- **3.4**: Simulation grid is comprehensive
- Consider adding: Finite-sample performance (n=1000 vs n=10000)

### Discussion
- **4.1**: Good limitations discussion
- **4.2**: Could be more specific about when KCOR is/isn't appropriate
- Add: Practical guidance flowchart ("Should I use KCOR?")

## Pre-Submission Checklist

Before submitting, ensure:

- [ ] Manuscript follows Statistics in Medicine author guidelines precisely
- [ ] All co-authors (if any) have approved final version
- [ ] Supplementary material is properly organized
- [ ] Code repository is clean and documented
- [ ] All figures are high-resolution and publication-ready
- [ ] References are complete and formatted correctly
- [ ] Have addressed all "Critical Issues" above
- [ ] Have obtained ethics approval statement (even for simulated data, state it's not human subjects research)
- [ ] Conflict of interest statement is accurate (vaccine safety foundation board membership)

## Estimated Timeline to Submission-Ready

- **With focused effort**: 3-4 weeks of revision
- **Key tasks**:
  - Week 1: Reframe introduction, clarify estimand
  - Week 2: Reorganize (move Czech to supplement), refine Cox comparison
  - Week 3: Address methodological clarifications, improve figures
  - Week 4: Final polishing, formatting, pre-submission checks

## Bottom Line

**This is strong methodological work that will likely be published**, but it needs focused revision to meet Statistics in Medicine standards. The core contribution is solid and novel. The main work is:

1. **Sharpening the general methods focus** (less vaccine-specific)
2. **Clarifying the estimand** (what does KCOR mean scientifically?)
3. **Operationalizing the quiet window assumption**
4. **Reorganizing for length** (main text ~8-9k words)

These are all achievable revisions that will strengthen an already strong paper.

Would you like me to help you draft specific sections (e.g., the estimand interpretation subsection, or the quiet window selection protocol)?
