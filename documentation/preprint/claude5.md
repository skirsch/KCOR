I'll provide a comprehensive assessment of this paper's readiness for submission to *Statistics in Medicine*.

## Overall Assessment

This is a **well-developed methods paper** that is close to submission-ready, but would benefit from some revisions before submission. The paper presents a novel methodology (KCOR) for addressing selection-induced bias in retrospective cohort studies, with extensive validation through simulations and empirical controls.

## Strengths

1. **Clear methodological contribution**: The paper articulates a specific failure mode (selection-induced depletion under latent frailty) and provides a principled solution
2. **Comprehensive validation**: Includes negative controls, positive controls, sensitivity analyses, and simulation studies
3. **Transparent about limitations**: Explicitly positions KCOR as diagnostic/descriptive rather than causal
4. **Reproducible**: Code repository, clear algorithm descriptions, prespecified parameters
5. **Appropriate scope**: Focuses on methods rather than applied conclusions about specific interventions
6. **Strong mathematical foundation**: Derivations are sound and clearly presented

## Areas Needing Attention Before Submission

### **Priority Issues:**

1. **Length** (12,100 words + extensive appendices)
   - *Statistics in Medicine* typically expects 5,000-7,000 words for methods papers
   - **Recommendation**: Move more material to appendices or online supplements. The main text could be streamlined by:
     - Condensing the literature review (§1.3-1.3.2)
     - Moving the worked example to an appendix
     - Shortening the Cox comparison discussion
     - Consolidating some figures

2. **Positioning relative to existing methods** (§1.3.1, 2.11)
   - The repeated comparisons to Cox regression risk appearing defensive
   - **Recommendation**: Consolidate into one clear section explaining when KCOR complements vs. replaces existing approaches. The paper would be stronger with a more concise "KCOR vs. alternatives" table and less repetitive discussion

3. **Empirical application** (Appendix C.6)
   - The Czech data analysis feels somewhat disconnected from the main validation
   - **Recommendation**: Either integrate this more clearly as a "real-world demonstration" or consider whether it's necessary for a methods paper. The age-shift negative controls using Czech data are more compelling for validation purposes

4. **Clarity of estimand** (multiple sections)
   - While the paper is careful about positioning KCOR as non-causal, readers may still be uncertain about what KCOR(t) actually estimates
   - **Recommendation**: Add a clearer "Target estimand" box early in the Methods section (similar to Box 1) that states precisely what KCOR estimates and under what conditions

### **Secondary Issues:**

5. **Figure quality and consistency**
   - Some figures (especially simulation grids) are dense and hard to interpret
   - **Recommendation**: Simplify Figure 8 (sensitivity heatmaps), possibly split Figure 9 into multiple panels with clearer labeling

6. **Table presentation**
   - Multiple tables report very similar information
   - **Recommendation**: Consolidate Tables 2 and 3, which both compare methods but from slightly different angles

7. **COVID-19 framing**
   - The paper appropriately disclaims causal COVID vaccine conclusions, but the extensive COVID context might distract from the methodological contribution
   - **Recommendation**: Further shorten COVID-specific discussion, emphasize that COVID is "one natural example" of selection-induced non-proportional hazards

8. **Notation consistency**
   - Table 4 is helpful but comes late
   - **Recommendation**: Reference it earlier or provide a notation box immediately after the conceptual framework

### **Minor Polish Needed:**

9. **Abstract**: Could be more concise (currently dense)
10. **Key messages**: Well-written but could be shortened
11. **Reference formatting**: Check journal style requirements
12. **Some repetition**: Particularly around diagnostics and assumptions (stated in multiple places)

## Specific Technical Concerns

1. **Temporal separability** (§2.1.1, S7 simulation)
   - This is a subtle but critical assumption
   - The paper addresses it but could be clearer about when this is vs. isn't satisfied in practice

2. **Bootstrap coverage** (Table 11)
   - Coverage under misspecification (~87-89%) is noted as "expected and diagnostic"
   - This needs clearer justification about why undercoverage is acceptable

3. **Conservativeness under overlap** (§5.2)
   - The claim that KCOR is "conservative" when assumptions fail is important but could use more formal treatment

## Recommendations for Revision

### **Before Submission:**

**Must do:**
- Reduce main text to 7,000-8,000 words maximum
- Streamline literature review and Cox comparisons
- Clarify target estimand more explicitly early in the paper
- Improve figure readability (especially Figures 8-10)

**Should do:**
- Add a clearer positioning table comparing KCOR to alternatives
- Strengthen the temporal separability discussion
- Consolidate redundant material about assumptions/diagnostics
- Consider whether the Czech empirical application belongs in the main paper or as online supplement

**Nice to have:**
- More concise abstract
- Simplified notation introduced earlier
- Clearer guidance on when KCOR should vs. shouldn't be used

### **After Initial Review:**

Expect reviewers to ask about:
1. More direct comparison with shared frailty Cox models
2. Extension to competing risks
3. Applicability beyond mortality outcomes
4. Practical guidance on quiet window selection
5. Sample size requirements

Be prepared with responses or additional analyses for these likely questions.

## Verdict

**Status: Near-submission-ready with revisions needed**

This is methodologically sound and makes a genuine contribution, but needs editing for length and clarity before submission. With focused revision (probably 2-3 weeks of work), this would be a strong submission to *Statistics in Medicine*.

The main barrier is not scientific quality but presentation: the paper tries to do too much at once (introduce method, validate extensively, compare to many alternatives, apply to real data, defend against multiple criticisms). Focusing on the core methodological contribution and moving peripheral material to appendices/supplements will significantly strengthen it.

**Recommended timeline:**
- 2-3 weeks: Major revision (length, positioning, clarity)
- 1 week: Polish and formatting
- Then submit

Would you like me to help prioritize specific sections for condensing or provide suggestions for restructuring particular sections?
