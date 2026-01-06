# Mock Associate Editor Review (10-minute skim)
**Journal**: *Statistics in Medicine*  
**Manuscript**: KCOR: A Depletion-Neutralized Cohort Comparison Framework  
**Decision**: **Send for review** (with minor pre-review suggestions)

---

## Overall Assessment

**Fit for journal**: ✅ Strong fit. Novel methodology addressing recognized problem in observational survival analysis. Appropriate scope for a methods paper.

**Novelty**: ✅ Clear. Distinct estimand (cumulative hazard ratio) and normalization-first approach differentiate KCOR from existing non-PH methods.

**Rigor**: ✅ Comprehensive. Extensive simulation validation (ADEMP framework), negative/positive controls, sensitivity analyses, and explicit diagnostics.

**Clarity**: ✅ Well-structured. Recent edits (non-causal framing, quiet-window example, practical guidelines) address potential reviewer concerns proactively.

---

## What Would Concern Me (Pre-Review Flags)

### 🔴 Potential Reviewer Concerns (Will Likely Be Raised)

1. **Quiet-window subjectivity** (Medium concern)
   - ✅ **Mitigated**: You've added the COVID-19 practical example (§2.4.4) and emphasized robustness diagnostics. This is now well-handled.
   - ⚠️ **Reviewer may still ask**: "How do we know the quiet window isn't cherry-picked?" → Your sensitivity analyses and diagnostic failures address this, but reviewers may want more explicit guidance on window selection criteria.

2. **Gamma-frailty assumption** (Low–Medium concern)
   - ✅ **Well-addressed**: §5 Limitations explicitly discusses non-gamma frailty and notes that diagnostics signal misspecification.
   - ⚠️ **Reviewer may ask**: "What if selection acts through mechanisms other than multiplicative frailty?" → You've covered this, but a brief simulation showing KCOR behavior under alternative frailty structures (e.g., log-normal) would strengthen robustness claims.

3. **Interpretability of cumulative hazard ratio** (Low concern)
   - ✅ **Addressed**: §4.1 clearly explains cumulative vs. instantaneous estimands.
   - ⚠️ **Minor**: Some reviewers may want a brief translation guide: "If KCOR(t) = 1.2, what does this mean in survival terms?" Consider a 1–2 sentence note in §4.1 or §4.3.

### 🟡 Minor Presentation Issues (Easy Fixes)

4. **Abstract density** (Low concern)
   - ✅ **Fixed**: New abstract is streamlined and front-loads innovation.
   - ✅ **Good**: Non-causal framing is now explicit.

5. **Figure/table callouts** (Very Low concern)
   - ✅ **Verified**: Cross-references appear consistent and in order.

6. **Software availability** (Low concern)
   - ✅ **Present**: Repository mentioned (§2.13, §2.14), reproducibility checklist included.
   - ⚠️ **Minor**: Ensure Code/Data Availability statement explicitly states repository URL and DOI (if archived). Some journals require this in a dedicated section.

### 🟢 Strengths (Will Impress Reviewers)

- **Diagnostics-first design**: The emphasis on transparent failure modes (§5.1, Appendix D) is sophisticated and aligns with modern statistical practice.
- **Honest limitations**: §5 Limitations is comprehensive and appropriately cautious. The non-causal framing (§4 opening) is now impossible to miss.
- **Validation strategy**: Negative controls, positive controls, and sensitivity analyses provide strong evidence of operating characteristics.
- **Practical guidelines**: §4.3 "Practical Guidelines for Implementation" signals usability—this is exactly what *Statistics in Medicine* values.

---

## Pre-Review Suggestions (Optional, Before Submission)

### Quick Wins (5–10 minutes each)

1. **Add repository URL/DOI** to Code/Data Availability section (if not already present).
2. **Brief survival interpretation** in §4.1 or §4.3: "KCOR(t) = 1.2 means cohort A experienced 20% higher cumulative hazard than cohort B after depletion normalization; this corresponds to approximately [brief survival translation]."
3. **Check word count**: Verify 12,100 words excludes Abstract/References/Supplement as stated.

### Nice-to-Have (If Time Permits)

4. **Alternative frailty simulation**: A brief simulation showing KCOR behavior under log-normal frailty (if diagnostics still work) would strengthen robustness claims. This is optional—your current coverage is adequate.

---

## What Reviewers Will Likely Focus On

### Reviewer 1 (Methods/Theory)
- ✅ **Will appreciate**: Rigorous derivation, explicit assumptions, diagnostics.
- ⚠️ **May ask**: "Can KCOR be extended to competing risks?" (You've noted this in §4.4, but reviewer may want more detail.)
- ⚠️ **May ask**: "What is the asymptotic behavior of KCOR under misspecification?" (Consider brief note if not already covered.)

### Reviewer 2 (Applied/Clinical)
- ✅ **Will appreciate**: Practical guidelines, real-data illustration, honest limitations.
- ⚠️ **May ask**: "How do I choose a quiet window in practice?" (You've added the example—this helps, but reviewer may want a decision flowchart or checklist.)
- ⚠️ **May ask**: "What sample size is needed for stable frailty estimation?" (Consider brief note in §2.5 or §5 Limitations if not already present.)

### Reviewer 3 (Survival Analysis Specialist)
- ✅ **Will appreciate**: Clear differentiation from Cox/RMST/flexible hazards, cumulative estimand motivation.
- ⚠️ **May ask**: "How does KCOR relate to Aalen additive hazards?" (You've positioned against time-varying Cox—consider brief note on additive hazards if space permits.)
- ⚠️ **May ask**: "What happens if quiet window is too short?" (You've covered this via diagnostics, but reviewer may want explicit guidance on minimum window length.)

---

## Decision Rationale

**Recommendation**: **Send for review**

**Reasoning**:
- Methodological contribution is clear and novel
- Validation is comprehensive and appropriately scoped
- Limitations are honestly presented
- Recent edits address the most obvious reviewer concerns (non-causal framing, quiet-window subjectivity, practical usability)
- Writing is clear and well-structured
- Reproducibility is addressed

**Expected outcome**: Likely **minor revision** after review. The paper is strong, but reviewers will likely request:
1. Additional guidance on quiet-window selection (decision criteria or flowchart)
2. Brief survival interpretation guide
3. Possibly a simulation under alternative frailty structures

**Confidence**: High. This is a well-executed methods paper that fits *Statistics in Medicine*'s scope. The recent edits (especially the non-causal framing and practical guidelines) address the most critical concerns proactively.

---

## Final Checklist (Before Submission)

- [x] Non-causal framing is explicit and prominent
- [x] Quiet-window example is concrete and operational
- [x] Practical guidelines section exists
- [x] Abstract is streamlined and clear
- [ ] Repository URL/DOI explicitly stated (verify)
- [ ] Word count verified (12,100 excluding Abstract/References/Supplement)
- [ ] All figures/tables cited in order
- [ ] Reference style matches journal guidelines exactly

---

**Bottom line**: This paper is ready for submission. The recent edits have addressed the most critical concerns. Expect minor revision requests focused on additional practical guidance and interpretation aids, but the core methodology and validation are strong.

