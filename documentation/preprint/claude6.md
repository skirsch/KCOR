Yes, at **12,100 words** (excluding abstract, references, and appendices), the paper is quite long for *Statistics in Medicine*. Most methodology papers in that journal run 6,000-8,000 words. Here are prioritized recommendations for shortening:

## High-Impact Cuts (Target: ~3,000 words)

**1. Consolidate redundant validations (save ~1,000 words)**
- You have three validation approaches: simulations, negative controls, AND positive controls
- Consider moving one validation type entirely to supplementary materials
- Recommendation: Keep simulations + negative controls in main text (they demonstrate the core problem). Move positive controls to supplement

**2. Streamline the conceptual introduction (~500 words)**
- Sections 1.1-1.4 repeat key concepts (selection, depletion, curvature) multiple times
- Compress 1.2-1.4 into a single "Motivation" section focusing on why standard methods fail
- The current separate treatment of "what is depletion" vs "why it matters" vs "how it creates curvature" can be unified

**3. Reduce appendix material referenced in main text (~800 words)**
- Appendices C and D are quite detailed
- If key diagnostic content is in appendices, consider whether some main-text material can move there instead
- Specifically: the detailed assumption diagnostics (D.1-D.8) could be condensed to a summary table

**4. Compress methods notation/derivation (~500 words)**
- Section 2.2-2.3 includes full mathematical development
- Could present final forms with key equations, moving derivations to supplement
- The gamma-frailty inversion is central but some intermediate steps could be abbreviated

**5. Shorten discussion (~200 words)**
- Section 6 revisits points made in introduction
- Focus on: novel contributions, limitations, future directions
- Remove conceptual review of why depletion matters (already established)

## Medium-Impact Edits (Target: ~500 words)

**6. Tighten figure legends and in-text descriptions**
- Some figures have extensive captions that duplicate main text
- Example: Figure descriptions in Section 4 could reference figures more briefly

**7. Reduce methodological alternatives discussion**
- Section 5 on Cox/IPW/etc could be more concise
- Make the key point: these methods assume away the problem KCOR addresses

## Structure Recommendation

Given Statistics in Medicine's style, consider this reorganization to hit ~7,500-8,000 words:

**Main paper:**
1. Introduction (condensed): ~1,500 words
2. Methods + Assumptions: ~2,000 words  
3. Simulation validation: ~1,500 words
4. Negative controls: ~1,500 words
5. Discussion: ~1,000 words

**Supplementary materials:**
- Full mathematical derivations
- Positive control details
- Extended diagnostic framework
- Implementation details

This would make the main paper more accessible while keeping all technical content available for reviewers and interested readers.

Would you like me to draft specific condensed sections to show how this could work?

Looking at the appendices, they're actually **not included in your 12,100-word count** (you specified "excluding Abstract, References, and Appendices"). So they're bonus content that won't directly help you hit a lower word count for the main manuscript.

However, the appendices do present some strategic considerations:

## Current Appendix Structure

From what I can see in the file:

- **Appendix D**: Diagnostic framework for assumptions A1-A5 (sections D.1-D.8) - this is quite detailed
- **Appendix E**: Reference implementation and default settings

## Strategic Issues

**1. Main text references to appendices**
- If you're frequently pointing readers to appendices for critical information (like assumption diagnostics), you might be able to move some main-text material TO the appendices instead
- This would reduce your main word count while keeping technical completeness

**2. Appendix length and journal norms**
- Statistics in Medicine generally allows substantial supplementary materials
- However, extremely long appendices can still create reviewer fatigue
- If Appendix D is running 1,500+ words on diagnostics alone, consider whether all of it needs to be there

## Recommendations

**Option A: Use appendices strategically to reduce main text**
- Move full mathematical derivations from Section 2 to appendix
- Move detailed simulation specifications to appendix (keep results/figures in main)
- Move one entire validation study (positive controls) to appendix
- This could save 2,000-3,000 words from your 12,100 count

**Option B: Streamline appendices themselves**
- Appendix D (diagnostics) could become a condensed table rather than 8 subsections
- Summary: "Assumption | Diagnostic signal | Failure mode" in tabular form
- Full diagnostic discussion could go to supplementary online materials

**Option C: Reorganize what's "appendix" vs "supplement"**
- Appendices (in the manuscript PDF): only material essential for understanding/reproducing
- Supplementary materials (separate file): extended validations, code, additional analyses

Would you like me to look at specific appendix sections to suggest what could be condensed or restructured?

Now I have a complete picture. Here's my detailed analysis of your appendices with specific recommendations:

## Appendix Summary & Word Counts (estimated)

**Appendix A: Mathematical derivations** (~600 words)
- Sections A.1-A.4: Frailty mixing, gamma identity, inversion, variance propagation

**Appendix B: Control-test specifications** (~1,200 words)
- B.1-B.6: Detailed specifications for all simulation/control studies

**Appendix C: Additional figures and diagnostics** (~800 words)
- C.1-C.6: Fit diagnostics, residual analysis, parameter stability, empirical applications

**Appendix D: Diagnostic framework** (~1,400 words)
- D.1-D.8: Detailed diagnostics for each assumption (A1-A5)

**Appendix E: Implementation details** (~200 words)
- Reference implementation table

**Total appendix content: ~4,200 words**

## Major Condensation Opportunities

### 1. **Appendix D can be radically condensed** (save ~1,000 words)

You already have **Table D.1** (line 1111) that summarizes all five assumptions with diagnostics in compact form. The sections D.1-D.8 are verbose expansions of what's already in the table.

**Recommendation:**
- Keep only Table D.1 in Appendix D
- Add 1-2 paragraphs explaining the "diagnostic enforcement" philosophy (currently D.8)
- Move detailed diagnostic discussions to **online supplementary materials**
- This reduces Appendix D from ~1,400 to ~300 words

### 2. **Appendix B is excessively detailed** (save ~600 words)

Sections B.1-B.6 provide parameter-by-parameter specifications for simulations. This level of detail belongs in code repositories, not manuscripts.

**Recommendation:**
- Condense B.1-B.6 into a **single summary table** with columns: Study | Design | Key parameters | Purpose
- Example row: "Negative control (synthetic) | Pathological frailty mixture | θ=[0.2,0.2,0.2,0.2,0.2] vs [0.3,0.2,0.2,0.2,0.1], z=[1,2,4,6,10] | Test null stability under extreme selection"
- Move full specifications to GitHub/Zenodo data repository
- Reduces from ~1,200 to ~400 words

### 3. **Appendix C has redundancy** (save ~400 words)

**Current issues:**
- C.1-C.4 repeat diagnostic concepts already in Appendix D
- C.5 (age stratification robustness) has 3 nearly identical figure descriptions
- C.6 (Czech empirical application) is extensive but stated as "not validation"

**Recommendation:**
- **Merge C.1-C.4 into Appendix D's diagnostic table** (add column for "How to check")
- **Condense C.5**: Single figure showing all three age strata in panels, brief caption
- **Move C.6 entirely to supplement** - it's an illustrative application, not core validation
- Reduces from ~800 to ~400 words

### 4. **Appendix A is actually fine** (minimal savings)

This is the cleanest appendix. The mathematical derivations are appropriately concise and essential for technical readers. I'd keep it as-is.

## Specific Rewrite: Condensed Appendix D

Here's how I'd restructure it:

---

## Appendix D: Diagnostic Framework

KCOR relies on five explicit assumptions (A1–A5), each with observable diagnostics and well-defined failure modes. When assumptions are violated, KCOR signals reduced interpretability through instability, poor fit, or residual structure rather than producing misleading contrasts—a design feature we term **diagnostic enforcement**.

[Insert Table D.1 here - already exists in your manuscript]

**Diagnostic coherence.** Several diagnostics operate across multiple assumptions. For example, KCOR(t) stabilization and post-normalization linearity provide evidence for both A4 (gamma adequacy) and A5 (quiet-window validity). These are observable consequences of successful frailty normalization, not additional assumptions.

**Sparse data.** KCOR does not impose minimum sample size requirements, but reliable parameter estimation requires sufficient events within the quiet window. Sparse-data regimes manifest as parameter instability under window perturbations and non-stabilizing KCOR(t) trajectories, explicitly signaling loss of identifiability.

For detailed diagnostic procedures and threshold recommendations, see Supplementary Methods S2.

---

**This reduces Appendix D from 1,400 words to ~250 words.**

## Summary of Total Savings

If you implement all recommendations:

| Appendix | Current | Proposed | Savings |
|----------|---------|----------|---------|
| A | 600 | 600 | 0 |
| B | 1,200 | 400 | 800 |
| C | 800 | 400 | 400 |
| D | 1,400 | 250 | 1,150 |
| E | 200 | 200 | 0 |
| **Total** | **4,200** | **1,850** | **2,350** |

## Strategic Note

Your appendices aren't counting against your 12,100-word main text, BUT:

1. **Reviewer burden**: 4,200 words of appendices + 12,100 main = 16,300 total words is A LOT to review
2. **Main text leverage**: Condensing appendices frees you to reference them more efficiently from main text, potentially allowing main text cuts
3. **Modern norms**: Detailed specs belong in code/data repositories with DOIs, not PDF appendices

Would you like me to draft the condensed versions of specific appendices?
