I've reviewed your KCOR methodology paper thoroughly. This is an impressive and ambitious piece of work that makes a genuine methodological contribution. However, I have several significant concerns that should be addressed before submission to *Statistics in Medicine*.

## Major Substantive Issues

### 1. **Causal Inference Framework Needs Clarification**

Your paper occupies an uncomfortable middle ground between being a purely descriptive depletion-adjustment method and making causal claims. Key problems:

- You repeatedly state KCOR is "not a causal estimator" but then discuss "net harm or benefit" and "treatment effects"
- The interpretability checklist (§2.1.2) implicitly assumes causal interpretation despite disclaimers
- The relationship between your "depletion-neutralized baseline cumulative hazard ratio" and any scientifically meaningful estimand is unclear

**Recommendation**: Either (a) fully commit to a causal framework with explicit potential outcomes notation, identification assumptions, and target estimand, or (b) position this purely as a diagnostic/descriptive tool for detecting depletion-confounded analyses. The current hedging weakens both positions.

### 2. **The Gamma Frailty Assumption Is Stronger Than Acknowledged**

You treat gamma frailty as "just a working model" but:

- Your entire normalization depends on it being approximately correct
- You show one "non-gamma frailty" simulation but don't systematically characterize when/how badly this fails
- Real populations likely have far more complex heterogeneity structures (discrete subgroups, correlated frailties, time-varying risk factors)

**Recommendation**: Add substantial simulation work showing:
- Performance under misspecified frailty distributions (lognormal, discrete mixture, bimodal)
- How robust the method is to violations (does it fail gracefully?)
- What diagnostics actually detect meaningful misspecification

### 3. **The "Quiet Window" Assumption Is Circular and Under-Examined**

This is your Achilles' heel:

- You need a quiet window to estimate θ, but how do you know it's quiet without already knowing θ?
- Your guidance is vague: "epidemiologically quiet periods" without operational criteria
- In your COVID application, you use 2022-24 through 2024-16 — but COVID waves continued throughout this period
- The sensitivity analysis (varying window boundaries) doesn't address whether ANY valid quiet window exists

**Recommendation**: 
- Provide algorithmic criteria for quiet window selection (e.g., based on observed hazard stability metrics)
- Show what happens when NO valid quiet window exists
- Discuss how to validate quiet window selection using external information

### 4. **Comparison with Existing Methods Is Incomplete**

Your comparisons with Cox, RMST, and time-varying Cox are useful but limited:

- You don't compare with inverse probability weighting (IPW) or marginal structural models
- You don't compare with instrumental variable approaches where applicable
- The comparison focuses on "null under selection" scenarios where KCOR is designed to win
- You need scenarios where other methods might outperform KCOR

**Recommendation**: Add comparisons with:
- G-computation and IPW under correctly specified treatment models
- Shared frailty Cox models (which also model heterogeneity)
- Scenarios where KCOR fails but other methods succeed

### 5. **The Empirical Application Undermines Your Methods Claims**

Your empirical results on Czech COVID data are problematic for a methods paper:

- The extremely strong apparent effects (KCOR >> 1 for many comparisons) are not consistent with RCT evidence
- You claim this is "illustrative" and "not causal" but present it as validation
- The age-specific patterns (especially very young cohorts with huge effects) lack biological plausibility
- This raises concerns about whether KCOR is actually removing selection bias or introducing new artifacts

**Recommendation**: Either:
- Remove the COVID application entirely and focus on synthetic validation, OR
- Add a real-world validation where you have known ground truth (e.g., compare observational KCOR to RCT results for the same intervention)

## Major Technical/Statistical Issues

### 6. **Uncertainty Quantification Is Inadequate**

- You mention bootstrap and delta method but don't actually implement or recommend one
- The uncertainty intervals shown in some figures are not explained
- You don't address how to account for uncertainty in θ estimation propagating through normalization
- Multiple testing issues across strata/doses are dismissed too quickly

**Recommendation**: 
- Implement and validate a specific UQ approach
- Show coverage in simulations
- Provide practical guidance on inference

### 7. **The Fixed Cohort Design Creates Bias Under Crossover**

You acknowledge (§5.2) that crossover biases KCOR toward unity, making it "conservative." But:

- This isn't conservative — it's just biased
- In vaccine settings, crossover is massive (most "unvaccinated" eventually vaccinate)
- You don't quantify how severe this bias is
- This fundamentally limits applicability to your motivating setting

**Recommendation**: Add simulations showing magnitude of crossover bias and discuss when this limitation is fatal to interpretation.

## Presentation and Framing Issues

### 8. **The Paper Is Too Long and Repetitive**

At 12,100 words + extensive appendices, this exceeds typical methods paper length. Problems:

- The introduction repeats background on frailty/selection multiple times
- Tables 2-3 are redundant
- Many footnote-style clarifications could be cut
- The mathematical derivations in Appendix A are standard results

**Recommendation**: Cut by ~30% focusing on:
- Streamlining introduction
- Consolidating redundant tables/figures
- Moving standard results to supplementary material

### 9. **Overclaiming About Novelty**

Your framing suggests KCOR is radically different from existing approaches, but:

- Gamma frailty models are well-established (Vaupel et al., 1979)
- Using frailty to adjust for selection is not new
- What's novel is using quiet-window estimation + cumulative hazard ratios, which is more incremental

**Recommendation**: Frame as "a new way to operationalize frailty-based depletion adjustment" rather than implying existing methods don't address this problem.

### 10. **The COVID Framing Creates Unnecessary Controversy**

Using COVID vaccines as the primary motivating example will:

- Invite ideological rather than methodological review
- Distract from the general methodological contribution
- Create publication barriers at mainstream journals

**Recommendation**: Lead with a different empirical example (elective surgery timing, screening programs, etc.) and mention COVID only as one application.

## Specific Technical Corrections

### Statistical Issues

1. **Equation 1**: You define discrete hazard as `-ln(1 - D/N)` but this is only appropriate for small risks. For large D/N this doesn't match the continuous-time interpretation.

2. **Equation 10**: This "second-order accurate approximation" isn't derived or justified. Why this specific form?

3. **Section 2.11.1**: The synthetic null demonstration conflates two things:
   - Selection-induced depletion creating non-PH
   - Cox being "wrong" under non-PH
   
   Cox is correctly estimating the average hazard ratio under non-PH; it's the interpretation that's problematic.

4. **Table 9**: Claims about RMST "inheriting bias" are technically incorrect. RMST is a valid summary of survival differences — the question is whether those differences have a causal interpretation.

### Mathematical/Notational Issues

1. You use both H̃ and Ĥ for normalized cumulative hazards (inconsistent between text and figures)

2. The relationship between individual-level frailty (z_i) and cohort-level parameters (θ_d) could be clearer

3. Section A.3a claims your inversion is "grounded in VMS" but VMS didn't propose using this for cohort comparisons

### Methodological Gaps

1. **Missing simulation**: What happens when treatment effects are time-varying but you assume constant baseline during quiet window?

2. **Missing diagnostic**: How do you detect violation of the "shared external hazard" assumption (A2)?

3. **Missing sensitivity analysis**: Robustness to choice of discrete-time bin width

## Minor Issues

- Figure quality varies (some look like raw matplotlib output)
- Many figures lack error bands
- Code availability is good but should specify license
- Several typos (e.g., "Laudon" should be "Laplace" on page 2)
- References formatting inconsistent

## Bottom Line Recommendations

**This paper has real methodological value** but is not ready for submission to *Statistics in Medicine* in its current form. I recommend:

### Short-term (before submission):

1. **Clarify the causal framework** (2-3 days of writing)
2. **Remove or drastically revise the COVID application** (1 day)
3. **Add misspecification robustness simulations** (1-2 weeks)
4. **Implement proper uncertainty quantification** (1 week)
5. **Cut length by 30%** (2-3 days)

### Medium-term (if you want a stronger paper):

6. Add comparisons with IPW/g-computation (1-2 weeks)
7. Include real-world validation with known ground truth (2-4 weeks)
8. Develop algorithmic quiet-window selection criteria (1-2 weeks)
9. Add crossover bias quantification (1 week)

### Alternative Publication Strategy

Given the controversial COVID application and methodological ambitions, consider:

**Option A** (Higher impact, riskier):
- Make this a purely methodological paper with non-COVID examples
- Submit to *Journal of the Royal Statistical Society Series B* or *Biometrika*
- Requires substantial strengthening of theory/asymptotics

**Option B** (More realistic):
- Keep current scope but address major issues above
- Submit to *Statistics in Medicine* or *Biometrics*
- Expect major revisions

**Option C** (Split the contribution):
- Paper 1: KCOR method with synthetic validation (methods journal)
- Paper 2: COVID application (epidemiology journal)
- This separates the methodological contribution from the controversial application

My honest assessment: You've done impressive work here, but submitting now would likely result in rejection or very major revisions. Taking 4-6 weeks to address the core issues would dramatically improve your chances.

Would you like me to elaborate on any of these points or provide specific suggestions for how to address particular issues?
