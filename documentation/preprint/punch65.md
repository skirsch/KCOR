Please make a narrow revision pass on `documentation/preprint/paper.md` and matching language in `documentation/preprint/supplement.md`.

Important constraints:
- Do NOT rewrite the paper broadly.
- Do NOT change the estimator architecture.
- Do NOT alter results, tables, figure data, or conclusions beyond the specific wording/clarification fixes below.
- Preserve the current v7.5 structure and notation unless the edits below require a local clarification.
- Keep the tone methods-journal appropriate.

Goal:
Fix four known weaknesses in the current draft:
1. sharpen the identifiability caveat for theta0
2. add a short derivation/explanation for the key recursion used in the delta-iteration step
3. state clearly that frailty variance is estimated separately for each cohort
4. reframe the “KCOR goes to 1” language as a diagnostic implication, not as proof of correctness

Make the edits directly in the manuscript and SI where relevant.

--------------------------------
A. IDENTIFIABILITY CAVEAT FOR theta0
--------------------------------

Problem:
The paper currently says theta0 is estimated from the data, but it does not state sharply enough that a constant proportional hazard effect inside the quiet window is observationally confounded with theta0 under minimal data.

Tasks:
1. In the main paper, strengthen the identifiability wording in the most natural locations:
   - `1.5`
   - `2.1`
   - `2.4.4`
   - `4.1`
2. Add one explicit sentence along the following lines, adapted to manuscript style:
   - “Under minimal aggregated data, enrollment-time frailty variance theta0,d is identifiable only conditional on the working assumption that no constant multiplicative hazard effect operates within the quiet-window identification regime; such an effect is observationally confounded with frailty-induced curvature over short horizons.”
3. Make sure the text does not overstate that theta0 is purely data-identified without model assumptions.
4. Mirror the same idea in the SI diagnostics/identifiability language where appropriate, especially in:
   - `S2`
   - `Table S3`
5. Keep this framed as a limitation of identifiability, not as a fatal flaw.

--------------------------------
B. DERIVE / EXPLAIN THE KEY RECURSION
--------------------------------

Problem:
The recursion in `2.5 Step 2` appears as a construction, but the manuscript does not clearly show that it comes from inversion of the gamma-frailty hazard relation.

Tasks:
1. In `2.4.2` or immediately before `2.5 Step 2`, add a short derivation or explanatory paragraph showing that under the gamma-frailty working model:
   - observed hazard satisfies something like
     `h_obs,d(t) = h0,d(t) / (1 + theta0,d * H0,d(t))`
   - therefore the reconstructed baseline hazard satisfies
     `h0,d(t) = h_obs,d(t) * (1 + theta0,d * H0,d(t))`
2. Make clear whether this relation is exact under the working model or a direct consequence of the gamma-frailty geometry being assumed.
3. Then make the wording in `2.5 Step 2` explicitly reference that derivation, so Step 2 reads as a justified inversion step rather than an unexplained update rule.
4. Keep the added derivation short, journal-style, and readable.

--------------------------------
C. MAKE COHORT-SPECIFIC theta0 EXPLICIT
--------------------------------

Problem:
The paper implies cohort-specific frailty variance, but this should be unmistakable.

Tasks:
1. Strengthen wording in the main paper to make clear that:
   - each cohort d has its own theta0,d
   - theta0,d is estimated independently for each cohort
   - there is no pooled/shared theta across cohorts in the core estimator
2. Check and tighten wording in:
   - Abstract
   - `2.1`
   - `2.5`
   - `2.9.1`
   - `2.11`
   - Table 4
   - Table 5
3. If needed, add one explicit sentence in Methods:
   - “Frailty variance is cohort-specific in KCOR and is estimated separately for each cohort rather than pooled across comparison groups.”
4. Make sure the SI uses the same language.

--------------------------------
D. REFRAME “GOES TO 1” AS DIAGNOSTIC, NOT PROOF
--------------------------------

Problem:
The current null-behavior language can be read as if convergence toward 1 proves the method is correct.

Tasks:
1. Find all places where the manuscript or SI says or implies that:
   - KCOR staying near 1
   - flattening to 1
   - asymptoting to 1
   proves correctness, validates the method, or establishes that confounding is removed.
2. Rewrite these so they instead say that near-1 behavior is:
   - expected under the working model and null conditions
   - a diagnostic implication or falsifiable consistency check
   - not by itself proof that the model is true or that all confounding has been removed
3. Pay special attention to:
   - Abstract
   - `3`
   - `3.1.1`
   - `2.11.1`
   - Figure 3 caption
   - Table 6 caption / notes
   - SI figures S2 and S3 and related text
4. Prefer wording like:
   - “is consistent with”
   - “behaves as expected under”
   - “provides a diagnostic check under the working model”
   Avoid wording like:
   - “proves”
   - “shows the method is correct”
   - “establishes that confounding is removed”

--------------------------------
E. QA AFTER EDITS
--------------------------------

After making the edits, do a focused QA pass:
1. Ensure the added identifiability caveat is consistent across main text and SI.
2. Ensure the derivation and Step 2 recursion use the same symbols and are mathematically aligned.
3. Ensure every mention of theta0 remains cohort-specific unless explicitly referring to a generic scalar in a simulation.
4. Ensure no sentence still implies that near-1 null behavior is proof of correctness.
5. Keep the overall paper length increase modest.

Deliverable:
- Update the manuscript and SI in place.
- Then provide a short summary of exactly what was changed, with file/section references.
