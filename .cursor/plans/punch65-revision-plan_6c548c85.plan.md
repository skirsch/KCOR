---
name: punch65-revision-plan
overview: Plan the next narrow revision pass for the KCOR manuscript and supplement based on `documentation/preprint/punch65.md`, focusing on four specific reviewer-facing weaknesses without changing architecture, results, figures, or conclusions beyond local wording/clarification fixes.
todos:
  - id: main-methods-clarifications
    content: Revise the main paper’s Abstract, identifiability sections, and Step 2 derivation references to lock the new caveat, inversion explanation, and cohort-specific theta0,d semantics.
    status: completed
  - id: main-validation-overclaim-sweep
    content: Propagate the new wording into the paper’s operational sections, validation text, and figure/table captions so near-1 behavior is framed as diagnostic rather than proof.
    status: completed
  - id: si-mirror-pass
    content: Align S2, Table S3, SI figures S2/S3, and nearby SI diagnostics/defaults text with the finalized main-text semantics.
    status: completed
  - id: final-manuscript-qa
    content: Perform a narrow cross-file QA sweep for notation drift, recursion-symbol mismatch, cohort-specific theta0 wording, and leftover proof-like near-1 language.
    status: completed
isProject: false
---

# Punch65 Narrow Revision Pass

## Scope

Use [documentation/preprint/punch65.md](documentation/preprint/punch65.md) as the execution checklist. Treat this as a constrained clarification pass on [documentation/preprint/paper.md](documentation/preprint/paper.md) and [documentation/preprint/supplement.md](documentation/preprint/supplement.md): no broad rewrite, no estimator-architecture changes, and no edits to results, tables, figure data, or conclusions beyond the requested wording and derivation clarifications.

## Phase 1: Lock Main-Text Methods Semantics

Edit [documentation/preprint/paper.md](documentation/preprint/paper.md) in this order:

1. `Abstract`
2. `1.5`
3. `2.1`
4. `2.4.2`
5. `2.4.4`
6. `2.5`
7. `4.1`

Goals:

- Sharpen the identifiability caveat so `theta0,d` is presented as conditionally identifiable only under the working assumption that no constant multiplicative hazard effect operates inside the quiet-window identification regime.
- Add one concise derivation/explanation in `2.4.2` showing the gamma-frailty hazard relation and its inversion, then make `2.5 Step 2` explicitly point back to that derivation so the recursion reads as justified inversion rather than an unexplained update rule.
- Clarify explicitly that the inversion relation used in `2.5 Step 2` follows from the gamma-frailty working model and is not model-free.
- Make cohort specificity unmistakable in the Methods framing by stating that each cohort has its own `theta0,d` and that frailty variance is estimated separately for each cohort rather than pooled across comparison groups.

Essential anchor to preserve while clarifying:

```text
h_obs,d(t) = h0,d(t) / (1 + theta0,d * H0,d(t))
h0,d(t) = h_obs,d(t) * (1 + theta0,d * H0,d(t))
```

Stop condition:

- The Methods sections can be read in isolation without implying that `theta0,d` is assumption-free, without ambiguity about Step 2, and without any suggestion of pooled/shared frailty variance across cohorts.

## Phase 2: Propagate The Same Semantics Into Operational And Interpretive Main-Text Sections

Still in [documentation/preprint/paper.md](documentation/preprint/paper.md), update the downstream sections and display items that inherit the Methods wording:

1. `2.9.1`
2. `2.11`
3. `2.11.1`
4. `3`
5. `3.1.1`
6. Figure 3 caption (`fig:cox_bias_kcor`)
7. `Table 4` / `tbl:notation`
8. `Table 5` / `tbl:KCOR_algorithm`
9. `Table 6` caption/notes / `tbl:cox_bias_demo`

Goals:

- Ensure every operational reference to frailty variance uses cohort-specific language (`theta0,d`, estimated separately per cohort).
- Reframe all “KCOR goes to 1” or “near 1” language as expected diagnostic behavior under the working model and null conditions, not proof that the model is true or that confounding has been removed.
- Keep validation claims reviewer-safe by preferring phrases such as “is consistent with,” “behaves as expected under,” and “provides a diagnostic check under the working model.”

## Phase 3: Mirror The Clarifications In The SI

After the main paper wording is stable, update [documentation/preprint/supplement.md](documentation/preprint/supplement.md) in this order:

1. `S2`
2. `Table S3` / `tbl:si_identifiability`
3. adjacent `Table S1` / `tbl:si_assumptions` and `Table S2` / `tbl:si_diagnostics` rows if needed for consistency
4. `S4.2.1`
5. SI Figure S2 caption (`fig:si_kcor_bias_vs_theta`)
6. SI Figure S3 caption (`fig:neg_control_synthetic`)
7. `tbl:si_defaults`, `S5.1`, and nearby cohort-parameter wording as needed

Goals:

- Mirror the same identifiability caveat from the main text, especially where the SI currently treats `theta0` as identified from diagnostics or quiet-window curvature alone.
- Keep SI notation and prose aligned with cohort-specific `theta0,d` semantics.
- Remove any SI wording that turns near-1 behavior into proof rather than a falsifiable diagnostic implication under the working model.
- Ensure the SI makes no stronger claim than the main paper.

## Phase 4: Focused QA Sweep

Run one last manuscript-wide pass across [documentation/preprint/paper.md](documentation/preprint/paper.md) and [documentation/preprint/supplement.md](documentation/preprint/supplement.md).

Check for:

- identical identifiability caveat meaning in main text and SI
- exact symbol alignment between the added derivation and `2.5 Step 2`
- cohort-specific `theta0,d` wording everywhere except places intentionally using a generic scalar in simulations
- no reversion from `theta0,d` to `theta` in SI tables, captions, or defaults
- any sentence, caption, note, or table entry that still implies that KCOR near 1 proves correctness or establishes that confounding is removed
- length discipline so the pass remains modest and local rather than expanding into a rewrite

## Success Criteria

- [documentation/preprint/paper.md](documentation/preprint/paper.md) and [documentation/preprint/supplement.md](documentation/preprint/supplement.md) both present the same reviewer-safe story on identifiability, the Step 2 inversion, cohort-specific frailty variance, and diagnostic interpretation of near-1 KCOR behavior.
- The revised manuscript addresses all four punch65 requests directly while preserving the current v7.5 structure and estimator architecture.
- No figure caption, table note, or SI diagnostic block silently preserves the old overclaiming language.
