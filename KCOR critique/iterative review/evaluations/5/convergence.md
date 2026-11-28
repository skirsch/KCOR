# Series 5 - reduced evaluation of m102

## Design and scope

Two independent blind GPT-6.2 Astra High reviews and one GPT-6.2 Astra Medium resolution review used the frozen m102 hypothetical post-approval candidate. The blind reviewers were instructed to inspect only m102, with the original question verbatim. The resolution reviewer received the earlier objection records. This is three fresh reviews, not the full earlier multi-stage protocol, and is not statistically equivalent to the 48-turn series 4. No new method changes, PR update, expensive simulations, or further evaluation round were commissioned.

- [Blind A](blind-a.md)
- [Blind B](blind-b.md)
- [Resolution review](resolution.md)
- [Input manifest](inputs.json)

## Convergence

Both blind reviewers independently endorse the proportional gamma inversion conditional on its model assumptions, and recognize the fixed-slope identification identity under its stated proportional history. The resolution reviewer confirms the original zero-row false success, alpha=1 branch discontinuity, gap double accumulation and first quiet-window short-seed obstruction were corrected. Table 8 was independently recomputed from both stored 999-draw arrays: all twelve intervals contain one, all stored flags zero, all reported JSON percentiles exactly reproduced. This is an empirical compatibility check, not proof of calibration or an exact null.

There is substantial convergence on the central algebra, but **not yet on ordinary pipeline/reporting correctness**. Both blind reviews independently found the invalid reference fallback, mismatch in ordinary bootstrap sampling design, unsupported first-order slope cancellation, and the eligible-death gate. The third reviewer independently reproduced the gate and early-anchor issues. Read the raw reviews for the individual numerical counterexamples.

## New or sharpened findings for discussion

These IDs are triage, not adopted changes. Severity concerns the affected path; no claim that any new issue altered Table 8. A/B/R denote the two blind reviews and targeted resolution review. Findings not independently replicated are provisional, even if supported by a source trace.

| ID | Severity | Finding | Related root / earlier objection | Review |
|---|---|---|---|---|
| 5.1 | High for optional wave fit | The final mask excludes wave rows, but their deaths and quiet-bin counts still contribute to the seed-based minimum-death/weak-information flags. All three independently reproduced this. | 4.1, eligibility consistency | A B R |
| 5.2 | Moderate | Zero early mortality prevents pooled optimization despite later positive quiet data. Separate an initialization from a required substantive anchor. | 4.5, sparse information | A B R |
| 5.3 | High for reporting | Undefined/zero reference ratio silently becomes 1 and a reported anchored ratio becomes an unanchored one; primary and optional estimands not separately exported. | 2.8, reporting/estimand alignment | A B |
| 5.4 | High for ordinary registry inference | Standard converter emits sex O, DCCI and overlapping aggregates that default bootstrap rejects. A single unsupported sheet can make intervals unavailable. | 4.6, production integration | B |
| 5.5 | Moderate/high for inference | Ordinary bootstrap holds birth-year/dose sizes fixed, while paper describes pooled resampling with variable sizes; Table 8 actually uses pooled component sampling. Both designs may be defensible; establish intended target. Nondeath-exit term missing in prose. | 1.3, 4.6, uncertainty specification | A B |
| 5.6 | Potentially high for source cohorts | Converter's non-mRNA product exclusion examines later doses before constructing earlier enrollment cohorts. Verify original population intent and actual affected records. | Fixed-cohort construction, new code-path issue | B |
| 5.7 | High for active-wave all-ages results | All-ages output recomputes ordinary inversion after conditional survivor correction; also a potential missing draw identifier. | 1.2, 4.6, reporting bypass | B |
| 5.8 | Moderate | Proportional gap diagnostic reports pre-optimization offsets rather than final offsets; shape diagnostics use straight elapsed time and ordinary inverse rather than the active Gompertz/offset model. Status vs failure policy needs clear distinction. | 3.1, 4.4, diagnostic alignment | A B |
| 5.9 | Moderate scientific claim | Common Gompertz slope misspecification need not cancel to first order. Both provide counterexamples; confirm actual sensitivity evidence or narrow wording. | 1.1, slope/model conditionality | A B |
| 5.10 | Moderate for optional alpha localization | Historical moment surrogate and cancellation of current wave intensity do not establish invariance to previous wave history. | 4.7, already deferred alpha identification | A B |
| 5.11 | Minor to moderate, edge cases | Death-fraction clipping, horizon-wide inversion cap, flagged fallback and secondary analytic interval limitations. The cap is already disclosed but not flagged; extending follow-up may change earlier inverted values. | 4.9, 4.6 | A B |
| 5.12 | Moderate reproducibility | Quiet scan relies on changed default rather than explicit forced alpha; preprint Makefile references missing assets and publication target. | 2.5, 4.7, documentation packaging | A B (build); A (scan) |

## Earlier objections still open or conditional

- 1.1: observational equivalence under arbitrary baseline trajectories; narrower claims, not a new inversion error.
- 1.3, 4.6: coverage and ordinary bootstrap/reporting robustness remain conditional. Dedicated 999-draw Table 8 is not a full test of default registry bootstrap. The previous nonlinear coverage was 90/100 (82.4-95.1%); stress mean 1.0402 against truth one.
- 1.4: Table 8 endpoints and intervals now reproduced; registry age-shift controls still have no guaranteed numerical null. This is validation scope.
- 2.5: scan denominator and historical results corrected, but replay mode now needs explicit selection.
- 2.7: universal contamination threshold narrowed; low-information bias and finite-sample frailty resolution remain model-dependent statistical limits.
- 3.2: modeled proportional pre-entry history corrected; unidentified/unmarked nonproportional shocks remain a model limitation.
- 4.1, 4.5: original exact counterexamples addressed; narrower eligibility loopholes remain (5.1, 5.2).
- 4.3: documented wave misspecification/attribution limitation, alternative allocation deferred by design.
- 4.7: primary configuration fixed; alpha estimation deferred.
- 4.9: documentation fixed; extreme-cap reporting and edge-case numerical behavior still open.

## Earlier objections whose original requested corrections hold

1.2 (original algebra), 1.5/2.1 (Cox benchmark disclosures), 1.6/2.2 (scaling and weekly fit/inversion), 2.3/2.4 (gap and first wave week), 2.5 (denominator), 2.6/3.3 (positive-control generator intent and fitted replay), 2.8 (anchor convention disclosure), 2.9 (fitting/pooling disclosure), 2.10 (illustration selection disclosure), 3.1 (reset), 4.2 (alpha=1 fit continuity), 4.4 (offset accumulation arithmetic), 4.7 (configuration and comparisons), 4.8 (validation scope), 4.9 (specified text/diagram citations). See [resolution](resolution.md) for qualifications. Closing an original defect does not close different downstream issues in the same path.

## Recommended order for discussion

Start with 5.1, because all three independently found it and it is a narrow consistency defect. Then assess 5.3, 5.4, 5.6 and 5.7 (ordinary reporting, input construction, optional reporting bypass). Remaining items include design decisions, validation limits, and documentation. Preserve original intent; do not adopt suggested redesign merely because a reviewer recommends it.

The candidate and existing draft PR remain unchanged.
