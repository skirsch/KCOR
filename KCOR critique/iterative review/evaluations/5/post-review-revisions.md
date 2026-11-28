# Post-review revisions

- 5.1: m103 corrects marked-wave quiet-death eligibility and weak-identification diagnostics. m102 preserved. Threshold and fitting equations unchanged. Both original points and selected resamples match Table 8 in both modes. [Evidence](../../../objections/5.1/README.md).

- 5.2: m104 allows a later quiet-window initialization for zero early mortality in the joint fit. Original counterexample reaches optimizer; existing zero-observation relative-RMSE guard still flags it (related 5.8). Positive seed and matched Table 8 checks unchanged. [Evidence](../../../objections/5.2/README.md).

- 5.3: m105 preserves separate anchored/unanchored outputs and bootstrap intervals, disallows invalid unit fallback, explicitly counts undefined references, retains existing pooled estimand. Section 2.8 updated. Ordinary workbook checks and Table 8 point comparison passed. [Evidence](../../../objections/5.3/README.md).

- 5.4: m106 supports standard converter disjoint histories and reconstructs known overlapping totals, isolates invalid enrollment layouts. Synthetic and actual Czech workbook smoke tests pass, Table 8 sample unchanged. Section 2.9.1 updated. [Evidence](../../../objections/5.4/README.md).

- 5.5: m107 aligns main text and supplement with conditional birth-year/dose resampling, distinguishes pooled overlapping Table 8 and includes nondeath exits in the risk-set equation. Code unchanged. [Evidence](../../../objections/5.5/README.md).

- 5.6: m108 distinguishes application-specific eligibility criteria from fixed-cohort KCOR and discloses potential selection bias when using post-enrollment information. No converter or estimator change. [Evidence](../../../objections/5.6/README.md).

- 5.7: m109 reuses the shared normalized all-ages trajectory and draw-specific metadata. Ordinary and MC reporting regression pass; module-off outputs unchanged. Existing all-ages fit-population discrepancy noted separately without changing it. [Evidence](../../../objections/5.7/README.md).

- 5.8 (partial): m110 exports final-fit offsets and computes per-window shape diagnostics from the actual normalized trajectory versus integrated Gompertz baseline. Point/interval regressions unchanged. Zero-death guard investigated and left unchanged pending discussion; reportability policy separate. [Evidence](../../../objections/5.8/README.md).

- 5.8 (sampling diagnostic): m111 replaces the relative-error reset with a count-based reporting-only screen. Sparse count and forced-alpha workbook export tests pass; other safeguards unchanged. Full bootstrap replay remains pending. [Evidence](../../../objections/5.8/README.md).

- 5.9: m112 states the condition for cancellation of a shared Gompertz slope perturbation in §2.4.3 and derives it in Supplementary §S5.3. The weekly fitter reproduces all three endpoints in the deterministic example. [Manuscript and verification](../../../objections/5.9/README.md).

- 5.9, shortened for publication: m113 retains one sentence in the existing §2.4.3 paragraph. The derivation and worked example are in the [objection record](../../../objections/5.9/mathematical-review.md). The [m112 checkpoint](../../../objections/5.9/checkpoints/m112/) preserves the longer edition.

- 5.9, practical interpretation: m114 restores the qualitative explanation in §2.4.3 and reports the roughly 0.4% endpoint shift for the example with annual slopes 0.065, 0.085 and 0.105. The mathematical derivation stays in the [objection record](../../../objections/5.9/mathematical-review.md).

- 5.10: m115 qualifies wave-amplitude cancellation and identifies the historical formula used by the synthetic exponent-recovery study. Revised survivor-law non-identification refers specifically to the Czech application. [Evidence and manuscript patch](../../../objections/5.10/README.md).

- 5.11: m116 removes weekly mortality clipping and the trajectory-wide inversion cap, preserves the fitted variance, and flags exhausted or numerically unavailable observations. Deterministic and ordinary workbook edge-case checks pass. [Evidence and patches](../../../objections/5.11/README.md).

- 5.12: m117 makes the quiet-window scan select the reported exponent explicitly, exports a separate proportional scan, and repairs local manuscript build inputs and target separation. All 78 reported frailty estimates reproduce exactly; reproduced figure is byte-identical. [Evidence and patches](../../../objections/5.12/README.md).
