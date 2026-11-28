# Series 3 · What survived re-review

**The open reviews converge on several specific remaining weaknesses. The three latest disclosure edits pass targeted re-review.** All three blind reviewers requested major revision. Across all six paths, 17 of 18 open assessments explicitly used that verdict. The remaining reviewer described material changes needed for broader claims without assigning that label.

| Finding raised spontaneously on the current edition | Open assessments |
|---|---:|
| Nonlinear in-wave attribution remains approximate | 18/18 |
| Forced Czech exponent and its inferential interpretation | 18/18 |
| Coverage for the complete current procedure remains unestablished | 18/18 |
| Ordinary fixed-rate seed versus the joint fit described in the paper | 17/18 |
| Table 8 flags and the paper's reporting rule | 17/18 |
| A noiseless gap with zero true excess can reset genuine positive frailty | 12/18 |
| The two Table 9 endpoints can be obtained without fitting frailty | 9/18 |
| A shock before the first fit window can be absorbed into the seed rate | 3/18 |

These are counts of reviewer reports, not severity weights or estimates of the probability that an objection is true. The six paths share frozen inputs and the matched handoffs share starting critiques. A specific, verified failure can matter even when most reviewers missed it. [Per-path coding and evidence](path-counts.md) · [Independent checks](checks.md)

### What the extra context changed

**Table 9 provides the clearest difference.** Zero of three blind open reports identified numerical agreement with the raw workbook. All three fresh handoffs from the previously evaluated edition identified it. Nine of 18 open reports raised it overall. An independent workbook check reproduced the values. By contrast, two blind reports found the noiseless positive-frailty reset without any prior critique. These observations show what the paths surfaced in this run. Three repetitions per path do not establish that one review format is generally superior.

The same starting critique went into each same-context / fresh-handoff pair. Several fresh handoffs followed the older concern about the positive control into the raw workbook. The continuing chains supplied a different kind of evidence: all three carried the original paper, the previously evaluated edition and the present edition through one conversation. Their final answers distinguished the corrected post-wave equation and weekly mathematics from the unresolved zero-offset fallback and underidentified empirical fits. One chain also tested how a nonlinear wave changes the frailty distribution used by a later quiet-window fit. The [three chain histories](README.md#answers-by-repetition) preserve the actual intermediate answers. The targeted judgments below were solicited after each blind answer and contribute zero to the spontaneous counts above.

## The three latest edits

All three blind reviewers later received the same targeted list, after their open answers were saved. All three explicitly judged the three **disclosure objections addressed**. The revised multiwave example identifies expected deaths and a fixed, known baseline rate. Figure 1 and the historical grid identify their actual earlier procedures. The deterministic Poisson pilot has been removed as independent-dataset coverage evidence. These statements narrow the claims that readers can draw from the old experiments. They do not supply a stochastic parameter-recovery study or coverage calibration for the current production procedure. [Targeted 1](repeat-1/blind-targeted.md) · [Targeted 2](repeat-2/blind-targeted.md) · [Targeted 3](repeat-3/blind-targeted.md)

## Which earlier changes held up?

| Specific objection | What was in the previously evaluated edition | Current disposition and remaining scope |
|---|---|---|
| N1 · Simulator-supplied frailty and Cox error bars | The manuscript said frailty was estimated; the generator passed its true value. | The checkpoint already disclosed the known variances. Iteration 2 added genuine pointwise intervals. All seven include one. The historical grid Makefile still calls the earlier plotting program and can replace the new image if run. A blinded fitted-variance comparison is a different experiment. |
| N2 · Week-long observation versus point hazard | The weekly reconstruction was corrected, but the seed and pooled fit still predicted point hazards. | The integrated weekly objective is implemented in the core fitter. A residual left-endpoint approximation in the optional wave reference is quantified in the checks. |
| N3 · Final gap week | Sampling the start of the last gap week omitted that week's deaths from the offset. | The boundary now includes that week. Applying a zero-variance fallback after calculating a nonpositive offset remains a separate problem, reproduced in [3.1](../../../3.1/README.md). |
| N4 · First wave week | The first end-of-week total entered the reference unchanged. | The first wave week now contributes to excess when indicated. The in-wave exposure approximation and the source of the chosen exponent remain separate issues. |
| N5 · Scan fitter and denominator | The legacy fit and exclusion of failures inflated the reported pass rate. | The revised scan uses the joint weekly fitter and all 78 attempts. All 78 passed. Moving the window can still move the estimated contrast; see the [matched window replay](window-contrast.md). |
| N6 · Positive-control intervention | The stated individual-level intervention differed from the generator. | The cited experiment now correctly describes its cohort-level injection. Whether Table 9 demonstrates the fitted procedure remains open. Both endpoints and the failing command were independently reproduced in [3.3](../../../3.3/README.md). |
| N7 · Contamination threshold | The no-addition group was already above one in 15/20 runs. | Open. Restricting the comparison to strictly positive tested doses can reproduce the stated grid arithmetic. It cannot establish a first reversal from the uncontaminated situation. |
| N8 · One reference week versus a mean | Two descriptions conflicted. | The single-week formula is now explicit. An adjacent-week convention remains across scripts: with a two-week skip and four accumulating weeks, the workbook uses raw zero-based index 5, while the contamination script uses 6. Its effect on that sweep was not measured in this review. |
| N9 · Fitting and pooling | The manuscript implied no separate subgroup fits even for the Table 8 calculation. | The revised manuscript and Table 8 implementation agree on subgroup-by-pseudo-dose fits before pooling. Decisions on reporting flagged fits and approval of the default fitter remain outstanding. |
| N10 · Selection of empirical examples | Figure 1's description obscured selection on the result. | The final-KCOR filter and curvature preference are disclosed. The example illustrates the earlier procedure. Its selection limits its value as independent validation. |

The corrected proportional-wave identity, conditional post-wave survival equation, and fixed-slope identification formula also survive re-review within their stated assumptions. Estimating the nonlinear wave contribution during the wave, deciding how to report a forced exponent and insufficient-signal fits, and interpreting repeated intervals from different estimator versions remain open. These items retain their previous [decision history](../../decisions.md). The completed [Table 8 and archived-study checks](checks.md) identify which quantities reproduced.

## Findings to discuss individually

1. **[3.1 · A no-shock gap can reset positive frailty to zero](../../../3.1/README.md).** Two of three blind reviewers independently demonstrated this on exact data. The zero-offset guard predates the current revision and earlier reviewers had already identified the fallback. The new tests show its consequence even with no sampling noise. With true variance 4, one quiet window recovered 4. On adding another quiet window, a reconstructed offset numerically indistinguishable from zero could trigger an applied variance of 0. In the reproduced two-cohort example the true ratio is 1, the unanchored endpoint is 0.7402 and the anchored endpoint is 0.7521. This does not show that every low-frailty problem or every Table 8 flag has this cause.
2. **[3.2 · Prior exposure can change the baseline rate recovered in the first quiet window](../../../3.2/README.md).** The exact weekly equation can fit equally well at an effective rate when a cumulative shock is already present. This can affect the joint seed even with perfectly observed hazards. The mathematical condition has been verified; its effect on the Czech endpoints is unknown.
3. **[3.3 · Fitted provenance of the Table 9 positive control](../../../3.3/README.md).** The raw calculation yields 1.106984 for harm and 0.824596 for benefit. The documented fitter fails on the scenario sheet names before fitting. Both findings trace across all four editions. They establish a reproducibility gap. The historical execution cannot be reconstructed from rounding agreement alone.

The latest evaluation also measures the consequence of existing issues. At the published Table 8 reporting date, the 2022 1/0 point changed from **1.0462 to 1.5765** when the final refit used the paper's quiet-only mask in place of the executable's quiet-plus-anchor mask. In a separate sensitivity check it changed to **1.3412** when only quiet weeks through that reporting date were available. The two interventions are distinct. They produce point estimates with flagged fits, and neither selects the interpretation to use in the paper. [Paired reruns](checks.md)
