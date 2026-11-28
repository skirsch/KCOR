# Round 7 - independent evaluation of m128

## Protocol

Two fresh GPT-6.2 Astra High blind reviews and a separate Astra High targeted resolution review. The original question was used verbatim: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" Blind reviewers saw only m128. The resolution reviewer considered m117, prior round-6 records and historical provenance as needed. The third reviewer used High instead of the previous Medium. No full simulation/bootstrap reruns, candidate changes or PR updates. All 1,192 frozen files were verified unchanged.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Resolution](resolution.md) | [Frozen manifest](inputs.json) | [Input verification](input-verification.json) | [Clean paper](KCOR-clean.pdf)

## Convergence

All three reviewers affirm the proportional gamma inversion and the fixed-slope identification result under stated model assumptions. Eleven of fourteen specific round-6 corrections are independently supported within their original scope. Three remain partially resolved: 6.3 diagnostic claims; 6.13 historical alpha reporting; 6.14 cross-application eligibility provenance. The blind reviews find six genuinely new issues, plus additional specific material under five returning issues. General model conditionality, coverage, low-information frailty resolution, unmarked nonlinear exposure and alpha identification are not newly counted as technical defects.

## Numbered findings (six new; five returning)

| ID | Severity | Finding | Origin | Review |
|---|---|---|---|---|
| [7.1](../../../objections/7.1/README.md) | High | Unreported normalization fallback after numerical failure | New implementation defect | B |
| [7.2](../../../objections/7.2/README.md) | High | ISO enrollment and missing-week calendar handling | New implementation defect | A, B |
| [7.3](../../../objections/7.3/README.md) | High for low-rate fitting | Absolute solver tolerance causes incorrect frailty on noiseless rare-event data | New numerical defect | B |
| [7.4](../../../objections/7.4/README.md) | Moderate to high for affected workbook row | Double-counted overlapping age-shift aggregate | New layout-specific defect | B |
| [7.5](../../../objections/7.5/README.md) | High for empirical provenance | Archived age-shift control and current restricted Czech CMR are different populations | Returning 6.14, new proven source mismatch | R |
| [7.6](../../../objections/7.6/README.md) | High for validation claim | Czech diagnostic pass overstates the actual checks | Returning 6.3, new window-omission evidence | A, B, R |
| [7.7](../../../objections/7.7/README.md) | Moderate for empirical figure provenance | Supplementary Figure S10 uses an unmatched trajectory and time range | New reporting/reproducibility defect | B |
| [7.8](../../../objections/7.8/README.md) | Moderate | Paper build and archived-study reproducibility claims | New documentation/evidence defect | B |
| [7.9](../../../objections/7.9/README.md) | Moderate for nonquiet early fitting anchors | Fitting includes early anchor rows outside designated quiet windows | Returning older assumption, independently quantified | A |
| [7.10](../../../objections/7.10/README.md) | Moderate for optional alpha reporting | Historical alpha diagnostics named as current | Returning 6.13, remaining wording | R |
| [7.11](../../../objections/7.11/README.md) | Moderate for estimand framing | Remaining abstract and comparison-table bias/conditioning language | Returning 6.7, additional passages | B |

## Prior-round resolution

Closed within original scope: **6.1, 6.2, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12**. New layout-specific aggregate defect 7.4 is separate from repaired ordinary all-ages fit 6.1; additional comparator language 7.11 does not undo 6.7's reproduced results. The original 6.8 age-shift curves and bands match archived data; input provenance is separately 7.5.

Partly resolved: **6.3 to 7.6**, **6.13 to 7.10**, **6.14 to 7.5**. The earlier 3.2/4.5 early-row assumption resurfaces as 7.9.

## Older unresolved scientific questions (not new objections)

- 1.1: baseline/Gompertz and gamma law conditionality, acknowledged.
- 1.3 / 4.6 / 5.11: general current-estimator coverage, fallback operating characteristics, secondary analytical intervals. Existing coverage studies have different regimes or earlier fitters; archived Table 8 percentile arithmetic checks out.
- 1.4: age-shift control target is conditional, not a known numerical null; now disclosed.
- 2.7: low-information frailty resolution; the new optimizer defect 7.3 is a distinct numerical issue, not the statistical information limitation.
- 3.2 / 4.3: unmarked nonlinear selection and wave attribution conventions; supplied-alpha inversion now numerically checked.
- 4.7 / 5.10 / 6.13: weekly alpha localization not validated; deferred.
- 6.14: sensitivity to the intentional retrospective Czech product eligibility rule remains an application-level question. The source mismatch is 7.5, not a new requirement to redesign eligibility.

## Recommendation

Discuss 7.1-7.4 first, because they represent actual pipeline numerical/implementation defects. Then address provenance and reporting (7.5-7.8), and remaining assumptions/wording (7.9-7.11). No edits authorized by this evaluation alone. The 24-draw quadrature check and all prior archive intervals remain as labeled; new point-only diagnostic replays are not replacement CIs.
